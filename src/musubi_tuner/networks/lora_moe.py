"""
LoRA-MoE (LoRA Mixture of Experts) implementation for WAN2.2
Specialized for surgical instrument adaptation with low-data constraints.

Architecture:
- Base LoRA: Shared general adaptation
- Expert LoRAs: Instrument-specific adaptations (Scissors, Hook/Electrocautery, Suction, Other)
- Router: Instrument-aware soft gating mechanism
- Timestep-selective expert application
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
import math


class LoRAMoEModule(nn.Module):
    """
    Single LoRA-MoE module that replaces a linear layer with:
    - Frozen base weight W
    - Shared base LoRA adaptation
    - Multiple expert LoRA adaptations
    - Gated routing mechanism

    Forward: y = W(x) + α_base * ΔW_base(x) + Σ_e g_e * α_e * ΔW_e(x)
    """

    def __init__(
        self,
        org_module: nn.Module,
        lora_dim: int = 4,
        alpha: float = 1.0,
        num_experts: int = 4,
        expert_names: List[str] = None,
        dropout: float = 0.0,
        rank_dropout: float = 0.0,
        module_dropout: float = 0.0,
        use_base_lora: bool = True,
        timestep_selective: bool = True,
        timestep_bands: Tuple[float, float, float] = (0.0, 0.3, 1.0),  # (early, mid, late)
    ):
        super().__init__()

        self.lora_dim = lora_dim
        self.num_experts = num_experts
        self.expert_names = expert_names or [f"expert_{i}" for i in range(num_experts)]
        self.use_base_lora = use_base_lora
        self.timestep_selective = timestep_selective
        self.timestep_bands = timestep_bands

        # Store original module
        self.org_module = org_module
        self.org_forward = self.org_module.forward

        # Get dimensions
        if isinstance(org_module, nn.Linear):
            self.in_dim = org_module.in_features
            self.out_dim = org_module.out_features
        elif isinstance(org_module, nn.Conv2d):
            self.in_dim = org_module.in_channels
            self.out_dim = org_module.out_channels
        else:
            raise ValueError(f"Unsupported module type: {type(org_module)}")

        # Scaling factor
        self.scale = alpha / lora_dim

        # Dropout layers
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

        # Base LoRA (shared general adaptation)
        if use_base_lora:
            self.base_lora_down = nn.Linear(self.in_dim, lora_dim, bias=False)
            self.base_lora_up = nn.Linear(lora_dim, self.out_dim, bias=False)
            # Initialize: kaiming for down, zeros for up (standard LoRA init)
            nn.init.kaiming_uniform_(self.base_lora_down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.base_lora_up.weight)
        else:
            self.base_lora_down = None
            self.base_lora_up = None

        # Expert LoRAs
        self.expert_lora_down = nn.ModuleList([
            nn.Linear(self.in_dim, lora_dim, bias=False)
            for _ in range(num_experts)
        ])
        self.expert_lora_up = nn.ModuleList([
            nn.Linear(lora_dim, self.out_dim, bias=False)
            for _ in range(num_experts)
        ])

        # Initialize expert LoRAs
        for i in range(num_experts):
            nn.init.kaiming_uniform_(self.expert_lora_down[i].weight, a=math.sqrt(5))
            nn.init.zeros_(self.expert_lora_up[i].weight)

        # Timestep gate (learnable 3-bin classifier: early/mid/late)
        if timestep_selective:
            self.timestep_gate = nn.Linear(1, 3, bias=True)
            nn.init.zeros_(self.timestep_gate.weight)
            # Initialize bias to favor mid/late timesteps
            with torch.no_grad():
                bias = self.timestep_gate.bias
                bias.copy_(torch.tensor([0.0, 1.0, 2.0], dtype=bias.dtype, device=bias.device))
        else:
            self.timestep_gate = None

        # Cache for routing gates (set externally by router)
        self.register_buffer('current_gates', torch.ones(num_experts) / num_experts)
        self.register_buffer('current_timestep', torch.tensor(0.5))

    def set_gates(self, gates: torch.Tensor):
        """Set routing gates from external router. Gates should sum to 1."""
        assert gates.shape[-1] == self.num_experts, f"Gates shape {gates.shape} doesn't match num_experts {self.num_experts}"
        self.current_gates = gates.detach()

    def set_timestep(self, timestep: float):
        """Set current diffusion timestep (normalized to [0, 1])."""
        self.current_timestep = torch.tensor(timestep, device=self.current_gates.device)

    def compute_timestep_weight(self) -> float:
        """
        Compute timestep-dependent weight for expert application.
        Returns weight in [0, 1] based on current timestep.
        """
        if not self.timestep_selective or self.timestep_gate is None:
            return 1.0

        # Timestep normalized to [0, 1]
        t = self.current_timestep.view(1, 1)

        # 3-bin soft classification: early/mid/late
        # Bands: [0, 0.3], [0.3, 0.6], [0.6, 1.0]
        logits = self.timestep_gate(t)  # [1, 3]
        probs = F.softmax(logits, dim=-1)

        # Weight: 0 for early, 0.5 for mid, 1.0 for late
        weights = torch.tensor([0.0, 0.5, 1.0], device=probs.device)
        weight = (probs * weights).sum()

        return weight.item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA-MoE composition.

        Args:
            x: Input tensor [B, ..., in_dim]

        Returns:
            y: Output tensor [B, ..., out_dim]
        """
        # Original module output (frozen)
        org_out = self.org_forward(x)

        # Apply dropouts
        if self.training:
            if self.module_dropout > 0 and torch.rand(1).item() < self.module_dropout:
                return org_out

        # Base LoRA adaptation
        lora_out = 0.0
        if self.use_base_lora and self.base_lora_down is not None:
            base_x = x
            if self.dropout is not None:
                base_x = self.dropout(base_x)

            # Rank dropout
            if self.training and self.rank_dropout > 0:
                mask = torch.rand(self.lora_dim, device=x.device) > self.rank_dropout
                base_down = self.base_lora_down(base_x) * mask
            else:
                base_down = self.base_lora_down(base_x)

            base_up = self.base_lora_up(base_down)
            lora_out = lora_out + self.scale * base_up

        # Expert LoRA mixture
        # Gates: [num_experts] or [B, num_experts]
        gates = self.current_gates
        if gates.dim() == 1:
            gates = gates.unsqueeze(0)  # [1, num_experts]

        # Timestep weight
        timestep_weight = self.compute_timestep_weight()

        # Compute expert outputs
        expert_out = 0.0
        for i in range(self.num_experts):
            if gates[..., i].abs().max() < 1e-6:
                continue  # Skip if gate is near zero

            expert_x = x
            if self.dropout is not None:
                expert_x = self.dropout(expert_x)

            # Rank dropout
            if self.training and self.rank_dropout > 0:
                mask = torch.rand(self.lora_dim, device=x.device) > self.rank_dropout
                expert_down = self.expert_lora_down[i](expert_x) * mask
            else:
                expert_down = self.expert_lora_down[i](expert_x)

            expert_up = self.expert_lora_up[i](expert_down)

            # Apply gate (broadcast-safe)
            gate_weight = gates[..., i:i+1]  # Keep last dim for broadcasting
            expert_out = expert_out + gate_weight * self.scale * expert_up

        # Apply timestep weight
        expert_out = timestep_weight * expert_out

        return org_out + lora_out + expert_out


class InstrumentRouter(nn.Module):
    """
    Instrument-aware routing mechanism for LoRA-MoE.

    Supports:
    - Rule-based routing from classifier logits
    - Learnable MLP router with teacher guidance
    - Top-k routing with load balancing
    - Temporal EMA smoothing
    """

    INSTRUMENT_FAMILIES = {
        0: "Scissors",           # 剪刀
        1: "Hook/Electrocautery", # 电凝钩/双极电凝
        2: "Suction",            # 吸引
        3: "Other",              # 抓钳/戳卡/其他
    }

    def __init__(
        self,
        num_experts: int = 4,
        routing_mode: str = "rule_based",  # "rule_based" or "learned"
        top_k: int = 2,
        temperature: float = 0.7,
        ema_beta: float = 0.9,
        learnable_hidden_dim: int = 64,
        input_dim: int = 512,  # Dimension of classifier logits or text embeddings
        use_text_conditioning: bool = False,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.routing_mode = routing_mode
        self.top_k = min(top_k, num_experts)
        self.temperature = temperature
        self.ema_beta = ema_beta
        self.use_text_conditioning = use_text_conditioning

        # Learnable router (optional)
        if routing_mode == "learned":
            self.router_mlp = nn.Sequential(
                nn.Linear(input_dim, learnable_hidden_dim),
                nn.GELU(),
                nn.Linear(learnable_hidden_dim, num_experts),
            )
            # Initialize to small values
            for layer in self.router_mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=0.01)
                    nn.init.zeros_(layer.bias)
        else:
            self.router_mlp = None

        # EMA gates for temporal smoothing
        self.register_buffer('ema_gates', torch.ones(num_experts) / num_experts)

        # Load balancing statistics
        self.register_buffer('expert_usage_count', torch.zeros(num_experts))

    def rule_based_routing(
        self,
        instrument_logits: torch.Tensor,
        confidence_threshold: float = 0.3,
    ) -> torch.Tensor:
        """
        Rule-based routing from instrument classifier logits.

        Args:
            instrument_logits: Classifier logits [B, num_instruments] or [num_instruments]
            confidence_threshold: Minimum confidence to trust prediction

        Returns:
            gates: Soft routing gates [B, num_experts] or [num_experts]
        """
        # Map instrument logits to expert families
        # This mapping should align with your instrument detector
        # Example: [scissors, hook, electrocautery, suction, grasper, pushrod, ...]

        if instrument_logits.dim() == 1:
            instrument_logits = instrument_logits.unsqueeze(0)  # [1, num_instruments]

        B = instrument_logits.shape[0]
        device = instrument_logits.device

        # Softmax to get probabilities
        probs = F.softmax(instrument_logits / self.temperature, dim=-1)

        # Initialize expert gates
        expert_gates = torch.zeros(B, self.num_experts, device=device)

        # Simple mapping (customize based on your instrument classes)
        # Assumes instrument_logits has at least 4 classes mapping to our 4 experts
        if instrument_logits.shape[-1] >= self.num_experts:
            expert_gates = probs[:, :self.num_experts]
        else:
            # If fewer classes, distribute uniformly
            expert_gates[:, :instrument_logits.shape[-1]] = probs

        # Check confidence (max prob)
        max_prob = probs.max(dim=-1, keepdim=True)[0]
        low_confidence = max_prob < confidence_threshold

        # Fallback: distribute between Other (expert 3) and second-best
        if low_confidence.any():
            fallback_gates = torch.zeros_like(expert_gates)
            fallback_gates[:, 3] = 0.6  # Other expert
            # Second best gets remaining
            second_best = torch.argsort(expert_gates, dim=-1, descending=True)[:, 1]
            fallback_gates.scatter_(1, second_best.unsqueeze(1), 0.4)

            expert_gates = torch.where(
                low_confidence,
                fallback_gates,
                expert_gates
            )

        # Normalize
        expert_gates = expert_gates / (expert_gates.sum(dim=-1, keepdim=True) + 1e-8)

        return expert_gates.squeeze(0) if B == 1 else expert_gates

    def learned_routing(
        self,
        router_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Learned MLP routing.

        Args:
            router_input: Input features [B, input_dim] or [input_dim]

        Returns:
            gates: Soft routing gates [B, num_experts] or [num_experts]
        """
        if self.router_mlp is None:
            raise ValueError("Learned routing requires router_mlp to be initialized")

        if router_input.dim() == 1:
            router_input = router_input.unsqueeze(0)

        logits = self.router_mlp(router_input)
        gates = F.softmax(logits / self.temperature, dim=-1)

        return gates.squeeze(0) if router_input.shape[0] == 1 else gates

    def apply_topk(self, gates: torch.Tensor) -> torch.Tensor:
        """Apply top-k routing (zero out all but top-k experts)."""
        if self.top_k >= self.num_experts:
            return gates

        # Get top-k indices
        topk_vals, topk_indices = torch.topk(gates, self.top_k, dim=-1)

        # Create mask
        mask = torch.zeros_like(gates)
        mask.scatter_(-1, topk_indices, 1.0)

        # Apply mask and renormalize
        gates = gates * mask
        gates = gates / (gates.sum(dim=-1, keepdim=True) + 1e-8)

        return gates

    def apply_ema(self, gates: torch.Tensor) -> torch.Tensor:
        """Apply temporal EMA smoothing (for video frames)."""
        if not self.training:
            # During inference, smooth across time
            if gates.dim() == 1:
                self.ema_gates = self.ema_beta * self.ema_gates + (1 - self.ema_beta) * gates
                return self.ema_gates.clone()

        return gates

    def update_usage_stats(self, gates: torch.Tensor):
        """Update expert usage statistics for load balancing."""
        with torch.no_grad():
            # Average gates across batch
            if gates.dim() == 2:
                avg_gates = gates.mean(dim=0)
            else:
                avg_gates = gates

            self.expert_usage_count += avg_gates

    def compute_load_balance_loss(self, gates: torch.Tensor) -> torch.Tensor:
        """
        Compute load balancing auxiliary loss (Switch Transformer style).
        Encourages uniform expert usage over batch.
        """
        if gates.dim() == 1:
            gates = gates.unsqueeze(0)

        # Average gate per expert over batch
        avg_gates = gates.mean(dim=0)  # [num_experts]

        # Target: uniform distribution
        target = torch.ones_like(avg_gates) / self.num_experts

        # L2 loss
        loss = F.mse_loss(avg_gates, target)

        return loss

    def forward(
        self,
        instrument_logits: Optional[torch.Tensor] = None,
        text_embeddings: Optional[torch.Tensor] = None,
        apply_topk: bool = True,
        apply_ema: bool = False,
    ) -> torch.Tensor:
        """
        Forward routing.

        Args:
            instrument_logits: Classifier logits [B, num_instruments] or [num_instruments]
            text_embeddings: Text conditioning [B, dim] or [dim]
            apply_topk: Whether to apply top-k routing
            apply_ema: Whether to apply temporal EMA smoothing

        Returns:
            gates: Routing gates [B, num_experts] or [num_experts]
        """
        # Choose routing mode
        if self.routing_mode == "rule_based":
            if instrument_logits is None:
                raise ValueError("Rule-based routing requires instrument_logits")
            gates = self.rule_based_routing(instrument_logits)

        elif self.routing_mode == "learned":
            # Combine instrument logits and text embeddings
            router_input = []
            if instrument_logits is not None:
                router_input.append(instrument_logits)
            if text_embeddings is not None and self.use_text_conditioning:
                router_input.append(text_embeddings)

            if not router_input:
                raise ValueError("Learned routing requires at least one input")

            router_input = torch.cat(router_input, dim=-1)
            gates = self.learned_routing(router_input)

        else:
            raise ValueError(f"Unknown routing_mode: {self.routing_mode}")

        # Apply top-k
        if apply_topk:
            gates = self.apply_topk(gates)

        # Apply EMA smoothing
        if apply_ema:
            gates = self.apply_ema(gates)

        # Update usage stats
        self.update_usage_stats(gates)

        return gates


class LoRAMoENetwork:
    """
    Network manager for LoRA-MoE.
    Handles module replacement and routing coordination.
    """

    def __init__(
        self,
        model: nn.Module,
        target_modules: List[str],
        lora_dim: int = 4,
        alpha: float = 1.0,
        num_experts: int = 4,
        expert_names: List[str] = None,
        use_base_lora: bool = True,
        router_config: Optional[Dict] = None,
    ):
        self.model = model
        self.target_modules = target_modules
        self.lora_dim = lora_dim
        self.alpha = alpha
        self.num_experts = num_experts
        self.expert_names = expert_names
        self.use_base_lora = use_base_lora

        # Router
        router_config = router_config or {}
        self.router = InstrumentRouter(
            num_experts=num_experts,
            **router_config
        )

        # LoRA-MoE modules
        self.lora_moe_modules: List[LoRAMoEModule] = []

    def apply_lora_moe(self):
        """Replace target modules with LoRA-MoE modules."""
        # Implementation: traverse model and replace Linear layers
        # This follows the same pattern as lora_wan.py

        for name, module in self.model.named_modules():
            # Check if this is a target module
            if any(target in name for target in self.target_modules):
                # Replace Linear layers within this module
                for child_name, child_module in module.named_children():
                    if isinstance(child_module, nn.Linear):
                        # Create LoRA-MoE wrapper
                        lora_moe = LoRAMoEModule(
                            org_module=child_module,
                            lora_dim=self.lora_dim,
                            alpha=self.alpha,
                            num_experts=self.num_experts,
                            expert_names=self.expert_names,
                            use_base_lora=self.use_base_lora,
                        )

                        # Move LoRA-MoE to the same device/dtype as the original Linear
                        try:
                            dev = child_module.weight.device
                            dt = child_module.weight.dtype
                            lora_moe.to(device=dev, dtype=dt)
                        except Exception:
                            pass

                        # Replace module
                        setattr(module, child_name, lora_moe)
                        self.lora_moe_modules.append(lora_moe)

        print(f"Applied LoRA-MoE to {len(self.lora_moe_modules)} modules")

    def set_routing_gates(
        self,
        instrument_logits: Optional[torch.Tensor] = None,
        text_embeddings: Optional[torch.Tensor] = None,
    ):
        """Compute and set routing gates for all LoRA-MoE modules."""
        gates = self.router(
            instrument_logits=instrument_logits,
            text_embeddings=text_embeddings,
            apply_topk=True,
            apply_ema=not self.model.training,
        )

        # Set gates for all modules
        for module in self.lora_moe_modules:
            module.set_gates(gates)

    def set_timestep(self, timestep: float):
        """Set diffusion timestep for all LoRA-MoE modules."""
        for module in self.lora_moe_modules:
            module.set_timestep(timestep)

    def get_load_balance_loss(self) -> torch.Tensor:
        """Compute load balancing loss across all modules."""
        # Use router's current gates
        gates = self.router.ema_gates.unsqueeze(0)
        return self.router.compute_load_balance_loss(gates)
