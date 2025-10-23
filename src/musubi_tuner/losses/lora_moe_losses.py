"""
Training objectives for LoRA-MoE surgical video generation.

Implements:
1. ROI-weighted reconstruction loss (focus on tip regions)
2. Identity preservation loss (instrument family classification)
3. Temporal consistency loss (flow-warped stability)
4. Routing regularization (entropy + load balancing)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import math


class ROIWeightedLoss(nn.Module):
    """
    ROI-weighted reconstruction loss.
    Emphasizes loss inside surgical tip regions of interest.
    """

    def __init__(
        self,
        base_weight: float = 1.0,
        roi_weight: float = 3.0,
        roi_blur_sigma: float = 2.0,
    ):
        """
        Args:
            base_weight: Weight for background regions
            roi_weight: Weight multiplier for ROI regions (typically 3-5x)
            roi_blur_sigma: Gaussian blur sigma for smooth ROI boundaries
        """
        super().__init__()
        self.base_weight = base_weight
        self.roi_weight = roi_weight
        self.roi_blur_sigma = roi_blur_sigma

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        roi_mask: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute ROI-weighted MSE loss.

        Args:
            pred: Predicted noise/latent [B, C, F, H, W]
            target: Target noise/latent [B, C, F, H, W]
            roi_mask: Binary ROI mask [B, 1, F, H, W] where 1=ROI, 0=background
                     If None, uses uniform weighting
            reduction: 'mean' or 'none'

        Returns:
            loss: Weighted MSE loss
        """
        # Base MSE loss (no reduction yet)
        mse_loss = F.mse_loss(pred, target, reduction="none")  # [B, C, F, H, W]

        # Weight map
        if roi_mask is None:
            # Uniform weighting
            weight_map = torch.ones_like(mse_loss[:, :1])  # [B, 1, F, H, W]
        else:
            # ROI-weighted: base_weight for background, roi_weight for ROI
            roi_mask = roi_mask.float()

            # Optional: smooth ROI boundaries with Gaussian blur
            if self.roi_blur_sigma > 0:
                roi_mask = self._gaussian_blur_3d(roi_mask, sigma=self.roi_blur_sigma)

            weight_map = self.base_weight + (self.roi_weight - self.base_weight) * roi_mask

        # Apply weights
        weighted_loss = mse_loss * weight_map

        # Reduce
        if reduction == "mean":
            return weighted_loss.mean()
        elif reduction == "sum":
            return weighted_loss.sum()
        elif reduction == "none":
            return weighted_loss
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

    def _gaussian_blur_3d(
        self,
        x: torch.Tensor,
        sigma: float = 2.0,
        kernel_size: int = 5,
    ) -> torch.Tensor:
        """
        Apply 3D Gaussian blur to smooth ROI mask.

        Args:
            x: Input tensor [B, 1, F, H, W]
            sigma: Gaussian sigma
            kernel_size: Kernel size (must be odd)

        Returns:
            blurred: Blurred tensor [B, 1, F, H, W]
        """
        # Create 3D Gaussian kernel
        if kernel_size % 2 == 0:
            kernel_size += 1

        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=x.device)
        xx, yy, zz = torch.meshgrid(ax, ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, kernel_size, kernel_size)

        # Apply conv3d
        padding = kernel_size // 2
        blurred = F.conv3d(x, kernel, padding=padding)

        return blurred


class InstrumentIdentityLoss(nn.Module):
    """
    Identity preservation loss.
    Ensures generated tip regions maintain instrument family identity.

    Uses a frozen instrument classifier to compute cross-entropy loss
    on predicted instrument class from generated frames.
    """

    def __init__(
        self,
        classifier: Optional[nn.Module] = None,
        roi_weight: float = 1.0,
        background_weight: float = 0.25,
        num_classes: int = 4,
    ):
        """
        Args:
            classifier: Frozen instrument classifier model
                       Input: [B, C, H, W] (single frame or crop)
                       Output: [B, num_classes] logits
            roi_weight: Weight for ROI regions
            background_weight: Weight for background regions
            num_classes: Number of instrument classes (default: 4 experts)
        """
        super().__init__()
        self.classifier = classifier
        self.roi_weight = roi_weight
        self.background_weight = background_weight
        self.num_classes = num_classes

        # Freeze classifier
        if self.classifier is not None:
            for param in self.classifier.parameters():
                param.requires_grad = False
            self.classifier.eval()

    def forward(
        self,
        generated_frames: torch.Tensor,
        target_instrument_labels: torch.Tensor,
        roi_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute identity preservation loss.

        Args:
            generated_frames: Generated video frames [B, C, F, H, W] (in pixel space)
            target_instrument_labels: Target instrument class indices [B] or [B, F]
            roi_mask: ROI mask [B, 1, F, H, W]

        Returns:
            loss: Cross-entropy loss weighted by ROI
        """
        if self.classifier is None:
            # No classifier available, return zero loss
            return torch.tensor(0.0, device=generated_frames.device)

        B, C, F, H, W = generated_frames.shape

        # Flatten temporal dimension
        frames_flat = generated_frames.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)  # [B*F, C, H, W]

        # Extract ROI crops if mask is provided
        if roi_mask is not None:
            # For simplicity, use full frames but weight the loss
            # In practice, you could extract actual crops here
            pass

        # Run classifier
        with torch.no_grad():
            logits = self.classifier(frames_flat)  # [B*F, num_classes]

        logits = logits.view(B, F, self.num_classes)  # [B, F, num_classes]

        # Expand target labels if needed
        if target_instrument_labels.dim() == 1:
            target_instrument_labels = target_instrument_labels.unsqueeze(1).expand(B, F)  # [B, F]

        # Flatten
        logits_flat = logits.reshape(B * F, self.num_classes)
        targets_flat = target_instrument_labels.reshape(B * F)

        # Cross-entropy loss
        ce_loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")  # [B*F]
        ce_loss = ce_loss.view(B, F)  # [B, F]

        # Weight by ROI
        if roi_mask is not None:
            # Average ROI mask over spatial dimensions
            roi_weight_map = roi_mask.mean(dim=[1, 3, 4])  # [B, F]
            weight = self.background_weight + (self.roi_weight - self.background_weight) * roi_weight_map
        else:
            weight = torch.ones_like(ce_loss)

        # Weighted loss
        weighted_loss = (ce_loss * weight).mean()

        return weighted_loss


class TemporalConsistencyLoss(nn.Module):
    """
    Temporal consistency loss.
    Ensures smooth motion in tip regions using optical flow warping.
    """

    def __init__(
        self,
        flow_estimator: Optional[nn.Module] = None,
        consistency_metric: str = "l1",  # "l1" or "lpips"
        roi_only: bool = True,
    ):
        """
        Args:
            flow_estimator: Frozen optical flow model (e.g., RAFT)
                           Input: [B, 2*C, H, W] (concatenated frame pairs)
                           Output: [B, 2, H, W] (flow field)
            consistency_metric: Loss metric ("l1" or "lpips")
            roi_only: If True, compute loss only in ROI regions
        """
        super().__init__()
        self.flow_estimator = flow_estimator
        self.consistency_metric = consistency_metric
        self.roi_only = roi_only

        # Freeze flow estimator
        if self.flow_estimator is not None:
            for param in self.flow_estimator.parameters():
                param.requires_grad = False
            self.flow_estimator.eval()

        # LPIPS model (optional)
        self.lpips_model = None
        if consistency_metric == "lpips":
            try:
                import lpips
                self.lpips_model = lpips.LPIPS(net='alex')
                self.lpips_model.eval()
                for param in self.lpips_model.parameters():
                    param.requires_grad = False
            except ImportError:
                print("Warning: LPIPS not available, falling back to L1")
                self.consistency_metric = "l1"

    def forward(
        self,
        generated_frames: torch.Tensor,
        roi_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute temporal consistency loss.

        Args:
            generated_frames: Generated video frames [B, C, F, H, W]
            roi_mask: ROI mask [B, 1, F, H, W]

        Returns:
            loss: Temporal consistency loss
        """
        if generated_frames.shape[2] < 2:
            # Need at least 2 frames
            return torch.tensor(0.0, device=generated_frames.device)

        B, C, F, H, W = generated_frames.shape

        # Compute pairwise consistency between consecutive frames
        total_loss = 0.0
        num_pairs = 0

        for t in range(F - 1):
            frame_t = generated_frames[:, :, t]      # [B, C, H, W]
            frame_t1 = generated_frames[:, :, t+1]   # [B, C, H, W]

            # Estimate optical flow (if available)
            if self.flow_estimator is not None:
                with torch.no_grad():
                    flow = self._estimate_flow(frame_t, frame_t1)  # [B, 2, H, W]
                    frame_t_warped = self._warp_frame(frame_t, flow)
            else:
                # No flow, just compare adjacent frames directly
                frame_t_warped = frame_t

            # Compute difference
            if self.consistency_metric == "l1":
                diff = torch.abs(frame_t_warped - frame_t1)
            elif self.consistency_metric == "lpips":
                if self.lpips_model is not None:
                    diff = self.lpips_model(frame_t_warped, frame_t1)
                else:
                    diff = torch.abs(frame_t_warped - frame_t1)
            else:
                raise ValueError(f"Unknown consistency metric: {self.consistency_metric}")

            # Weight by ROI
            if roi_mask is not None and self.roi_only:
                roi_t = roi_mask[:, :, t]  # [B, 1, H, W]
                if diff.shape != roi_t.shape:
                    # LPIPS may return scalar, need to broadcast
                    loss_t = (diff * roi_t.mean(dim=[1, 2, 3])).mean()
                else:
                    loss_t = (diff * roi_t).mean()
            else:
                loss_t = diff.mean()

            total_loss = total_loss + loss_t
            num_pairs += 1

        return total_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0)

    def _estimate_flow(self, frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
        """Estimate optical flow between two frames."""
        # Concatenate frames
        frame_pair = torch.cat([frame1, frame2], dim=1)  # [B, 2*C, H, W]

        # Run flow estimator
        flow = self.flow_estimator(frame_pair)  # [B, 2, H, W]

        return flow

    def _warp_frame(self, frame: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Warp frame using optical flow."""
        B, C, H, W = frame.shape

        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=frame.device, dtype=torch.float32),
            torch.arange(W, device=frame.device, dtype=torch.float32),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0)  # [2, H, W]
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, 2, H, W]

        # Add flow
        flow_grid = grid + flow

        # Normalize to [-1, 1]
        flow_grid[:, 0] = 2.0 * flow_grid[:, 0] / (W - 1) - 1.0
        flow_grid[:, 1] = 2.0 * flow_grid[:, 1] / (H - 1) - 1.0

        # Permute to [B, H, W, 2]
        flow_grid = flow_grid.permute(0, 2, 3, 1)

        # Warp
        warped = F.grid_sample(
            frame,
            flow_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )

        return warped


class RoutingRegularizationLoss(nn.Module):
    """
    Routing regularization losses.

    Includes:
    1. Entropy regularization: Prevents overly peaked gates early in training
    2. Load balancing: Encourages uniform expert usage
    3. Teacher forcing: KL divergence for learned router (optional)
    """

    def __init__(
        self,
        entropy_weight: float = 0.01,
        load_balance_weight: float = 0.05,
        teacher_kl_weight: float = 1.0,
        target_entropy: Optional[float] = None,
        num_experts: int = 4,
    ):
        """
        Args:
            entropy_weight: Weight for entropy penalty
            load_balance_weight: Weight for load balancing loss
            teacher_kl_weight: Weight for KL divergence to teacher gates
            target_entropy: Target entropy (in nats). If None, uses log(num_experts)
            num_experts: Number of experts
        """
        super().__init__()
        self.entropy_weight = entropy_weight
        self.load_balance_weight = load_balance_weight
        self.teacher_kl_weight = teacher_kl_weight
        self.num_experts = num_experts

        # Target entropy (max entropy = uniform distribution)
        if target_entropy is None:
            self.target_entropy = math.log(num_experts)
        else:
            self.target_entropy = target_entropy

    def forward(
        self,
        gates: torch.Tensor,
        teacher_gates: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute routing regularization losses.

        Args:
            gates: Current routing gates [B, num_experts] or [num_experts]
            teacher_gates: Teacher gates for KL divergence (optional) [B, num_experts]

        Returns:
            losses: Dict with keys:
                - entropy_loss: Entropy penalty
                - load_balance_loss: Load balancing loss
                - teacher_kl_loss: KL divergence to teacher (if provided)
                - total: Sum of all losses
        """
        if gates.dim() == 1:
            gates = gates.unsqueeze(0)  # [1, num_experts]

        losses = {}

        # 1. Entropy regularization
        # We want to maximize entropy (minimize peaky distributions) early in training
        # Loss = -entropy (we minimize, so this encourages high entropy)
        entropy = -torch.sum(gates * torch.log(gates + 1e-8), dim=-1).mean()
        entropy_penalty = self.target_entropy - entropy  # Positive when entropy is too low
        losses["entropy_loss"] = self.entropy_weight * F.relu(entropy_penalty)

        # 2. Load balancing loss
        # Encourage average gate per expert to be uniform
        avg_gates = gates.mean(dim=0)  # [num_experts]
        target_avg = torch.ones_like(avg_gates) / self.num_experts
        load_balance_loss = F.mse_loss(avg_gates, target_avg)
        losses["load_balance_loss"] = self.load_balance_weight * load_balance_loss

        # 3. Teacher KL divergence (for learned router)
        if teacher_gates is not None:
            if teacher_gates.dim() == 1:
                teacher_gates = teacher_gates.unsqueeze(0)

            # KL(student || teacher)
            kl_div = F.kl_div(
                torch.log(gates + 1e-8),
                teacher_gates,
                reduction="batchmean",
                log_target=False
            )
            losses["teacher_kl_loss"] = self.teacher_kl_weight * kl_div
        else:
            losses["teacher_kl_loss"] = torch.tensor(0.0, device=gates.device)

        # Total
        losses["total"] = sum(losses.values())

        return losses


class LoRAMoECombinedLoss(nn.Module):
    """
    Combined loss for LoRA-MoE training.

    Combines:
    - Base diffusion loss (MSE)
    - ROI-weighted reconstruction
    - Identity preservation
    - Temporal consistency
    - Routing regularization
    """

    def __init__(
        self,
        base_diffusion_weight: float = 1.0,
        roi_recon_weight: float = 3.0,
        identity_weight: float = 0.5,
        temporal_weight: float = 0.5,
        routing_entropy_weight: float = 0.01,
        routing_load_balance_weight: float = 0.05,
        roi_loss: Optional[ROIWeightedLoss] = None,
        identity_loss: Optional[InstrumentIdentityLoss] = None,
        temporal_loss: Optional[TemporalConsistencyLoss] = None,
        routing_loss: Optional[RoutingRegularizationLoss] = None,
    ):
        super().__init__()

        self.base_diffusion_weight = base_diffusion_weight
        self.roi_recon_weight = roi_recon_weight
        self.identity_weight = identity_weight
        self.temporal_weight = temporal_weight

        # Loss modules
        self.roi_loss = roi_loss or ROIWeightedLoss(roi_weight=roi_recon_weight)
        self.identity_loss = identity_loss
        self.temporal_loss = temporal_loss
        self.routing_loss = routing_loss or RoutingRegularizationLoss(
            entropy_weight=routing_entropy_weight,
            load_balance_weight=routing_load_balance_weight,
        )

    def forward(
        self,
        model_pred: torch.Tensor,
        target: torch.Tensor,
        generated_frames: Optional[torch.Tensor] = None,
        roi_mask: Optional[torch.Tensor] = None,
        instrument_labels: Optional[torch.Tensor] = None,
        routing_gates: Optional[torch.Tensor] = None,
        teacher_gates: Optional[torch.Tensor] = None,
        stage: str = "stage_a",  # Training stage
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.

        Args:
            model_pred: Model prediction (noise) [B, C, F, H, W]
            target: Target (noise) [B, C, F, H, W]
            generated_frames: Generated frames in pixel space [B, C, F, H, W] (optional)
            roi_mask: ROI mask [B, 1, F, H, W] (optional)
            instrument_labels: Instrument class labels [B] (optional)
            routing_gates: Current routing gates [B, num_experts] (optional)
            teacher_gates: Teacher routing gates [B, num_experts] (optional)
            stage: Training stage ("stage_a", "stage_b", "stage_c")

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual loss components
        """
        loss_dict = {}

        # 1. Base diffusion loss (always computed)
        base_loss = F.mse_loss(model_pred, target, reduction="mean")
        loss_dict["base_diffusion"] = base_loss.item()
        total_loss = self.base_diffusion_weight * base_loss

        # 2. ROI-weighted reconstruction (if mask available)
        if roi_mask is not None and self.roi_recon_weight > 0:
            roi_loss = self.roi_loss(model_pred, target, roi_mask)
            loss_dict["roi_recon"] = roi_loss.item()
            total_loss = total_loss + self.roi_recon_weight * roi_loss

        # 3. Identity preservation (if classifier and labels available)
        if (self.identity_loss is not None and
            generated_frames is not None and
            instrument_labels is not None and
            self.identity_weight > 0):
            identity_loss = self.identity_loss(
                generated_frames,
                instrument_labels,
                roi_mask
            )
            loss_dict["identity"] = identity_loss.item()
            total_loss = total_loss + self.identity_weight * identity_loss

        # 4. Temporal consistency (if frames available)
        if (self.temporal_loss is not None and
            generated_frames is not None and
            self.temporal_weight > 0):
            temporal_loss = self.temporal_loss(generated_frames, roi_mask)
            loss_dict["temporal"] = temporal_loss.item()
            total_loss = total_loss + self.temporal_weight * temporal_loss

        # 5. Routing regularization (only in stage_b and stage_c)
        if routing_gates is not None and stage in ["stage_b", "stage_c"]:
            routing_losses = self.routing_loss(routing_gates, teacher_gates)
            for key, val in routing_losses.items():
                loss_dict[f"routing_{key}"] = val.item() if isinstance(val, torch.Tensor) else val
            total_loss = total_loss + routing_losses["total"]

        loss_dict["total"] = total_loss.item()

        return total_loss, loss_dict
