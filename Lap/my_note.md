结论：Stage 2 保存的是 LoRA‑MoE 权重（专家 + router），不是融合后的 DiT。训练里如果存在 lora_moe_network，保存分支只写 LoRA‑MoE 权重；否则才会落回保存基模型权重。
训练保存位置（只保存 LoRA‑MoE 权重）的代码依据：
# If LoRA-MoE is active, save ONLY LoRA-MoE weights using the main ckpt filename.
try:
    if hasattr(self, "lora_moe_network") and self.lora_moe_network is not None:
        self.lora_moe_network.save_lora_moe_weights(ckpt_file)
    else:
        # Fallback: save base network weights (if any)
        unwrapped_nw.save_weights(ckpt_file, save_dtype, metadata_to_save)
except Exception as e:
    logger.warning(f"Failed to save checkpoint: {e}")


总损失的路由项来源于 RoutingRegularizationLoss.total，其在实现中确实为三项之和（加权后）：entropy_loss + load_balance_loss + teacher_kl_loss。
代码位置：
RoutingRegularizationLoss.forward 里组装三项并在 L472 处 losses["total"] = sum(losses.values())
LoRAMoECombinedLoss.forward 在 L586 处把 routing_losses["total"] 加入 total_loss
因此指标里 routing_total 不应只等于 KL，除非 entropy_weight 和 load_balance_weight 太小，或这两项数值本身接近 0。