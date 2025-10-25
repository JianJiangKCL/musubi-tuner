# WAN 2.2 训练代码详解

本文档详细说明 vanilla WAN 2.2 的训练代码实现，包括文件调用关系、高低噪切换逻辑以及 HV train network 的实现。

## 目录
1. [核心文件结构](#核心文件结构)
2. [代码调用流程](#代码调用流程)
3. [高低噪切换逻辑](#高低噪切换逻辑)
4. [HV Train Network 实现](#hv-train-network-实现)
5. [关键配置参数](#关键配置参数)
6. [重要实现细节](#重要实现细节)

---

## 核心文件结构

### 项目文件组织

```
/home/user/musubi-tuner/
├── wan_train_network.py                    # 训练入口脚本（封装）
├── hv_train_network.py                     # HunyuanVideo 训练入口
├── diffusers_lora_wan2-2.py               # 推理/采样脚本
└── src/musubi_tuner/
    ├── wan_train_network.py                # WAN 训练核心实现 (738 行)
    ├── hv_train_network.py                 # NetworkTrainer 基类实现
    ├── wan/
    │   ├── configs/                        # 配置文件目录
    │   │   ├── shared_config.py           # 通用 WAN 配置
    │   │   ├── wan_t2v_14B.py             # WAN 2.1 T2V 配置
    │   │   ├── wan_t2v_A14B.py            # WAN 2.2 T2V 配置 (boundary: 0.875)
    │   │   ├── wan_i2v_14B.py             # WAN 2.1 I2V 配置
    │   │   └── wan_i2v_A14B.py            # WAN 2.2 I2V 配置 (boundary: 0.900)
    │   └── modules/                        # 模型模块
    │       ├── model.py                   # WAN 模型加载
    │       ├── t5.py                      # T5 文本编码器
    │       ├── clip.py                    # CLIP 编码器（I2V）
    │       └── vae.py                     # VAE 编码/解码器
    └── networks/
        └── lora_wan.py                     # WAN 的 LoRA 网络模块
```

### 主要类和文件说明

#### 1. **wan_train_network.py** (入口脚本)
- 简单的封装脚本
- 调用 `src.musubi_tuner.wan_train_network` 中的 `main()` 函数

#### 2. **src/musubi_tuner/wan_train_network.py** (核心实现)
- **`WanNetworkTrainer` 类**：继承自 `NetworkTrainer`
- 实现所有 WAN 2.2 特定的训练逻辑
- 包含高低噪模型切换机制
- 支持 T2V（文本到视频）和 I2V（图像到视频）模式

#### 3. **src/musubi_tuner/hv_train_network.py** (基础训练器)
- **`NetworkTrainer` 基类**：从第 375 行开始
- **`FineTuningTrainer` 类**：HunyuanVideo 微调
- `train()` 方法：第 1610 行，核心训练循环
- 提供通用的训练工具函数

---

## 代码调用流程

### 完整的训练流程图

```
入口: wan_train_network.py (line 738)
    └── main()
        └── WanNetworkTrainer().train(args)
            │
            ├─┬ 步骤 1: handle_model_specific_args(args)
            │ ├── 根据任务加载 WAN 配置 (t2v-14B, i2v-A14B, 等)
            │ ├── 检测和验证 DiT dtype
            │ └── 如果提供 --dit_high_noise，设置高低噪训练
            │
            ├─┬ 步骤 2: prepare_accelerator(args)
            │ └── 设置分布式训练环境
            │
            ├─┬ 步骤 3: process_sample_prompts(args, accelerator, sample_prompts)
            │ ├── 加载 T5 文本编码器
            │ ├── 编码提示词到 embeddings
            │ ├── (I2V 模式) 加载 CLIP 并编码图像
            │ └── 缓存所有 embeddings 用于采样
            │
            ├─┬ 步骤 4: load_vae(args, vae_dtype, vae_path)
            │ └── 加载 WanVAE 进行 VAE 编码
            │
            ├─┬ 步骤 5: load_transformer(accelerator, args, dit_path, ...)
            │ ├── load_wan_model() - 加载低噪模型
            │ └── 如果 high_low_training:
            │     └── load_wan_model() - 加载高噪模型
            │
            ├─┬ 步骤 6: 创建 LoRA 网络（如果指定 network_dim）
            │ └── lora_wan.create_arch_network()
            │
            ├─┬ 步骤 7: 设置优化器和调度器
            │ ├── get_optimizer() - 创建优化器
            │ └── get_lr_scheduler() - 创建学习率调度器
            │
            └─┬ 步骤 8: 训练循环 (line 1700+)
              │
              对于每个 epoch:
                对于每个 batch:
                  │
                  ├─┬ 获取批次数据
                  │ └── 加载 latents, T5 embeddings, clip embeddings
                  │
                  ├─┬ get_noisy_model_input_and_timesteps()
                  │ ├── 从调度器采样时间步
                  │ ├── 如果 high_low_training:
                  │ │   └── 重新采样以匹配高低噪边界
                  │ └── 向 latents 添加噪声
                  │
                  ├─┬ call_dit()
                  │ ├── 如果 high_low_training:
                  │ │   └── swap_high_low_weights()  # 切换模型权重
                  │ └── _call_dit()
                  │     ├── 准备模型输入 (context, image_latents, 等)
                  │     └── WAN 模型前向传播
                  │
                  ├── 计算损失 (flow matching)
                  ├── 反向传播
                  ├── 优化器步进和调度器步进
                  │
                  └─┬ 如果需要采样:
                    └── do_inference()
                        ├── 如果 high_low_training:
                        │   └── swap_high_low_weights() 用于采样
                        ├── 用噪声初始化 latents
                        ├── 去噪循环 (反向时间步):
                        │   ├── 调用模型预测噪声
                        │   ├── 分类器无关引导 (如果启用)
                        │   └── 调度器步进
                        └── VAE 解码为视频
```

### 关键方法调用链

#### 模型加载
```
load_transformer() [wan_train_network.py:481-508]
    └── load_wan_model() [wan/modules/model.py]
        ├── 加载基础 DiT 模型配置
        ├── 实例化 WAN 模型
        └── 如果启用高低噪:
            └── 再次调用 load_wan_model() 加载高噪模型
```

#### 训练步骤
```
训练循环 [hv_train_network.py:1700+]
    ├── get_noisy_model_input_and_timesteps() [wan_train_network.py:515-566]
    │   ├── 采样时间步
    │   ├── 判断是否为高噪 (timestep >= boundary)
    │   └── 添加噪声到 latents
    │
    ├── call_dit() [wan_train_network.py:600-617]
    │   ├── swap_high_low_weights() [wan_train_network.py:568-598]
    │   │   └── 交换活动/非活动模型的权重
    │   └── _call_dit() [wan_train_network.py:619-687]
    │       └── 模型前向传播
    │
    └── 计算 flow matching 损失
```

---

## 高低噪切换逻辑

WAN 2.2 的核心创新之一是使用两个独立的模型分别处理高噪和低噪时间步。

### 配置参数

#### WAN 2.2 T2V 配置 (wan_t2v_A14B.py)
```python
t2v_A14B.v2_2 = True
t2v_A14B.boundary = 0.875        # 时间步边界 (0-1 范围)
t2v_A14B.sample_guide_scale = (3.0, 4.0)  # (低噪引导, 高噪引导)
```

#### WAN 2.2 I2V 配置 (wan_i2v_A14B.py)
```python
i2v_A14B.v2_2 = True
i2v_A14B.boundary = 0.900        # I2V 使用不同的边界
i2v_A14B.sample_guide_scale = (3.5, 3.5)
```

### 实现细节

#### 1. 初始化 (wan_train_network.py:82-109)

```python
# 第 82-84 行
self.dit_high_noise_path = args.dit_high_noise
self.high_low_training = self.dit_high_noise_path is not None

# 第 96-107 行：时间步边界设置
if self.high_low_training:
    if args.timestep_boundary is not None:
        self.timestep_boundary = args.timestep_boundary
    else:
        self.timestep_boundary = self.config.boundary

    # 如果边界 > 1.0，转换为 0-1 范围
    if self.timestep_boundary > 1.0:
        self.timestep_boundary = self.timestep_boundary / 1000.0
```

**关键点**：
- 高低噪训练仅在提供 `--dit_high_noise` 参数时启用
- 边界值决定何时从低噪模型切换到高噪模型
- 边界可以从配置文件或命令行参数获取

#### 2. 模型加载 (wan_train_network.py:481-508)

```python
def load_transformer(self, accelerator, args, dit_path, ...):
    # 加载低噪模型（主模型）
    model = load_wan_model(
        dit_path, self.config, torch_dtype=dit_dtype,
        fp8_scaled=args.fp8_scaled,
    )

    if self.high_low_training:
        # 加载高噪模型
        model_high_noise = load_wan_model(
            self.dit_high_noise_path,
            self.config,
            torch_dtype=dit_dtype,
            fp8_scaled=args.fp8_scaled,
        )

        # 保存高噪模型的权重
        self.dit_inactive_state_dict = model_high_noise.state_dict()
        del model_high_noise  # 释放内存

        # 初始化状态标志
        self.current_model_is_high_noise = False
        self.next_model_is_high_noise = False

    return model, ...
```

**内存管理**：
- 同时保存两个模型的状态字典
- 活动模型在 GPU 上，非活动模型的权重在内存中
- 可选：使用 `--offload_inactive_dit` 将非活动模型卸载到 CPU

#### 3. 时间步采样与分配 (wan_train_network.py:515-566)

```python
def get_noisy_model_input_and_timesteps(self, args, latents, noise, ...):
    # 采样时间步
    timesteps, sample_timesteps = self.sample_timesteps(
        scheduler, latents.shape[0]
    )

    if self.high_low_training:
        # 判断第一个样本是否为高噪
        high_noise = sample_timesteps[0] / 1000.0 >= self.timestep_boundary

        # 如果批次中有多个样本，确保它们都使用相同的模型
        # 重新采样最多 100 次以获得一致的分配
        attempts = 0
        while sample_timesteps.shape[0] > 1 and attempts < 100:
            for i in range(1, sample_timesteps.shape[0]):
                current_high_noise = (
                    sample_timesteps[i] / 1000.0 >= self.timestep_boundary
                )
                if current_high_noise != high_noise:
                    # 重新采样这个时间步
                    timesteps[i], sample_timesteps[i] = self.sample_timesteps(
                        scheduler, 1
                    )
            attempts += 1

        # 设置下一次前向传播要使用的模型
        self.next_model_is_high_noise = high_noise

    # 添加噪声到 latents
    noisy_model_input = scheduler.add_noise(latents, noise, timesteps)

    return noisy_model_input, timesteps
```

**采样策略**：
- 对于批次中的每个样本，根据时间步判断应该使用哪个模型
- 时间步 >= boundary → 高噪模型
- 时间步 < boundary → 低噪模型
- 确保批次中所有样本使用相同的模型（通过重新采样）

#### 4. 权重交换 (wan_train_network.py:568-598)

```python
def swap_high_low_weights(self, args, accelerator, model):
    """在高噪和低噪模型之间交换权重"""

    # 检查是否需要切换
    if self.current_model_is_high_noise != self.next_model_is_high_noise:
        logger.info(
            f"Swapping DiT weights to "
            f"{'high' if self.next_model_is_high_noise else 'low'} noise model"
        )

        # 获取当前模型的状态字典
        state_dict = model.state_dict()

        # 加载非活动模型的权重
        info = model.load_state_dict(
            self.dit_inactive_state_dict,
            strict=True,
            assign=True  # 直接赋值，避免额外复制
        )

        # 将当前模型的权重保存为非活动模型
        self.dit_inactive_state_dict = state_dict

        # 更新状态标志
        self.current_model_is_high_noise = self.next_model_is_high_noise

        # 如果启用了卸载，将非活动模型移到 CPU
        if args.offload_inactive_dit:
            self.dit_inactive_state_dict = {
                k: v.cpu() for k, v in self.dit_inactive_state_dict.items()
            }
```

**交换机制**：
- 仅在需要切换模型时执行交换
- 使用 `assign=True` 避免额外的内存复制
- 交换后，原来的活动模型变为非活动模型
- 支持将非活动模型卸载到 CPU 以节省 GPU 内存

#### 5. 前向传播 (wan_train_network.py:600-617)

```python
def call_dit(self, args, accelerator, transformer, ...):
    """调用 DiT 模型进行前向传播"""

    if self.high_low_training:
        # 在前向传播前确保使用正确的模型
        self.swap_high_low_weights(args, accelerator, transformer)

    return self._call_dit(
        args, accelerator, transformer,
        noisy_model_input, timesteps, context, ...
    )
```

#### 6. 推理时的切换 (do_inference 方法)

在生成样本视频时，也需要根据时间步切换模型：

```python
def do_inference(self, args, accelerator, ...):
    # 去噪循环
    for t in reversed(timesteps):
        if self.high_low_training:
            # 根据当前时间步判断使用哪个模型
            high_noise = t / 1000.0 >= self.timestep_boundary
            self.next_model_is_high_noise = high_noise

            # 切换到正确的模型
            self.swap_high_low_weights(args, accelerator, transformer)

        # 模型预测
        with torch.no_grad():
            model_output = transformer(...)

        # 调度器步进
        latents = scheduler.step(model_output, t, latents).prev_sample
```

### 高低噪切换的优势

1. **专门化**：每个模型专注于特定噪声范围，提升质量
2. **灵活性**：可以使用不同的训练策略训练两个模型
3. **引导控制**：高噪和低噪可以使用不同的引导比例
4. **性能优化**：可以针对不同噪声范围优化模型架构

---

## HV Train Network 实现

`NetworkTrainer` 是一个通用的训练基类，位于 `src/musubi_tuner/hv_train_network.py`。

### NetworkTrainer 基类结构

#### 类定义 (第 375 行开始)

```python
class NetworkTrainer:
    """
    通用网络训练器基类
    支持各种架构：HunyuanVideo, WAN, CogVideoX 等
    """

    @property
    def architecture(self):
        """子类必须实现，返回架构名称"""
        raise NotImplementedError

    def __init__(self):
        self.vae = None
        self.transformer = None
        self.text_encoders = []
        # ... 其他初始化
```

### 核心方法

#### 1. train() - 主训练循环 (第 1610 行)

这是整个训练流程的核心：

```python
def train(self, args):
    """主训练方法"""

    # === 初始化阶段 ===

    # 1. 验证参数
    self.assert_user_passed_valid_args(args)

    # 2. 处理架构特定参数
    self.handle_model_specific_args(args)  # 子类重写

    # 3. 准备 accelerator（分布式训练）
    accelerator = self.prepare_accelerator(args)

    # 4. 加载数据集
    train_dataloader = self.load_datasets(args)

    # === 模型加载阶段 ===

    # 5. 处理采样提示词
    sample_prompts_te_outputs = self.process_sample_prompts(
        args, accelerator, sample_prompts
    )

    # 6. 加载 VAE
    vae, vae_dtype = self.load_vae(args, vae_dtype, vae_path)

    # 7. 加载 Transformer/DiT
    transformer, dit_dtype = self.load_transformer(
        accelerator, args, dit_path, dit_dtype
    )

    # 8. 创建 LoRA 网络（可选）
    if args.network_dim:
        network = lora_module.create_arch_network(...)

    # === 优化器设置阶段 ===

    # 9. 创建优化器
    optimizer = self.get_optimizer(args, trainable_params)

    # 10. 创建学习率调度器
    lr_scheduler = self.get_lr_scheduler(
        args, optimizer, num_train_epochs, ...
    )

    # 11. Accelerator 准备
    transformer, optimizer, train_dataloader, lr_scheduler = (
        accelerator.prepare(
            transformer, optimizer, train_dataloader, lr_scheduler
        )
    )

    # === 训练循环阶段 ===

    for epoch in range(num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                # 获取批次数据
                latents = batch["latents"].to(dit_dtype)
                context = batch["context"]  # T5 embeddings

                # 采样噪声
                noise = torch.randn_like(latents)

                # 获取噪声输入和时间步
                noisy_model_input, timesteps = (
                    self.get_noisy_model_input_and_timesteps(
                        args, latents, noise, ...
                    )
                )

                # 前向传播
                model_pred = self.call_dit(
                    args, accelerator, transformer,
                    noisy_model_input, timesteps, context, ...
                )

                # 计算损失
                target = noise - latents  # Flow matching
                loss = F.mse_loss(
                    model_pred.float(), target.float(), reduction="mean"
                )

                # 反向传播
                accelerator.backward(loss)

                # 优化器步进
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # 定期采样
            if step % args.sample_every_n_steps == 0:
                self.do_inference(args, accelerator, ...)

            # 定期保存
            if step % args.save_every_n_steps == 0:
                self.save_model(args, accelerator, ...)
```

#### 2. 关键辅助方法

##### handle_model_specific_args()
```python
def handle_model_specific_args(self, args):
    """
    处理架构特定的参数
    子类必须重写此方法
    """
    raise NotImplementedError
```

WAN 的实现：
```python
# wan_train_network.py:119-215
def handle_model_specific_args(self, args):
    # 加载 WAN 配置
    if args.task == "t2v-14B":
        from musubi_tuner.wan.configs.wan_t2v_14B import t2v_14B
        self.config = t2v_14B
    elif args.task == "t2v-A14B":
        from musubi_tuner.wan.configs.wan_t2v_A14B import t2v_A14B
        self.config = t2v_A14B
    # ... 其他任务配置

    # 检测 DiT dtype
    self.dit_dtype = self.detect_dtype(args.dit)

    # 设置高低噪训练
    if args.dit_high_noise:
        self.high_low_training = True
        # ... 高低噪相关设置
```

##### load_transformer()
```python
def load_transformer(self, accelerator, args, dit_path, dit_dtype):
    """
    加载 Transformer/DiT 模型
    子类可以重写以实现特定加载逻辑
    """
    # 默认实现
    model = load_model(dit_path, dtype=dit_dtype)
    return model, dit_dtype
```

##### get_noisy_model_input_and_timesteps()
```python
def get_noisy_model_input_and_timesteps(
    self, args, latents, noise, scheduler
):
    """
    添加噪声并采样时间步
    WAN 重写此方法以实现高低噪逻辑
    """
    # 采样时间步
    timesteps = self.sample_timesteps(scheduler, latents.shape[0])

    # 添加噪声
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)

    return noisy_latents, timesteps
```

##### call_dit()
```python
def call_dit(
    self, args, accelerator, transformer,
    noisy_model_input, timesteps, context, **kwargs
):
    """
    调用 DiT 模型前向传播
    WAN 重写此方法以实现权重交换
    """
    # 基础实现
    model_output = transformer(
        noisy_model_input,
        timesteps=timesteps,
        context=context,
        **kwargs
    )
    return model_output
```

##### do_inference()
```python
def do_inference(self, args, accelerator, transformer, ...):
    """
    生成样本视频
    使用去噪循环从噪声生成视频
    """
    # 初始化随机噪声
    latents = torch.randn(...)

    # 去噪循环
    for t in reversed(timesteps):
        # 模型预测
        model_output = transformer(latents, t, context, ...)

        # 调度器步进
        latents = scheduler.step(model_output, t, latents).prev_sample

    # VAE 解码
    videos = vae.decode(latents)

    # 保存视频
    save_videos(videos, output_path)
```

#### 3. 优化器和调度器支持

##### get_optimizer()
```python
def get_optimizer(self, args, trainable_params):
    """
    创建优化器
    支持多种优化器类型和特性
    """
    optimizer_type = args.optimizer_type.lower()

    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    elif optimizer_type == "adamw8bit":
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(...)
    elif optimizer_type == "adafactor":
        from transformers.optimization import Adafactor
        optimizer = Adafactor(...)
    # ... 其他优化器类型

    return optimizer
```

支持的优化器：
- AdamW
- AdamW8bit (使用 bitsandbytes)
- Adafactor
- Lion
- SGD
- 等等

##### get_lr_scheduler()
```python
def get_lr_scheduler(self, args, optimizer, num_train_epochs, ...):
    """
    创建学习率调度器
    """
    scheduler_type = args.lr_scheduler.lower()

    if scheduler_type == "constant":
        scheduler = get_constant_schedule(optimizer)
    elif scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.lr_warmup_steps,
            num_training_steps=max_train_steps,
        )
    elif scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(...)
    # ... 其他调度器类型

    return scheduler
```

支持的调度器：
- constant
- linear
- cosine
- cosine_with_restarts
- polynomial
- constant_with_warmup

### WanNetworkTrainer 的扩展

`WanNetworkTrainer` 继承 `NetworkTrainer` 并重写关键方法：

```python
class WanNetworkTrainer(NetworkTrainer):
    @property
    def architecture(self):
        return "wan"

    def handle_model_specific_args(self, args):
        # WAN 特定的参数处理
        # 加载配置、设置高低噪等
        pass

    def load_transformer(self, accelerator, args, dit_path, dit_dtype):
        # 加载 WAN 模型
        # 如果启用，加载高噪模型
        pass

    def get_noisy_model_input_and_timesteps(self, args, latents, noise, ...):
        # 实现高低噪时间步采样
        pass

    def call_dit(self, args, accelerator, transformer, ...):
        # 在前向传播前交换权重
        if self.high_low_training:
            self.swap_high_low_weights(...)
        return self._call_dit(...)

    def do_inference(self, args, accelerator, ...):
        # WAN 特定的推理逻辑
        # 支持 T2V、I2V、控制视频等模式
        pass
```

---

## 关键配置参数

### WAN 2.2 特定参数

通过 `wan_setup_parser()` 添加 (wan_train_network.py:690-719)：

```python
--task                 # 任务类型
                       # 选项: t2v-14B, i2v-14B, t2v-A14B, i2v-A14B
                       # 默认: t2v-14B

--t5                   # T5 文本编码器路径 (必需)

--clip                 # CLIP 编码器路径 (仅 I2V WAN 2.1 需要)

--dit_high_noise       # 高噪 DiT 模型路径
                       # 提供此参数即启用高低噪训练

--timestep_boundary    # 时间步边界 (0-1 范围)
                       # 未指定时使用配置文件中的默认值

--fp8_scaled           # 为 DiT 权重使用缩放的 fp8

--fp8_t5               # 为 T5 编码器使用 fp8

--vae_cache_cpu        # 在 CPU 上缓存 VAE 特征

--one_frame            # 使用单帧采样方法

--offload_inactive_dit # 将非活动模型卸载到 CPU（节省内存）
```

### 模型架构规格

从配置文件 (shared_config.py):

```python
# Transformer 规格
transformer_dim = 5120              # 模型维度
transformer_layers = 40             # 层数
transformer_heads = 40              # 注意力头数
ffn_dim = 13824                     # FFN 隐藏维度

# 补丁尺寸
spatial_patch_size = (1, 2, 2)      # (时间, 高度, 宽度)
temporal_patch_size = (1,)

# 输入维度
input_dim = 16                      # T2V
input_dim = 36                      # I2V (拼接图像 latents)

# 文本长度
text_len = 512                      # T5 token 数量

# VAE 步长
vae_stride = (4, 8, 8)              # 时间, 高度, 宽度
```

### 通用训练参数

```python
# 数据
--dataset_config       # 数据集配置文件路径 (必需)
--resolution           # 训练分辨率 "高度,宽度"
--num_latent_frames    # Latent 帧数

# 模型
--dit                  # DiT 模型路径 (必需)
--vae                  # VAE 模型路径
--network_dim          # LoRA 秩
--network_alpha        # LoRA alpha

# 训练
--output_dir           # 输出目录
--max_train_steps      # 总训练步数
--max_train_epochs     # 总训练 epoch 数
--learning_rate        # 学习率
--lr_scheduler         # 调度器类型
--lr_warmup_steps      # 预热步数

# 优化
--optimizer_type       # 优化器类型
--mixed_precision      # fp16, bf16, 或 no
--gradient_accumulation_steps  # 梯度累积步数

# 内存
--gradient_checkpointing  # 启用梯度检查点
--blocks_to_swap       # 要交换的块数
--cpu_offload_checkpointing  # CPU 卸载检查点

# 采样
--sample_every_n_steps    # 每 N 步采样一次
--sample_prompts          # 采样提示词文件

# 保存
--save_every_n_steps      # 每 N 步保存一次
--save_model_as           # 保存格式 (safetensors, ckpt)

# 其他
--seed                 # 随机种子
--logging_dir          # 日志目录
--log_with             # 日志工具 (tensorboard, wandb)
```

---

## 重要实现细节

### 1. Flow Matching 损失

WAN 使用连续流匹配而非传统的噪声预测：

```python
# 目标是噪声和 latents 的差异
target = noise - latents

# 模型预测这个差异
model_pred = transformer(noisy_latents, timesteps, context, ...)

# MSE 损失
loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
```

**与传统扩散的区别**：
- 传统：预测添加的噪声
- Flow Matching：预测从当前状态到目标状态的"速度场"

### 2. 内存优化特性

#### 块交换 (Block Swap)
```python
--blocks_to_swap 8
```
将 Transformer 的某些层卸载到 CPU，训练时再加载回来。

#### 梯度检查点 (Gradient Checkpointing)
```python
--gradient_checkpointing
```
重新计算某些激活而非存储，节省内存但增加计算。

#### 非活动模型卸载
```python
--offload_inactive_dit
```
将非活动的高/低噪模型卸载到 CPU。

#### VAE 缓存
```python
--vae_cache_cpu
```
在 CPU 上缓存 VAE 特征，节省 GPU 内存。

### 3. 引导比例

WAN 2.2 对高低噪使用不同的引导比例：

```python
# T2V
sample_guide_scale = (3.0, 4.0)  # (低噪, 高噪)

# I2V
sample_guide_scale = (3.5, 3.5)
```

在推理时应用：
```python
if cfg_scale > 1.0:
    # 分类器无关引导
    noise_pred_uncond, noise_pred_text = model_output.chunk(2)
    model_output = noise_pred_uncond + cfg_scale * (
        noise_pred_text - noise_pred_uncond
    )
```

### 4. 时间步分桶

可选的均匀时间步分布：

```python
--timestep_bucketing  # 启用分桶
--num_buckets 10      # 桶数量
```

确保跨时间步范围的平衡训练，对多阶段训练有用。

### 5. 数据集配置

数据集通过 TOML 配置文件定义：

```toml
[general]
resolution = [768, 1360]
num_latent_frames = 121
caption_extension = ".txt"

[[datasets]]
  [[datasets.subsets]]
  video_dir = "/path/to/videos"
  num_repeats = 1

  # 可选：预计算的 latents
  latents_dir = "/path/to/latents"

  # 可选：预计算的 T5 embeddings
  context_dir = "/path/to/contexts"
```

### 6. 支持的模式

WAN 训练器支持多种模式：

#### T2V (文本到视频)
```bash
--task t2v-A14B --t5 /path/to/t5
```

#### I2V (图像到视频)
```bash
--task i2v-A14B --t5 /path/to/t5 --clip /path/to/clip  # WAN 2.1
--task i2v-A14B --t5 /path/to/t5                        # WAN 2.2
```

#### 控制视频 (Fun-Control)
```bash
--control_video  # 添加控制信号
```

#### 单帧推理
```bash
--one_frame  # 使用单帧采样方法
```

### 7. LoRA 训练

创建并训练 LoRA 适配器：

```python
# 创建 LoRA 网络
network = lora_wan.create_arch_network(
    multiplier=1.0,
    network_dim=args.network_dim,      # 秩，例如 32
    network_alpha=args.network_alpha,  # alpha，例如 16
    transformer=transformer,
)

# 获取可训练参数
trainable_params = network.prepare_optimizer_params(
    args.learning_rate,
    args.learning_rate,  # unet_lr (WAN 中相同)
    args.learning_rate,  # text_encoder_lr (通常不训练)
)

# 训练 LoRA 权重而非完整模型
optimizer = get_optimizer(args, trainable_params)
```

LoRA 权重分别保存，可以与基础模型组合。

---

## 总结

WAN 2.2 训练系统的核心特点：

1. **模块化设计**：清晰的基类和子类结构
2. **双模型架构**：高低噪专门化训练
3. **高效内存管理**：多种优化选项
4. **灵活配置**：支持多种任务和模式
5. **Flow Matching**：先进的训练目标
6. **生产就绪**：完整的检查点、日志、采样功能

关键文件：
- `src/musubi_tuner/wan_train_network.py` - WAN 特定逻辑
- `src/musubi_tuner/hv_train_network.py` - 通用训练框架
- `src/musubi_tuner/wan/configs/` - 模型配置
- `src/musubi_tuner/wan/modules/` - 模型组件

理解这些组件及其交互对于有效使用和扩展 WAN 2.2 训练系统至关重要。
