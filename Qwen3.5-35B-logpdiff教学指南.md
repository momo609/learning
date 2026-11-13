# Qwen3.5-35B-A3B 训推一致性教学指南

Ascend A3 · BF16 · verl FSDP2 / FSDP Turbo · vLLM-Ascend · Geo3K GRPO  
实验快照：2026 年 11 月 13 日 10:11 左右（北京时间）

**结果。** FSDP2 的 Geo3K 长跑已完成 64/100 个训练 step，第 65 步已完成评分。重新逐位核对第 1–65 步的 **841,693 个有效响应 token**，每步 `logpdiff_mean=0`、最大绝对差为 0、位级差异数为 0。FSDP Turbo 独立跑完 3 步，共 **46,012 个 token** 逐位一致，第 2、3 步均覆盖 actor 更新及权重同步后的评分。

## 1. 先定义验收，再开始改代码

verl 的 actor 以 teacher forcing 对 rollout 已生成的响应重新计算 `old_log_probs`，rollout 提供采样时的 `rollout_log_probs`。固定同一批 prompt、图片、响应 token、权重版本和 R3 路由，用 `response_mask` 选出有效位置。这里的 `logpdiff_mean` 是 `training/rollout_logprobs_diff_mean`，即两个 **原始 FP32 token logprob** 的绝对差均值。验收同时检查 `diff_max=0`、所有评分有限，以及 FP32 按 `int32` 查看时位模式完全相同。第一枚响应 token 取最后一个 prompt 位置的预测分数。actor 必须独立重算，`algorithm.rollout_correction.bypass_mode` 保持关闭。

排障时固定同一组 token，分别走三条路径。先比较 actor 与 rollout prefill，再比较 rollout prefill 与逐 token decode，最后在两条有差异的路径上找第一个分歧的层和张量。首次能复现后，再逐项改变 batch size、批内顺序、是否带图、响应长度、eager 与 FULL 图、权重版本。把一次变更限制在一个维度，能用固定输入验证因果。 [Inferact 固定 token 三路排查](https://github.com/Inferact/zero-mismatch-rl/blob/main/skill/guides/debugging-mismatch/01-isolate-the-side.md) · [MindSpeed-Agent 批次不变性验证](https://gitcode.com/Ascend/MindSpeed-Agent/tree/master/plugins/batch-invariance-validation)

| 最小对照 | 优先排查 |
|---|---|
| actor 与 prefill 分歧，prefill 与 decode 一致 | 预处理、前向 dtype、RoPE、R3 权重、词表掩码 |
| actor 与 prefill 一致，prefill 与 decode 分歧 | KV / GDN 状态、因果上下文长度、单 query 和多 query 算术、图捕获 |
| 同侧单条与混批分歧 | batch-invariant 算子分派、归约、padding、packed offsets |
| 初始一致，更新后分歧 | FSDP 前向参数视图、导出 dtype、在线 refit、图引用的存储 |
| 隐藏状态和 logits 一致，分数仍有 FP32 末位差 | 打分时的词表范围、屏蔽 token 和 log-softmax 口径 |

## 2. 从环境打通到首个可测基线

1. 使用完整 **Qwen3.5-35B-A3B BF16** 权重，路径为 `/home/vllm/weights/Qwen3.5-35B-A3B`。`A3B` 描述每 token 的激活规模，部署仍需装载全模型参数。训练侧两卡 FSDP 分片，rollout 用 **TP1/PP2** 按层分两段。该拓扑保留了每层单卡计算，便于在验证 Turbo 时固定推理算术。本文的位级结果覆盖这个拓扑。
2. 固定三套源码和 NPU 扩展。实验基线为 verl `4b396702074369ae977db99387bd8470f02f62b9`、vLLM 0.30.0 `ced6857afa0ea7b2e3f0846a62e1394e90f15607`、vLLM-Ascend `00f74f23614a9af1bbdc0344e750379379a5f7c5`。镜像实验栈为 Torch 2.10.0、torch-npu 2.10.0.post7、Transformers 5.14.1。切到 v0.30.0 后，在 A3 上重新构建 Ascend 自定义算子和 BI 扩展。验证实际 import 来源及 commit，同时保留工作树里的改动和已编译扩展。
3. 先用短序列合成加法把运行链路闭合。早期日志出现 NPU IPC `507899`、`npu_rotary_mul` `561103`、packed 张量断言，以及 actor entropy 编译失败。逐一处理后，关闭 `actor.fsdp_config.use_torch_compile`，rollout 继续用 `FULL_DECODE_ONLY`。隔离 Ray 临时目录及通信端口。Geo3K 另需安装 `qwen_vl_utils`，否则图像样本预处理会失败。容器曾因 PID 1 被宿主清理进程杀死而重启，后续验证容器启用 Docker `--init`。
4. 先固定打分口径。早期 actor BF16 打分、rollout FP32 打分记录为均值 `0.01376376`、2,023 个差异。两侧都用 FP32 打分后，首步 2,048 个 token 仍有 1,875 个不同，均值 `0.01639608`，最大差 `0.59269238`。两次运行各自采样，后续固定轨迹后再定位算子。BF16 权重用于模型前向，词表 logits 转 FP32 后用于 log-softmax，两处精度各有用途。 [早期实验记录与复现文件见第 8 节](#8-复现入口与证据)

## 3. 短序列上先打通四条路径

| 路径 | 实际尝试与定位 | 最终实现 |
|---|---|---|
| **BI 与确定性** | 四层代理中，同一 actor 的 batch 1 对 batch 2 有 87/91 个评分不同。启用 BI 后同侧变为 0/91，actor 与 rollout 仍有 91/91 个不同。 | 两端启用 NPU `set_deterministic_level(3)`，actor 复用 Ascend BI 注册，避免普通确定性 setter 覆盖 level 3。补 BI 末维 sum 的 autograd，并同步替换 FIA 前向和图捕获的 workspace 查询。 |
| **R3 专家回放** | 先在四层固定轨迹验证 router 梯度，再验证 PP1/PP2 的路由和评分。 | Ascend V2 runner 记录每层实际 expert ID，经 `IntermediateTensors` 从 PP0 到 PP1 合并，过滤 dummy/padding。verl 校验 packed offsets、层数、top-k 与 ID，actor 重算和更新复用 ID，同时用当前 router logits 计算所选专家权重以保留梯度。全模型为 40 层，每 token 选 8 个专家。 |
| **GDN 与前后向** | 短前缀的 RMSNorm、QKVZ/BA 投影能精确匹配，PyTorch causal conv + SiLU 与 Ascend 融合路径却有差异。训练和 rollout 的 chunk / recurrent GDN 也在状态更新及舍入上分叉。 | 实现 AscendC 一致性 recurrent GDN，BF16 Q/K/V、FP32 recurrent state，内部固定 8-token tile，支持长序列。actor 与 rollout prefill 共用递推前向，并对照原有 decode 递推。统一 causal conv、SiLU、Q/K L2、gate/beta 和 gated RMSNorm。actor 的自定义 backward 用可微 chunk 规则重建图并求梯度。 |
| **全注意力与评分** | 固定 token 逐层比对 QKV、Q/K RMSNorm、RoPE cache、packed MRoPE 和 FIA 输入输出。 | 统一 Gemma RMSNorm 的残差舍入、融合 shared expert SwiGLU、RoPE 算术和样本隔离。修复等长 MRoPE nested tensor 的 token offsets。两端共用 FP32 BI log-softmax，actor 补反向。FSDP 已管理 BF16 前向精度，一致性路径去掉额外 autocast，保留算子内显式 FP32 运算。 |

**先测更新，再扩长度。** 完整模型短序列首步 64 个 token 已经全零，但第 2 步 64 个 token 中有 58 个不一致，均值 `5.46785e-5`。定位到 FSDP2 的 FP32 主权重与 actor 实际使用的 BF16 参数视图。直接导出 `state_dict()` 时，rollout 以 FP32 存储的 GDN `A_log` 保留了 actor 前向已舍去的低位。修复后，按 actor 前向 dtype 导出浮点参数，DTensor 先在分片上转换再聚合，buffer 保留原 dtype，优化器继续更新 FP32 主权重。`A_log` 探针从 512 个不一致元素降到 0。

**验证在线 refit。** FULL 图需要继续读取图捕获时引用的 MoE 权重地址。Ascend post-load 会重排专家权重，曾在临时分配阶段 OOM。补齐 Ascend 专家模块的 `replace_parameter` 路径，按层将重排结果写回原存储，控制临时缓冲。真实 NPU 探针确认权重更新后 FULL 图评分发生变化，并与新 prefill 一致。两项同步修复后，短序列完整模型连续 3 步共 192 个有效 token 逐位一致。

## 4. 换 Geo3K 后，长序列和图片分别排查

用 verl 的 Geo3K 预处理脚本保留原图，三步回归取训练集前 64 条、测试集前 16 条。参考样例设 prompt 上限 **1024**、response 上限 **2048**、总长度 **3072**。每步 4 个 prompt，每题 2 条响应。首次长序列运行首步均值 `0.04958232`、最大差 `9.68904686`。固定同一组 token 做四层对照后，短带图样本的 actor 与 prefill 已经分歧，长纯文本样本还出现 prefill 与 decode 分歧。这两个现象对应不同修复。 [verl Geo3K 预处理](https://github.com/verl-project/verl/blob/main/examples/data_preprocess/geo3k.py)

1. **FIA 的 prefill/decode 算术。** 同一份 Q/K/V 重放定位到第 4 层全注意力，多 query 和单 query 路径的 BF16 舍入不同。actor 与 rollout prefill 改用 decode 的单 query 算术，每个 query 的 KV 长度是它自己的因果前缀，同序列 query 共用物理 KV pages 后合批调用。补齐已有缓存的 prefill 和 prefill/decode 混批路径。单算子重放 2,103 个位置逐位一致，真实错峰图片请求也验证了各自的分数和路由不受混批影响。
2. **视觉输入和 FSDP 包装。** 先让 rollout 使用与 actor 相同的 CPU 图像归一化，设置 `mm_device_do_normalize=False`。给两侧相同 pixel values 和 grid 后，视觉编码仍有差异，因此 rollout 的 BF16 TP1 路径复用 HF 视觉编码器，同时保留 checkpoint 和 refit 参数名。完整模型还发现 FSDP2 会将视觉 block 入口的 FP32 RoPE cos/sin 参数转换成 BF16。只对视觉 block 关闭 `cast_forward_inputs`，保留视觉 RoPE 精度。实际四张图双 rank 重放覆盖 634,880 个视觉嵌入元素和 333 个导出视觉参数。
3. **专家 ID 相同，权重仍可能分歧。** 40 层固定轨迹定位到第 34、36 层的 MoE 输出。首处不同的 gate logits、expert ID 都一致，但某个 BF16 expert weight 分别为 `0.12158203125` 和 `0.12109375`。只替换这一项就复现了另一侧输出。统一用 `moe_selected_logit_weights`，先 gather 选中的 8 个 logits，再做 FP32 softmax，最后转回 BF16。rollout 保留原路由器的选择与顺序，actor 回放相同 ID 并保留梯度。三步的残差收敛到 9、7、10 个 FP32 末位差。
4. **核对评分词表。** 残差位置的隐藏状态和词表 logits 已相同。实际 verl rollout 在归一化前屏蔽 tokenizer 范围外的词表尾部及图像、视频占位 token，actor 也接入同一个 `mask_rollout_logits_`。entropy 用掩码后的概率与有限原始 logits 计算，避免 `0 × -inf`。先用原有 logits 重放 9 个差异点，再用双 rank FSDP2 重放三条 2048-token 轨迹，最后进行整模型新跑。最终 Geo3K 三步分别核对 **14,848、14,159、15,759** 个 token，全部逐位一致。

长序列中间两次首步复测的均值分别是 `0.00585496`（`driver-geo3k-r3-long-fix.log`）和 `0.00326377`（`driver-geo3k-r3-pd-vision-fix.log`）。随后专家权重归一化把残差压到约 `1e-10`，词表掩码消掉最后的位级差异。跨运行数字用于回顾排查进展，单项归因以各步固定输入的对照为准。

## 5. 复现配置与 Turbo 迁移

下表的 `actor.*`、`rollout.*` 均位于 `actor_rollout_ref`。`rl_config.*` 位于 `rollout.engine_kwargs.vllm.additional_config`。

| 配置 | 来源与作用 |
|---|---|
| `VERL_NPU_TRAIN_INFER_CONSISTENCY=1` | 本次新增 actor 一致性入口。接入 BI、level 3、对齐前向、共享评分与按前向 dtype 导出。 |
| `actor.fsdp_config.router_replay.mode=R3`、`rollout.enable_rollout_routing_replay=True` | verl 原有 R3 概念及 rollout 开关。本次新增 FSDP 执行接入和 Ascend V2 PP2 路由回传。两侧一起启用。 |
| `rl_config.enabled=True`、`rl_config.enable_batch_invariant=True` | 原有 Ascend RL/BI 入口，使用 ND 权重布局、确定性通信设置和 BI 算子。torch-npu 的 level 3 API 原有，本次在两端接入。 |
| `rl_config.enable_gdn_consistency=True` | 本次新增。rollout prefill 使用一致性 GDN 状态与递推前向。PD 一致性还依赖 FIA 和状态修复。 |
| `VERL_FP32_LOGPROBS=1`、`rollout.calculate_log_probs=True`、`rollout.logprobs_mode=raw_logprobs` | FP32 环境变量是新增诊断入口，后两项为原有评分选项。NPU 对齐分支优先使用共享 BI scorer。采样使用 `temperature=1.0, top_p=1.0, top_k=-1`。 |
| `VLLM_USE_V2_MODEL_RUNNER=1`、`FULL_DECODE_ONLY`、`compilation_config.mode=0`、捕图 `[1,2,4]` | 原有运行选项。decode 使用完整图，prefill 走图外。关闭 actor/ref torch.compile、prefix cache、chunked prefill 和推测解码。 |
| `mm_device_do_normalize=False`、`checkpoint_engine.backend=naive`、`VERL_FORCE_WEIGHT_TRANSFER_SHM=1` | 前两项是原有选项，SHM 强制开关为新增。分别控制图像归一化、权重同步后端和传输介质。 |
| `VERL_LOGPDIFF_DUMP_DIR=<目录>` | 本次新增。每步保存独立评分、mask 和 R3 路由，供离线复核。 |

**切到 FSDP Turbo。** 以官方样例为入口，改成双卡 `fully_shard_parallel_size=2`、EP1、Ulysses SP1，继续使用相同 rollout 和 Geo3K 长度。新增改动集中在 `engine_workers.py`、`transformer_impl.py`、`fsdp_turbo_impl.py`。给 Turbo 打开 R3 worker 分派和 packed 回放，允许进入已有 NPU 一致性前向，跳过 Turbo 自带的 Qwen fused GDN/causal conv 及 attention 覆盖。Turbo 继续执行分片、预取和激活重算，日志确认 `FSDPTurboEngineWithLMHead` 包装 72 个模块。将视觉 block 的 FP32 RoPE 输入策略放进 Turbo sharding plan，使用该 plan 的 BF16 `param_dtype` 控制 actor 前向和 rollout 导出。运行依赖 FSDPTurbo `1aed4b8`。三步实测涵盖两次更新后的重新评分，适用范围为当前双卡 EP1 和对齐前向。 [verl 官方 FSDP Turbo 样例](https://github.com/verl-project/verl/blob/main/examples/grpo_trainer/run_qwen3_5_35b_fsdp_turbo.sh)

### 课堂验收顺序

1. 固定轨迹三路对照，验证 shift、mask、FP32 有限值和有效 token 数。用同一条短轨迹做 batch 1、混批和批内换序。
2. 在双卡运行全模型短序列至少三步，确认优化器确实有更新、rollout 确实收到了更新后的权重，再比第二步和第三步。
3. 进入 Geo3K，分开测纯文本与带图、短与 2048-token 响应、带缓存混批、eager 与 FULL 图，以及图捕获后的在线 refit。每类都用原始逐 token 分数验收。
4. 长跑逐步留存原始张量，核对完成步数与评分步数。启用断点后按真实 global step 校验连续性并区分不同 run。实际 response mask 宽度可小于配置上限，原长跑第 48 步为 1887，以下示例按有效 mask 校验。

```python
from pathlib import Path

import torch

root = Path('/home/vllm/wxx-rl/consistency-qwen35/logprob-geo3k-r3-100step')
expected_steps = range(1, 66)  # 换成已保存且要验收的连续步范围
for step in expected_steps:
    row = torch.load(root / f'logprob_step_{step}.pt', map_location='cpu', weights_only=True)
    mask = row['response_mask'].bool()
    actor_full = row['old_log_probs']
    rollout_full = row['rollout_log_probs']
    assert mask.shape == actor_full.shape == rollout_full.shape, step
    assert row['routed_experts'].shape[-2:] == (40, 8), step
    actor = actor_full[mask].contiguous()
    rollout = rollout_full[mask].contiguous()
    assert actor.numel() and actor.dtype == rollout.dtype == torch.float32, step
    assert torch.isfinite(actor).all() and torch.isfinite(rollout).all(), step
    delta = (actor - rollout).abs()
    mismatch = (actor.view(torch.int32) != rollout.view(torch.int32)).sum().item()
    print(step, actor.numel(), delta.mean().item(), delta.max().item(), mismatch)
    assert delta.max().item() == 0 and mismatch == 0, step
```

## 6. 性能对比和 actor 瓶颈

### 端到端 A/B

同一双卡 Geo3K 配置、每组 4 步，以第 2–4 步作为稳态窗口。对照关闭本次训推一致性路径，对齐组启用完整组合。两组输入题目相同，生成 token 数和训练轨迹各自变化。端到端吞吐为窗口内 `有效 response token 总数 / 整步耗时总和`，生成吞吐用 `gen` 耗时作分母。

| 稳态指标（step 2–4） | 修复前 | 对齐后 | 变化 |
|---|---:|---:|---:|
| 端到端响应吞吐，token/s | 21.55 | 19.23 | −10.7% |
| 生成阶段响应吞吐，token/s | 83.34 | 74.77 | −10.3% |
| 平均整步耗时，s | 740.07 | 793.42 | +7.2% |
| 平均生成耗时，s | 191.33 | 204.10 | +6.7% |
| 平均 actor 重算 logprob，s | 24.39 | 23.95 | −1.8% |
| 平均 actor 更新，s | 388.85 | 438.86 | +12.9% |
| 平均权重同步，s | 134.49 | 125.48 | −6.7% |
| 有效 token / 位级差异数 | 47,839 / 46,832 | 45,778 / 0 | |

整步均值增加 53.36 秒，actor 更新项增加 50.02 秒。`−10.7%` 是整套对齐功能的端到端观测，包含 R3、BI、GDN/FIA、视觉、词表掩码和 refit 路径。若纳入首步，两组为 20.08 → 18.69 token/s，降幅约 **6.9%**。对照数据来自 `perf-ab-20261112/comparison.json`。

### 单算子拆分

下表取长度 2,103 的单算子测试，单位为每次调用的毫秒。单算子记录适合比较同一输入的计算成本，整步开销还取决于调用次数、调度、重算、通信和 CPU offload。

| 单项 | 前向，修复前 → 对齐后 | 反向，修复前 → 对齐后 | 对齐反向拆分 |
|---|---:|---:|---|
| GDN，原生 chunk → recurrent 前向 | 26.31 → 13.51 | 61.97 → 76.06 | 重算 19.53，求梯度 56.52。反向净增 **14.08 ms** |
| FIA，训练 FA → 对齐单 query | 0.59 → 9.62 | 1.16 → 1.28 | 重算 0.32，求梯度 0.96。反向净增 **0.12 ms** |
| QKV / mRoPE | 1.66 → 0.65 | 1.85 → 4.57 | 重算 1.78，求梯度 1.66。反向净增 **2.72 ms** |
| FP32 scorer | 7.14 → 10.48 | 12.83 → 16.82 | BI scorer 单独是 6.59 / 12.84，含掩码后是 10.48 / 16.82 |
| 矩阵乘前向 | level 0 原生 0.144 → level 3 原生 0.152 → level 3 BI 0.156 | — | 同一形状 |

R3 的 512-token rollout 小测三次中位数为 **11.146 → 11.197 s**，约 `+0.46%`，样本间波动大于中位数差。四张图的编码器中位数范围由 **43.23–60.07 ms** 变为 **43.52–54.19 ms**。权重导出微测 33,554,432 个本地元素时，FP32 直接聚合为 **7.43 ms**，BF16 转换后聚合为 **7.28 ms**。这些小测各自的输入、统计口径见 `perf-components-20261112/`。

### 为什么两个 rank 的 GDN 反向看起来是 395 秒和 10 秒

早期流事件把 rank 0 的 GDN backward 范围记为 **394.64 s**，rank 1 为 **10.12 s**，但整段 actor backward 分别为 **643.52 s** 与 **643.74 s**。匹配卡上的完整 NPU 流水复测中，GDN 范围缩到 **13.02 / 11.09 s**，FIA backward 范围为 **0.20 / 0.27 s**。395 秒是包含队列和等待的区间，不能作为 GDN 内核耗时。

同一流水中，采样到的一个 backward 窗口约 **65 s**，两 rank 的设备计算各约 **4.9 / 4.5 s**。trace 在 `FSDP::post_backward_reduce` 下记录到大量 `empty_strided`、`fill_`、`add_`、`copy_`。另一次固定轨迹里，rank 0 主要表现为 host 提交和 copy，rank 1 在 collective 与 stream sync 等待同伴。完整 step trace 的 optimizer 段约 **267–298 s**。按证据，整体 actor 的主要耗时在 **反向期间的 host 调度、FSDP 归约 / CPU offload 路径及 optimizer**，GDN 重算确有成本，FIA 的新增成本主要出现在前向。不同 trace 的同步点和轨迹不一样，嵌套区间也有重叠，不能相加作为对照组净开销。 [本地分段计时原表](qwen35-actor-profile-20261112/actor-segments.json)

## 7. 长跑曲线如何解读

![Geo3K 长跑第 1–65 步 logpdiff 与有效 token](qwen35-logpdiff-guide-assets/geo3k-longrun-parity.png)

100 步是 FSDP2 长跑的计划步数。快照时进度条到 64/100，第 65 步已有评分。图中 65 步都来自 `.pt` 原始张量复核。第 65 步的评分完成和训练更新完成是两个时间点。

![Geo3K 长跑训练 reward、响应长度和各阶段耗时](qwen35-logpdiff-guide-assets/geo3k-longrun-training.png)

这份日志有第 1–63 步的完整性能与 reward 指标。前 10 步训练 batch reward 均值为 **0.180**，第 54–63 步为 **0.585**，呈上升趋势。该运行关闭定期测试，曲线衡量的是训练批次 reward。完整指标的整步耗时中位数为 **22.71 分钟**，其中 actor 更新中位数 **15.64 分钟**，生成 **3.38 分钟**，权重同步 **3.14 分钟**。各阶段中位数分别计算。

Turbo 的三步正式验证已经正常退出。另一次 Turbo 100 步尝试保存了首步零差异结果，快照时该容器已退出；随后启动的 sample-parallel Turbo 长跑仍在初始化。这两次 Turbo 长跑与上面的 FSDP2 曲线属于不同运行。

## 8. 复现入口与证据

远端任务根目录为 `/home/vllm/wxx-rl/consistency-qwen35`。在相应的容器内使用该目录下对应运行的工作树与环境。按实验快照复现时，从已有改动的工作树启动，并核对实际生效参数。早期 `alignment-*.patch` 是阶段性快照，Geo3K 长序列与 Turbo 的最终修复还需要对应工作树中的后续改动。

已配好依赖和源码的 `wxx-rl` 容器中，FSDP2 三步 Geo3K 回归入口为 `bash /home/vllm/wxx-rl/consistency-qwen35/run-geo3k-r3-vocab-aligned.sh`。Turbo 使用独立容器，把任务目录以相同绝对路径挂载进容器，以 `TURBO_RUN_NAME=run-repro TURBO_STEPS=3 bash /home/vllm/wxx-rl/consistency-qwen35/fsdp-turbo-r3/launch.sh` 启动，选择空闲的两张卡。FSDP2 三步脚本使用固定日志和 dump 目录，复测前先检查路径，给新运行单独分配输出目录以保留旧证据。

| 用途 | 任务根目录下的文件 |
|---|---|
| 两种初始评分口径 | `driver11-v030.log`、`driver12-fp32.log` |
| 短序列首步零、第二步漂移 | `driver-full-r3.log`、`fsdp2-alog-refit-npu.log`、`npu-graph-moe-refit.log` |
| 短序列三步复现与验收 | `run-full-r3-sync-precision.sh`、`full-r3-sync-precision-verification.json` |
| Geo3K 长序列故障与最终三步 | `driver-geo3k-r3.log`、`driver-geo3k-r3-long-router.log`、`run-geo3k-r3-vocab-aligned.sh`、`geo3k-r3-vocab-aligned-verification.json` |
| 固定 token / 算子探针 | `long-parity/`、`mm-parity/` |
| FSDP2 100 步入口、日志与原始评分 | `run-geo3k-r3-100step.sh`、`driver-geo3k-r3-100step.log`、`logprob-geo3k-r3-100step/` |
| 性能 A/B | `perf-ab-20261112/comparison.json` 和两组 `run/driver.log` |
| 单算子与 NPU trace | `perf-components-20261112/` 下的 `op-microbench.json`、`actor-segments.json`、`frozen-trace/trace-cause.txt`、`actor-stream-trace/aligned/run/npu-trace/` |
| Turbo 三步入口、日志与验收 | `fsdp-turbo-r3/launch.sh`、`fsdp-turbo-r3/run-qwen35-fsdp-turbo.sh`、`fsdp-turbo-r3/run-02/driver.log`、`fsdp-turbo-r3/run-02/verification.json` |

| 要讲解的改动 | 对应代码位置（仓库内相对路径） |
|---|---|
| actor R3、GDN、FIA | verl `verl/models/transformers/{qwen3_5_router_replay,qwen3_5_npu_consistency,qwen3_5_npu_attention}.py` |
| actor 权重导出、词表、Turbo | verl `verl/workers/engine/fsdp/{transformer_impl,fsdp_turbo_impl}.py`、`verl/workers/rollout/vllm_rollout/utils.py` |
| 视觉 FSDP 策略、MoE refit | verl `verl/utils/fsdp_utils.py`、`verl/utils/vllm/vllm_unquant_utils.py` |
| PP2 R3、BI 与 GDN 扩展 | vLLM-Ascend `vllm_ascend/worker/v2/model_runner.py`、`vllm_ascend/batch_invariant.py`、`vllm_ascend/ops/gdn_consistency.py`、`csrc/torch_ops_extension/consistent_gdn/` |
| rollout 长 FIA、视觉、专家权重 | vLLM-Ascend `vllm_ascend/attention/attention_v1.py`、`vllm_ascend/patch/worker/patch_qwen3_5_vision.py`、`vllm_ascend/ops/fused_moe/router/fused_topk_router.py` |
| PP2 路由能力校验 | vLLM `vllm/platforms/interface.py`、`vllm/config/vllm.py` |
