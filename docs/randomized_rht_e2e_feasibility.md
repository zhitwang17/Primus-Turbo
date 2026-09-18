# Primus-Turbo Static Randomized RHT：实现与实验总结

**日期：** 2026-09-18\
**基线：** `main@9f05cd85a54a9db3e60acf9520ba3a05a96228c4`\
**分支：** `zhitao/randomized-rht-e2e`\
**平台：** AMD gfx950；ROCm 7.2；PyTorch 2.13 nightly；FlyDSL 0.2.4

## 结论

本次已在 Primus-Turbo 的 MXFP4 训练链路中完成 **static randomized RHT**，并打通 dense/grouped GEMM、dense/grouped MLP、RMSNorm 以及 fused GLU/dGLU producer。实现通过本地 GPU 正确性验证，且代表性 full-op 与训练 step 的平均性能回退低于 2%。

但在三组独立初始化/数据、每组 1000 step 的单卡 dense MLP 训练代理中，randomized 与现有 fixed H16×2 的最终 loss 总体差异仅 **+0.0009%**（正数代表更差），不同 seed/run 的方向不一致。因此当前证据是：**工程实现可行、性能成本可接受，但尚未观察到可复现的训练收益，不建议默认启用。** 是否值得继续，只能由真实 GPT-OSS/MoE、多训练 seed 的中长程 E2E A/B 决定。

## 实现范围与语义

- 新增 `rht_seed`；`0` 完整保留旧 fixed H16×2，非零 seed 经 MurmurHash3 finalizer 映射为 32-bit Rademacher mask。
- mask 的 bit 0–15/16–31 分别控制 block32 内两个 H16；执行顺序为先符号调制、再 H16，即 `H·D(mask)`。
- HIP 以运行时 `uint32_t` 传 mask；FlyDSL 首版编译特化 mask，并纳入 kernel/autotune cache key。
- dense/grouped、single/dual quant，及 RMSNorm、GLU、dGLU 融合输出均已覆盖；row/col mask 独立传递。
- 同一 wgrad 配对操作数使用相同 mask；autograd forward 会快照可变 config，防止 forward/backward 之间修改 seed 造成静默错配。
- 人工 padding 与原始 `+0/-0` 不翻符号，保持历史 raw-zero 语义；`use_rht=False` 时 mask 强制为 0、seed 不影响量化结果，但仍校验 uint32 取值范围。

本版是验证假设用的 **random_static MVP**：同一个 32-bit mask 在一个 campaign 的所有 block、layer 和 wgrad pair 上周期复用。它不是 per-block、per-layer 或 per-step 的 dynamic transform-ID 实现。

## GPU 正确性

完整扩展在 gfx950 上构建成功。定向测试共 20 项通过：配置/FX 13 项、quantization 3 项、dense MLP/RMSNorm 3 项、grouped MLP 1 项。关键门禁包括：

- HIP 与 FlyDSL 对相同 mask 的 FP4 payload/scale 逐字节一致；
- 与独立 oracle“先按 mask 翻符号，再走旧 fixed H16”逐字节一致；
- 覆盖 BF16、row/col RHT、不规则 grouped padding、空 expert 与 tail；
- randomized 前向和 dgrad 与 fixed 逐字节相同；wgrad 会随 mask 改变，但对 FP32 参考保持合格 SNR（dense `dw1=10.65 dB`、`dw2=10.81 dB`）；
- RMSNorm→MLP、grouped MLP 以及 forward 后修改原 config 的回归测试通过；mismatched-mask 负例能检测配对错误。

静态检查通过：`compileall`、`ruff check/format`、`clang-format`、`git diff --check`。

## 性能结果

所有数字均用 GPU event；fixed/randomized 在同一进程预热后逐次轮转，避免固定执行顺序和 GPU 时钟漂移。gradient stochastic rounding 被关闭，因为当前全局 SR counter 无法在 campaign 间公平复位。

| 工作负载 | fixed H16×2 | randomized 相对 fixed | 判断 |
|---|---:|---:|---|
| Dense dual quant，`8192×2880`，500 次 | 0.0410 ms | +3.30%～+4.88% | 小 kernel 的 sign 指令可见，但绝对值约 0.0014–0.0020 ms |
| Grouped dual quant，`M=32768,N=2880,G=4`，300 次 | 0.1019 ms | +0.74%～+1.53% | 通过 2% 门槛 |
| GPT-OSS GG1 full-op，`N=5760,K=2880`，60 次 | 1.1398 ms | -1.10%～+0.60% | 处于测量波动内 |
| GPT-OSS GG2 full-op，`N=2880,K=2880`，60 次 | 1.0605 ms | +0.61%～+0.85% | 通过 2% 门槛 |
| Dense MLP 完整训练 step，3×1000 step | 1.0866/1.2356/1.1124 ms | 三个 mask 平均 +0.53%～+0.79%；总体 +0.69% | 通过 2% 门槛 |

full-op 包含 BF16 输入量化、forward、dgrad 和 wgrad；训练 step 还包含 loss、backward 与 AdamW update。结果表明 randomized RHT 的主要成本集中在 quant producer，放入完整训练路径后被摊薄。

## 训练代理结果与建议

训练代理为真实 `mlp_fp4` forward/backward/optimizer update，形状 `M=512,K=1024,I=1024`，BF16，AdamW；三个 randomized mask 与 fixed 使用相同初始化和逐 step 相同输入，并轮转执行顺序。

| RHT seed | 三次最终 loss 相对 fixed | 三次平均 |
|---:|---:|---:|
| 1 | -0.237%、+0.131%、+0.255% | +0.050% |
| 42 | -0.234%、+0.103%、+0.017% | -0.038% |
| 20260918 | -0.104%、+0.189%、-0.111% | -0.009% |

三个 mask、三次 run 合并后的平均最终 loss 差为 **+0.0009%**，远小于 run-to-run/seed 波动；这不支持“static randomized RHT 能改善 E2E 训练”的结论，也没有发现稳定性退化。

三次长跑均使用 `--preheat 200 --warmup 20 --steps 1000`。A/B/C 的 `(init seed, data seed, campaign 顺序)` 分别为 `(20260918,314159,0/1/42/20260918)`、`(20260919,314160,42/20260918/0/1)`、`(20260920,314161,20260918/1/42/0)`；执行模板为：

```bash
HIP_VISIBLE_DEVICES=4 PYTHONPATH=. python benchmark/randomized_rht_e2e.py \
  --preheat 200 --warmup 20 --steps 1000 --init-seed INIT --data-seed DATA \
  --seeds SEED_ORDER --json RESULT.json
```

建议保留本实现作为显式 opt-in 实验开关，不改变默认行为。下一阶段只有在真实 GPT-OSS/MoE recipe 上完成至少 3 个训练 seed、1k+ step，并同时记录 train/eval loss、NaN/Inf、grad norm、吞吐和达到目标 loss 所需 sample 后再做 promotion 判断。若仍无稳定收益，应停止扩展；若出现明确收益，再实现 per-layer/pair、per-step dynamic transform ID，并把 FlyDSL mask 改为运行时参数以避免 seed 导致编译缓存膨胀。
