# Lumina-Accessory：CSGO Benchmark v2 aligned 运行手册

本实验标注 **Lumina-Accessory (based on Lumina-Image-2.0)**。独立项目 `/home/jiahao/task/Lumina-Accessory`；上游代码固定 `f260ba28dcd76fd6176036e9657de1bce1081a68`。只使用官方已完成条件任务微调的 normal Accessory 权重 `Alpha-VLLM/Lumina-Accessory@711d5d6656c62957e8625b02ea53cc74f2c5589d`，SHA256 `b787a35ab72e8fe14b908e3795c08baa81556dd6b219b0bdc8c495c18386e700`。不加载旧CSGO或原始T2I主干。

**状态（2026-09-30）**：已完成的CPU检查及GPU未测范围见 [CSGO_ALIGNED_VALIDATION.md](CSGO_ALIGNED_VALIDATION.md)。13个官方资产均已校验；52项CPU回归、真实模型加载/LoRA梯度与同进程checkpoint重放通过。官方基座CPU在discrete/continuous各生成1张、每张49 NFE，真实共享逐帧smoke均通过。正式命令只供用户手动执行。本轮 `RUN_FORMAL=0`，不会由环境安装、check、smoke自动进入正式任务。

## 环境与官方资产

两台服务器的项目位置和数据布局不同，因此 `cd` 和 `DATA_ROOT` 不同。迁移服务器的旧 UniLIP 数据入口包含指向目录外的图片、雷达软链接，需使用真实完整 bundle 根目录，避免共享协议报 `escapes data root`。选择对应服务器的整段命令，在同一终端执行。

### 原服务器

```bash
set -euo pipefail
cd /home/jiahao/task/Lumina-Accessory
export EXP=csgo_seen10_exp32gen_aligned
export DATA_ROOT=/home/jiahao/task/UniLIP/data/csgo_benchmark_v2
export SHARED_EVAL_DIR="$PWD/../csgo_benchmark_v2_eval_general"
export MODEL_PYTHON="$PWD/.venv/bin/python"
export EVAL_PYTHON="$SHARED_EVAL_DIR/.venv/bin/python"
export RUN_ROOT="$PWD/outputs/$EXP/Lumina-Accessory/seed_42"

# 仅安装项目环境，不启动正式任务。
bash scripts/setup_csgo_seen10.sh --env-only

# 下载或补齐固定版本的官方资产；路径未覆盖时使用项目默认 checkpoints 布局。
"$MODEL_PYTHON" scripts/download_csgo_seen10_assets.py --experiment "$EXP"

# 完整 check 会遍历全部 split，并对模型资产做完整哈希；不会启动训练。
bash scripts/run_csgo_seen10.sh check --experiment "$EXP"
```

### 迁移服务器

```bash
set -euo pipefail
cd /data/jiahao/task/Lumina-Accessory
export EXP=csgo_seen10_exp32gen_aligned
export DATA_ROOT=/data/jiahao/data/csgo_benchmark_v2
export SHARED_EVAL_DIR="$PWD/../csgo_benchmark_v2_eval_general"
export MODEL_PYTHON="$PWD/.venv/bin/python"
export EVAL_PYTHON="$SHARED_EVAL_DIR/.venv/bin/python"
export RUN_ROOT="$PWD/outputs/$EXP/Lumina-Accessory/seed_42"

# 仅安装项目环境，不启动正式任务。
bash scripts/setup_csgo_seen10.sh --env-only

# 下载或补齐固定版本的官方资产；路径未覆盖时使用项目默认 checkpoints 布局。
"$MODEL_PYTHON" scripts/download_csgo_seen10_assets.py --experiment "$EXP"

# 完整 check 会遍历全部 split，并对模型资产做完整哈希；不会启动训练。
bash scripts/run_csgo_seen10.sh check --experiment "$EXP"
```

## 输入和科学配置

唯一科学配置为 `configs/csgo_seen10_exp32gen_aligned.json`。输入仅发布radar、map、当前原始物理x/y/z/pitch/yaw。角度直接使用弧度；UniLIP实际文本转为度并保留一位小数，本实验公开这项表示差异。prompt全文/数值格式在 `csgo_seen10/data.py`，hash写入身份；token超256即失败。推理数据拒绝加载target，continuous各帧独立。

radar224/FPV448，FLUX VAE×8、patch2 → 196/784图像tokens。`position_type=offset`；实验名aligned表示比较口径一致，不表示像素对齐。Gemma/VAE/DiT基座冻结，仅197个原生Linear LoRA；rank capped128，scale1，B带bias。Prodigy lr1/wd.01/constant，无外部warmup，clip2。预算128条源生成记录/update×19500=2,496,000；不存在定位loss或额外曝光。

4000、8000、12000、16000、19500保存并完整验证5000条；主结果late只在19500生成，best为五次验证最小loss、平局较早。详细三方比较和模块映射见 [CSGO_SEEN10_PLAN.md](CSGO_SEEN10_PLAN.md)。

## 有限验收

smoke先做CPU检查；资源预检确认选定GPU至少40GB空闲时才进入有限训练smoke。40GB是保护下限而非容量保证；当前共享GPU忙时用以下命令保证CPU检查。`smoke`使用独立root，不修改正式别名。

```bash
bash scripts/run_csgo_seen10.sh smoke --experiment "$EXP" --seed 42 \
  --run-root "$PWD/outputs/smoke_csgo_seen10_exp32gen_aligned" --cpu-only

# 官方真实权重的CPU结构验收：2次micro1更新，再恢复重放第2次，共3次。
# 仅核查加载、梯度/优化器、checkpoint机制，不计入正式128条/update曝光。
# 目录必须全新；重复验收更换输出目录。
PROBE_ROOT="$PWD/outputs/implementation_audit/real_model_recheck_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=16 "$MODEL_PYTHON" scripts/audit_csgo_seen10_model.py \
  --output-root "$PROBE_ROOT" --smoke-steps 2 --verify-resume
"$MODEL_PYTHON" scripts/plot_csgo_seen10_loss.py \
  --events "$PROBE_ROOT/cpu_probe_events.jsonl" \
  --output "$PROBE_ROOT/cpu_probe_loss.png"

# GPU空闲后，少量真实训练：仍为128源记录/update，但subset循环且最多3 updates。
CUDA_VISIBLE_DEVICES=0 "$MODEL_PYTHON" train_seen10.py --experiment "$EXP" --seed 42 \
  --micro-batch-size 1 --gradient-accumulation-steps 128 --smoke --smoke-steps 2 \
  --run-root "$PWD/outputs/smoke_gpu_csgo_seen10_exp32gen_aligned"

# 官方完整49 NFE；每task为3个完整batch+尾batch，比较两种compile与eager。
# 该命令加载官方基座+零初始化LoRA，不需要先运行正式训练。
CUDA_VISIBLE_DEVICES=0 bash scripts/run_csgo_seen10.sh compare --experiment "$EXP" \
  --checkpoint official --task all --batch-size 2 --vae-batch-size 1 \
  --output-root "$PWD/outputs/benchmark_engine_parity_v1"
```

`compare`要求空输出目录，不读取测试GT，不输出正式coverage。编译异常直接报错；default显式关闭CUDA Graph，reduce-overhead允许图并保护输出storage。三引擎性能需本模型实际GPU测量后选择；在未测之前，手册使用eager作为基准入口。B16/VAE4仅为候选，当前机器尚无容量/吞吐保证。

wrapper内置共享评测CPU smoke使用人工常色测试图，只验I/O和逐帧指标。本轮另以官方基座实际生成discrete/continuous各1张，并通过共享逐帧smoke，证据见验收记录；两类smoke均不验证FID/FVD/TWE/TDE。真实预测的共享smoke命令如下（先有相应生成结果）：

```bash
bash scripts/run_csgo_seen10.sh eval --experiment "$EXP" --checkpoint late --task discrete \
  --inference-engine eager --batch-size 1 --vae-batch-size 1 --eval-smoke --limit 1 --device cpu
bash scripts/run_csgo_seen10.sh eval --experiment "$EXP" --checkpoint late --task continuous \
  --inference-engine eager --batch-size 1 --vae-batch-size 1 --eval-smoke --frame-only --device cpu
```

## 手动正式训练和恢复（本轮未执行）

单卡保守组合为1×1×128；micro1仅表示小batch，实际可用显存须先验证。只在用户选择的空闲GPU上启动。多卡示例需要服务器实际有对应可用卡；当前服务器只有一张卡。

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_csgo_seen10.sh train --experiment "$EXP" --seed 42 \
  --nproc-per-node 1 --micro-batch-size 1 --gradient-accumulation-steps 128

CUDA_VISIBLE_DEVICES=0 bash scripts/run_csgo_seen10.sh train --experiment "$EXP" --seed 42 \
  --nproc-per-node 1 --micro-batch-size 1 --gradient-accumulation-steps 128 --resume latest

# 仅用于有两张可用卡的服务器：2×4×16=128。
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_csgo_seen10.sh train --experiment "$EXP" --seed 42 \
  --nproc-per-node 2 --micro-batch-size 32 --gradient-accumulation-steps 2
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_csgo_seen10.sh train --experiment "$EXP" --seed 42 \
  --nproc-per-node 2 --micro-batch-size 32 --gradient-accumulation-steps 2 --resume latest
```

其他合法组合：1×4×32、4×4×8、8×4×4。省略accum时从实际world×micro推导；不能整除128即失败。不按world扩大LR。全局样本流跨epoch连续，不使用drop_last或DistributedSampler补齐。完整checkpoint保存LoRA（含B bias）、Prodigy、scheduler、所有rank RNG、全局offset、曝光、选择结果、身份与payload SHA256。恢复同软硬件/同拓扑用相同RNG；改变合法并行组合保留预算和样本流，但明确不保证逐位相同。从更早checkpoint分支恢复用新的 `--run-root`。

旧点分支会通过不可变链接保留来源checkpoint及当时best，依赖记在`provenance/branch.json`。迁移这种分支时要带上被引用的checkpoint实体，并重建链接；不能只复制一个含失效外部链接的目录。

## best/late推理与三种引擎

默认checkpoint是late，缺失即报错。所有任务使用Euler50网格/49速度调用、CFG4、shift6、renorm1、CFG trunc100、空negative、BF16计算与FP32 ODE。每样本噪声由seed/task/sample_id的SHA256派生，与顺序/batch/重启无关；浮点输出不承诺跨硬件逐位相同。推理输出JPEG448RGB，quality75，optimize/progressive False，默认subsampling。

以下四条给出best/late各自离散/连续入口；用户选择需要的任务手动运行。

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_csgo_seen10.sh infer --experiment "$EXP" --seed 42 \
  --checkpoint late --task discrete --inference-engine eager --batch-size 1 --vae-batch-size 1
CUDA_VISIBLE_DEVICES=0 bash scripts/run_csgo_seen10.sh infer --experiment "$EXP" --seed 42 \
  --checkpoint late --task continuous --inference-engine eager --batch-size 1 --vae-batch-size 1
CUDA_VISIBLE_DEVICES=0 bash scripts/run_csgo_seen10.sh infer --experiment "$EXP" --seed 42 \
  --checkpoint best --task discrete --inference-engine eager --batch-size 1 --vae-batch-size 1
CUDA_VISIBLE_DEVICES=0 bash scripts/run_csgo_seen10.sh infer --experiment "$EXP" --seed 42 \
  --checkpoint best --task continuous --inference-engine eager --batch-size 1 --vae-batch-size 1

# all为同checkpoint先discrete再continuous；已完成且有效的JPEG会跳过。
CUDA_VISIBLE_DEVICES=0 bash scripts/run_csgo_seen10.sh infer --experiment "$EXP" --checkpoint late \
  --task all --inference-engine eager --batch-size 1 --vae-batch-size 1
CUDA_VISIBLE_DEVICES=0 bash scripts/run_csgo_seen10.sh infer --experiment "$EXP" --checkpoint best \
  --task all --inference-engine eager --batch-size 1 --vae-batch-size 1

# 三种引擎各有独立profile。compiled省略mode时固定为default。
CUDA_VISIBLE_DEVICES=0 bash scripts/run_csgo_seen10.sh infer --experiment "$EXP" --checkpoint late \
  --task all --inference-engine eager --batch-size 2 --vae-batch-size 1
CUDA_VISIBLE_DEVICES=0 bash scripts/run_csgo_seen10.sh infer --experiment "$EXP" --checkpoint late \
  --task all --inference-engine compiled --compile-mode default --batch-size 2 --vae-batch-size 1
CUDA_VISIBLE_DEVICES=0 bash scripts/run_csgo_seen10.sh infer --experiment "$EXP" --checkpoint late \
  --task all --inference-engine compiled --compile-mode reduce-overhead --batch-size 2 --vae-batch-size 1
```

resume重用相同命令，逐图检查完整解码/RGB/448；坏图重生成。checkpoint、科学配置、代码或profile不一致拒绝混写，换新root。`--limit N`仅可配独立 `--output-root` 并标记partial，不能用partial冒充正式结果。

独立性能benchmark（同参数连续3batch；目录必须全新且空）：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_csgo_seen10.sh benchmark --experiment "$EXP" \
  --checkpoint late --task discrete --inference-engine compiled --compile-mode default \
  --batch-size 2 --vae-batch-size 1 --benchmark-batches 3 \
  --output-root "$PWD/outputs/benchmark_late_default_b2_v1"
```

## 共享评测、完整性与曲线

只调用 `SHARED_EVAL_DIR/run_eval.py`，不复制或修改指标。equal-map macro、clip16/stride16、FVD224、frame gap2、min track4由共享固定配置决定。正式评测要求全部expected身份，无额外/缺失/坏图，且评测输出目录必须空。例中CPU环境为独立共享评测环境，不使用UniLIP环境。

```bash
# 明确默认eager-b1 profile：每个checkpoint分别执行两项正式评测（本轮未运行）。
export PROFILE=eager-native-b1-vae1
for CKPT in late best; do
  bash scripts/run_csgo_seen10.sh eval --experiment "$EXP" --checkpoint "$CKPT" --task discrete \
    --pred-root "$RUN_ROOT/predictions/$CKPT/$PROFILE/discrete/gen_imgs" --device cpu
  bash scripts/run_csgo_seen10.sh eval --experiment "$EXP" --checkpoint "$CKPT" --task continuous \
    --pred-root "$RUN_ROOT/predictions/$CKPT/$PROFILE/continuous/gen_imgs" --device cpu
done

# 如选择compiled profile，显式匹配引擎/mode/batch参数，防止目录猜测。
bash scripts/run_csgo_seen10.sh coverage --experiment "$EXP" --checkpoint late --task all \
  --inference-engine eager --batch-size 1 --vae-batch-size 1
bash scripts/run_csgo_seen10.sh coverage --experiment "$EXP" --checkpoint best --task all \
  --inference-engine eager --batch-size 1 --vae-batch-size 1
bash scripts/run_csgo_seen10.sh plot --experiment "$EXP"
```

`--pred-root`用于单一task，不能与`--task all`一起猜目录。重跑评测使用新的 `--output-root`；公共评测器不覆盖已有结果。完整性报告列出缺失/额外/坏图，不通过时不能提交正式结果。

## 输出目录

```text
outputs/csgo_seen10_exp32gen_aligned/Lumina-Accessory/seed_42/
  config_resolved.json
  provenance/{identity,environment,trainable_parameter_audit}.json
  train/events.jsonl
  train/checkpoints/step_00004000/ ... step_00019500/
    adapter.pt  state.pt  metadata.json  COMPLETE
  train/{best,late,latest}       # checkpoints/中也有对应引用
  train/loss.png
  predictions/<best|late>/<engine-mode-bN-vaeN>/<discrete|continuous>/
    prediction_identity.json
    gen_imgs/<map>/<file_frame>.jpg
  evaluation/<best|late>/<profile>/<discrete|continuous>/
  logs/coverage_*.json
```

每个checkpoint均有完整训练状态；CPU smoke实测adapter约0.912GB、state约1.936GB，单点合计约2.848GB、五点约14.242GB，另加预测/日志/环境。时间/磁盘公式和文本缓存估计见方案。尚无GPU实测吞吐时不提供小时数或加速倍数承诺。

## 新服务器迁移

1. 新机clone官方仓库并固定本手册所列revision，再解包 `outputs/implementation_audit/csgo_seen10_source_overlay.tar.gz` 到仓库根。先用同目录 `.sha256` 校验压缩包，再按包内 `CSGO_SOURCE_MANIFEST.json` 核对源码。该包包含所有新增和修改源码；只有 `git diff` 会遗漏未跟踪文件。不要带旧项目环境；在新机执行env-only。
2. 独立执行官方资产下载，或复制上述清单中的文件后执行 `--check`。官方组件能从固定发布源独立获取，不需要旧Lumina工程。无需下载T2I主权重、EMA或其他模型。
3. 准备完整Benchmark v2数据bundle：manifest、minimal report、splits、calibration、radars、images；保持内容不变，设置DATA_ROOT。不能扫描GT重建split或重算calibration。
4. 准备只读共享evaluator及其独立兼容环境/指标资产；设置SHARED_EVAL_DIR/EVAL_PYTHON。公共目录已有环境时只读复用。安装新评测环境/下载指标资产应使用新机的独立路径和共享evaluator提供的接口，不改公共算法。
5. 路径优先级为CLI > 环境变量 > `configs/machine.local.json` > 项目相对默认；相对路径按项目解析。可覆盖DATA_ROOT、SHARED_EVAL_DIR、MODEL_PYTHON、EVAL_PYTHON、OFFICIAL_BASE_CHECKPOINT、GEMMA_PATH、TOKENIZER_PATH、VAE_PATH、RUN_ROOT。环境变量会覆盖机器配置；机器路径不进入科学hash。
6. 运行check/CPU smoke/GPU小规模补测后再手动正式训练。改变GPU数量只允许合法乘积128，checkpoint会记录拓扑改变导致的非逐位恢复。没有可用GPU时不能占用他人训练资源。

新机源码还原示例（在源码包与校验文件所在目录执行，目标目录尚不存在）：

```bash
sha256sum -c csgo_seen10_source_overlay.sha256
git clone https://github.com/Alpha-VLLM/Lumina-Accessory.git /new/task/Lumina-Accessory
git -C /new/task/Lumina-Accessory switch --detach f260ba28dcd76fd6176036e9657de1bce1081a68
tar -xzf csgo_seen10_source_overlay.tar.gz -C /new/task/Lumina-Accessory
cd /new/task/Lumina-Accessory
bash scripts/setup_csgo_seen10.sh --env-only
```

`/new/task`是待替换的目标路径。源码包不包含`.venv`、官方权重、数据、机器路径配置或实验产物；这些按上述步骤独立准备。包内`CSGO_SOURCE_MANIFEST.json`记录每个新增/修改文件SHA256与固定上游revision，`upstream_patch.diff`只包含三个上游文件的修改，不能单独代替完整源码包。
