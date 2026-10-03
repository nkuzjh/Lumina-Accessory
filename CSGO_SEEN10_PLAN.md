# Lumina-Accessory CSGO aligned 实施方案

首次审计：2026-09-30。本页在实施前建立，记录预先确定的方案，后补源码比较证据；实际完成状态与未测范围统一见 [CSGO_ALIGNED_VALIDATION.md](CSGO_ALIGNED_VALIDATION.md)，不把方案当通过记录。

## 决策与隔离

采用独立官方仓库 `/home/jiahao/task/Lumina-Accessory`，上游 `f260ba28dcd76fd6176036e9657de1bce1081a68`。正常发布主权重来自 `Alpha-VLLM/Lumina-Accessory@711d5d6656c62957e8625b02ea53cc74f2c5589d`，SHA256 `b787a35ab72e8fe14b908e3795c08baa81556dd6b219b0bdc8c495c18386e700`。不使用 EMA、旧 CSGO 或原始 T2I DiT。加载完整基座后禁止 init_cond_refiner；再注入原生 LoRA。

RUN_FORMAL=0。只允许 CPU 检查和资源允许的有限 smoke。当前 RTX PRO 6000 Blackwell 的约90 GB被其他任务使用，GPU 验收暂不运行。旧工程、环境、结果、共享数据和评测实现只读；初始状态在 outputs/implementation_audit/。

## 文件级方案与模块合同

- `configs/csgo_seen10_exp32gen_aligned.json`、`csgo_seen10/config.py`：固定科学配置；路径 CLI > 环境 > 机器配置 > 项目默认，科学身份不含机器路径。
- `csgo_seen10/data.py`：使用共享 protocol 读取发布 split，物理 pose 文本；训练/验证才加载 FPV，推理拒绝 target 读取；确定性 radar224/FPV448。
- `csgo_seen10/model.py`、`lora_audit.py`：严格官方完整加载；冻结 Gemma/VAE/DiT 基座；197 个原生 Linear LoRA；完整参数/optimizer 审计及 base/condition 不变检查。
- `csgo_seen10/training.py`、`checkpoint.py`、`train_seen10.py`：Prodigy、FP32 LoRA、BF16 autocast；全局128样本流、DDP 累积、完整里程碑验证、原子保存、所有 rank RNG/Prodigy状态恢复。
- `csgo_seen10/fast_inference.py`、`inference.py`、`infer_seen10.py`：原生 Euler50网格/49NFE、CFG4/shift6/renorm1；样本稳定seed，eager、Inductor default无图、reduce-overhead；固定真实mask/位置及条件t=1；条件缓存、VAE microbatch、有界原子 JPEG writer。
- `scripts/setup_csgo_seen10.sh`、`download_csgo_seen10_assets.py`、`csgo_seen10_assets.json`：全新独立环境、固定版本资产、校验/离线检查，不复制旧环境。
- `scripts/run_csgo_seen10.sh`、`csgo_seen10/cli.py`：check/smoke/train/infer/eval/coverage/benchmark 显式入口；共享 evaluator 子进程调用。
- `scripts/plot_csgo_seen10_loss.py`、`tests/test_seen10_*.py`：曲线及语义回归。
- `CSGO_SEEN10.md`：唯一操作手册；`CSGO_ALIGNED_VALIDATION.md`：本轮真实证据与未测项。

## 固定配方和预算

输入仅 radar、map、当前 x/y/z/pitch/yaw，Gemma 文本最大256且禁止截断。offset position；condition 196 tokens，target784 tokens。冻结 FLUX VAE mode encoding，scale0.3611/shift0.1159。原生 Linear velocity/lognorm/训练分辨率shift；caption dropout0.1仅训练。仅448原生loss：官方 finetune_accessory.py:858-863 的第二次低分辨率计算最终置零，因此不复制死计算。

LoRA rank=min(128,in,out), scale1, dropout0, A无bias, B weight/bias零初始化。真实完整模型已审计为197目标/227,963,968可训练参数；逐张量名单及optimizer覆盖见 `outputs/implementation_audit/real_model/trainable_parameter_audit.md`。统一 Prodigy lr1/betas(0.9,0.999)/eps1e-8/wd0.01/decouple/bias correction/safeguard/slice_p11/d0=1e-6/d_coef1/beta3=None；constant，无warmup，clip2。

world×micro×accum=128，19500 optimizer updates，2,496,000 generation exposures。全局流跨epoch接续；保存并全验证在4000/8000/12000/16000/19500；best最小validation loss平局选较早，late仅19500，latest为最新完整保存。主比较late。

## 风险及验收门槛

1. 官方训练 init_from 会覆盖condition；新loader必须strict加载且禁止再初始化，真实张量hash检查。
2. 官方 forward_with_cfg 含批量布尔错误；向量化按样本重标定，B1/B>1回归。
3. 原生动态packing/FlashAttention可能导致compile图断；独立固定布局保留真实mask和位置，连续batch及尾batch/跨task检查，不静默退回。
4. 编译图输出生命周期必须独立拥有存储；GPU不足时明确未测，不能声称加速通过。
5. 精确resume比较样本/RNG/loss/LoRA/Prodigy/scheduler；CPU小模型可验证状态机制，但不替代真实模型GPU测试。
6. 推理目标读取spy、manifest/calibration/hash/计数、JPEG75默认subsampling、marker一致性/坏图修复/异步错误传播。
7. 共享评测CLI真实smoke仅限足够支持的指标，partial不冒充正式覆盖。
8. 参考事实优先trainer_state/logs，三方表在审计后填入；额外官方条件预训练明确披露，不能声称总预训练相同或已排除所有CSGO重叠。

官方源码证据均固定至上游revision；本地参考审计见 reference_audit.md。最终验收由主代理核对关键证据。

## 三方比较（源码/实际训练证据，2026-09-30）

| 项目 | 官方 Lumina-Accessory | UniLIP exp32_gen（主对照） | 本次 aligned |
|---|---|---|---|
| 初始化/预训练 | Lumina-Image-2.0后经条件生成/编辑指令训练的独立发布模型；部分 Internal Data | UniLIP-1B、InternVL3-1B、SANA/生成VAE组合，已有各自预训练 | 官方Accessory normal固定发布权重，Gemma2-2B+FLUX VAE；无旧CSGO权重 |
| 输入信息 | 通用文本+可选条件图，任务不同 | radar+map+当前5DoF文本 | radar+map+当前5DoF文本，无定位分支 |
| 角度文本 | 任意任务文本 | 原始弧度转成度、1位小数 | 原始弧度repr保留精度；公开表示差异 |
| 尺寸 | 示例1024；外围resize64对齐 | radar224，FPV448 | radar224/FPV448，直接处理，16倍数 |
| 条件路径 | 条件图VAE→cond_embedder/refiner→统一DiT；Gemma文本 | 冻结视觉塔→LLM/connector→生成DiT | 官方condition路径；offset，196/784条件/目标tokens；条件t固定1 |
| 冻结/训练 | 发布metadata training_type=full_model；下游脚本支持native LoRA | 视觉塔与inactive定位head冻结；LLM/connector/生成头LoRA，latent_queries/projector全训 | Gemma/VAE/DiT所有base冻结，仅197 Linear LoRA；无新增adapter |
| LoRA | r=min(128,in,out)，scale1，无dropout，B带bias零初始化；脚本默认full_model | LLM/DiT r32 alpha64 dropout.05，connector r16 alpha64 dropout.05 | 原生r128 capped/scale1/dropout0/B bias，不拆fused QKV |
| 可训练参数/比例 | full_model基座2,784,546,368；默认字段不代表发布用LoRA | 实际log 38,910,208/2,274,580,227=1.71%；LoRA36,614,144，小模块2,296,064 | 真实完整模型197目标；227,963,968/3,012,510,336 DiT=7.567%；含Gemma/VAE共5,710,671,907，训练比例3.9919% |
| LR分组 | 实际Prodigy统一lr1（日志遗留AdamW文字不可信） | 活跃组1e-4；latent_queries/projector同1e-4，定位配置5e-4属于inactive分支 | 全部LoRA统一lr1/wd.01，包括B bias，不按卡数放大 |
| optimizer/scheduler | Prodigy，脚本lr1/wd.01/clip2；保留原生配置语义 | 实际ADAMW_TORCH，wd0，warmup.003，cosine_min1e-5 | Prodigy1.1.2/betas(.9,.999)/eps1e-8/slice_p11/d0=1e-6/d_coef1/constant/无外部warmup/clip2 |
| CFG dropout | caption_dropout.1，清空文本 | cfg_drop_prob默认.1，生成提示文本drop路径 | .1，仅清空文本，保留radar；验证0 |
| loss | Linear velocity/lognorm/分辨率训练shift；低分辨率第二项被置零 | generation flow MSE、logit-normal timestep；无定位loss | 单448原生flow loss，不运行最终置零的低分辨率计算 |
| batch/update/曝光 | 示例global8，max_steps3000000仅配置，完成历史未知 | 实际128×19500=2,496,000；50,000源记录 | world×micro×accum128，19500，2,496,000，49.92等效epoch |
| 保存/验证 | 官方脚本保存/EMA通用机制，不代表CSGO协议 | eval_strategy=no；实际最终19500；无best | 4000/8000/12000/16000/19500保存并完整验证5000条 |
| 结果选择 | 发布normal/EMA分别存在 | final | 主late仅19500；best五次loss最小，平局较早；无EMA选优 |
| sampler/NFE/CFG/shift | 不同入口默认不同，源码另有250默认；不能漏传 | 实际统一模型generate_image硬编码20步DPMSolverMultistep，CFG4.5；不将pipeline表面50步当实际 | Euler50网格，真实CPU逐图计数49NFE，CFG4/shift6/renorm1/trunc100；每条件一张 |
| precision | bf16示例，grad_precision fp32 | bf16训练/推理路径（源码/训练配置） | bf16 autocast；LoRA/累积/Prodigy优先FP32；ODE state/update FP32；CPU验收使用FP32 |
| 图像编码 | 通用示例保存格式，非CSGO合同 | eval_csgo.py:834-835 Pillow .jpg默认 | JPEG quality75/optimizeFalse/progressiveFalse/subsampling省略遵循默认 |

LoRA职能映射：UniLIP LLM的条件语言处理对应冻结Gemma加可训练LoRA的cap/context处理；生成头对应统一layers/noise_refiner/final_layer；radar视觉connector对应cond_embedder/cond_refiner；视觉编码对应冻结FLUX VAE。它们是职能比较，不声称架构一一同构，不把统一DiT再按“LLM”和“DiT”重复注入两套LoRA。

参考证据：`reference_audit.md`；`/home/jiahao/task/UniLIP/logs/csgo_1b/exp32_gen/train_20260902_014907/train.log:233-244` 实际可训练计数/optimizer表；`outputs/implementation_audit/unilip_trainable_reference.json` 为该日志逐项提取；`/home/jiahao/task/UniLIP/unilip/model/language_model/unified_unilip.py:6202,6267,6331` 为实际采样路径；官方脚本 `scripts/run_1024_finetune.sh` 及 `finetune_accessory.py:643,858-863`。所有官方代码固定为本方案所列revision。

## 预算、选择与迁移身份细节

全局128条源记录按seed+epoch置换的无限流消费，epoch末不足128直接拼下一个epoch，不丢弃、不补齐重复计权。rank/micro/accum只改变每次update内切分；组合变更时样本流offset保留，但随机流重新按step/rank派生并明确非逐位恢复。验证按固定全局样本索引派生独立noise/timestep，按rank跨步切片，不做padding，sum/count跨rank归约。

科学identity不包含机器路径；checkpoint含base、组件hash、data/split/radar/calibration/protocol/prompt、代码内容hash、seed及训练配方。官方condition身份在FP32资产加载后、计算dtype转换前求hash，另单独强制比较计算dtype下LoRA注入前后的condition hash，避免CPU/GPU使同一资产身份改变。代码内容包含新建未跟踪源码。新机器配置只替换路径，不修改科学hash。预测还记录engine/mode/batch/VAE batch、JPEG、采样、样本selection、环境与代码版本；不同profile不混写。

验证随机流对每个全局样本索引用SHA256(`[validation_seed,index,domain]`紧凑JSON)的前8字节little-endian派生seed，`noise`和`timestep`两个domain分离，防止CPU上复用同一随机数。规则保存在科学配置；分batch和rank改变不改变各样本的验证输入。

## 加速语义与验收

Eager使用官方forward_with_cfg（修复按样本norm判断）；compiled使用静态256 caption slots+196 condition+784 target，真实caption mask仍为false。真实token的RoPE：text 0..L-1；condition时间轴L，空间28..41；target时间轴L+1，空间0..27。condition仍t=1。context/cond refiner仅在每batch开始计算一次，主layers每ODE步重新计算。固定latent shape不绑定radar/FPV尺寸。默认compiled使用Inductor options关闭cudagraph；reduce-overhead允许图，结果立刻在编译区外clone拥有独立storage。无静默eager回退。

比较先以官方eager重复/不同batch差异定标；再比较相同checkpoint/输入/seed完整49NFE的两种compile，记录latent/像素误差和冷/热/端到端速度，不能按GT指标调阈值。未运行GPU检查就保留“未测”；在新实测前推荐eager作为保守基准入口，不声称compile更快。

## 时间与磁盘估计

当前尚无本模型GPU实测吞吐。正式训练时间公式 `19500×平均update秒 + 5×完整5000条验证秒 + 保存/初始化时间`；离散+连续生成时间 `32800/稳态端到端img_per_sec + 冷编译/初始化`。共享GPU竞争会影响外推，不能套用旧模型时间。

已知主权重11,138,404,986字节；官方分发Gemma组件约10.46GB，FLUX VAE335,306,212字节，tokenizer约22MB。LoRA每份FP32张量约0.912GB；五份约4.56GB，另加完整Prodigy状态、RNG与metadata。真实CPU smoke的adapter.pt为912,042,977字节、state.pt为1,936,353,492字节；完整单点约2.848GB，五点约14.242GB（状态结构相同，正式多rank RNG/metadata可略增）。预测磁盘为32800×实测平均JPEG字节，best/late或多个engine需分别乘份数。默认不做全量文本特征缓存；若显式未来加入，50000×256×2304×2≈59.0GB（不含索引与padding元数据）须独立预算。下载分块组装可能短时需额外一份主权重空间；环境/编译缓存另计。

实际额外编码证据：UniLIP `unilip/pipeline_edit.py:103` 使用 `(images*255).round().astype("uint8")`，本接入同样round后编码；官方Accessory原始sample用torchvision to_pil_image的截断路径，量化这处差异按主对照固定并写入encoding identity。

## 实际上游修改边界

其余接入代码均为新建文件；官方已有文件只修改以下三处，通用入口继续保留：

| 文件 | 实际修改与原因 | 回归证据 |
|---|---|---|
| `models_accessory/model.py` | FlashAttention改为可选导入以支持原生FP32 CPU检查；BF16/FP16缺失时明确失败。CFG norm按样本向量化并保护零分母，解决B>1布尔判断错误 | tiny原生/静态与B2 CFG数值回归；GPU未测 |
| `models_accessory/lora.py` | 原生LoRA增加可选FP32参数dtype、输入/输出dtype衔接、重复注入拒绝；保留原始rank/scale/B bias默认语义 | 零初始化不变、FP32 trainables、B bias训练/合并及冻结检查 |
| `sample_accessory.py` | 删除未使用且整个上游data模块均未定义的`read_general`导入 | 独立环境原生sample/train的`--help`均退出0 |

未修改官方transport数学，也没有把旧项目的模型或已训练checkpoint复制进来。CSGO loader绕开通用训练入口中会覆盖condition的初始化分支；checkpoint及预测记录固定上游revision和实际新建/修改源码内容hash。本轮没有git push或远程PR。
