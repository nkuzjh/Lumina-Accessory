# 本轮有限验收记录

日期：2026-09-30（Asia/Hong_Kong）。完成独立项目、环境、官方资产与接入实现，并执行下列有限CPU验收，包括真实模型更新/恢复和两task各1张生成图。GPU相关项目因资源被其他任务占用而未测，本页不将CPU结果等同于CUDA验收。

**RUN_FORMAL=0。未启动19,500-step正式训练、20,000/12,800样本全量推理或全量评测；未创建自动继续队列。**

## 资源、隔离与资产

唯一GPU为RTX PRO 6000 Blackwell，驱动580.173.02/CUDA driver13.0，约90,209 MiB被既有任务占用，后续快照仍仅7,041 MiB空闲。`gpu_resource_block.json`记录显式选择GPU 0时40,000 MiB预检下限未满足，未尝试分配或运算。所有本轮模型运算均显式隐藏CUDA；没有停止、暂停、重启其他进程。初始磁盘约283GB可用，主机可用RAM约950GB。

官方代码HEAD固定 `f260ba28dcd76fd6176036e9657de1bce1081a68`。官方normal Accessory资产固定 `711d5d6656c62957e8625b02ea53cc74f2c5589d`；主权重SHA256为 `b787a35ab72e8fe14b908e3795c08baa81556dd6b219b0bdc8c495c18386e700`。13/13文件完整校验通过，总21,952,950,054字节，`assets_full_check.json` 中 `ready=true`。未下载EMA/T2I DiT，未加载旧CSGO权重。

旧Lumina/ControlAR/OmniGen2/UniLIP代码、环境、结果与共享数据/评测实现只读。初始状态为 `initial_resources.txt`、`initial_git_status.json`；最终隔离检查为 `isolation_git_status_check.json`。除另行注明外，本页相对证据路径均位于 `outputs/implementation_audit/`。

## 13项验收结果

| 项目 | 实际结果及边界 | 证据 |
|---|---|---|
| 1 数据协议 | 通过：train/validation/discrete/continuous为50000/5000/20000/12800，manifest/calibration hash一致；全87,800条prompt最长145 tokens，无截断 | `data_check.json`；最终smoke的`checks/` |
| 2 输入与GT隔离 | 通过CPU数据回归：radar224/FPV448、原始物理pose与弧度；两个推理split均拦截target读取，radar可正常读取 | `tests/test_seen10_data.py`；真实生成补充见下节 |
| 3 官方初始化 | 通过真实资产：strict-load、完整keys/shapes与hash；condition未重初始化 | `real_model/model_real_audit.json` |
| 4 LoRA与optimizer | 通过真实完整模型：197目标、591个LoRA张量；完整1551项参数名单和optimizer覆盖；仅LoRA可训练 | `real_model/trainable_parameter_audit.json/.md` |
| 5 原生forward/梯度 | 真实CPU224/448通过：零LoRA输出差0、condition hash不变；冻结梯度0；全部B weight/bias有梯度，第二步全部A有梯度 | `real_model/model_real_audit.json`；CUDA未测 |
| 6 batch/累积/DDP | CPU通过：合法/非法组合、跨epoch全局流、2进程Gloo no_sync缩放和验证真实计数；真实CUDA多卡未测 | `tests/test_seen10_training.py` |
| 7 更新与resume | 真实CPU结构探针通过：2个micro1更新，保存step1后重放step2，591张量/Prodigy/scheduler/loss精确一致；生产loop另有stub恢复测试 | `real_model/model_real_audit.json`；非冷进程/GPU恢复 |
| 8 保存/选择 | CPU通过：五里程碑、完整验证计数机制、best平局较早、late仅19500、原子链接、损坏拒绝、拓扑变化标记 | `tests/test_seen10_checkpoint.py`；未执行正式5000条模型验证 |
| 9 原生推理 | tiny CPU原生/静态/CFG B2通过；torchdiffeq Euler50网格实测49 NFE，手写递推差0；真实两task各1图、每图49次原生forward通过，详见下节 | `model_cpu_validation.json`、`tests/test_seen10_inference.py` |
| 10 加速 | tiny非零模型CPU Inductor fullgraph通过；真实GPU default/reduce-overhead、连续batch/尾batch/跨task性能未测 | `model_cpu_inductor_independent.json/.log` |
| 11 推理恢复 | CPU通过：稳定样本seed、有效图跳过、坏图修复、identity冲突拒绝、原子writer异常传播；真实GPU恢复未测 | `tests/test_seen10_inference.py` |
| 12 共享评测 | 真实共享CLI对fixture与两张模型图均退出0（discrete单图、continuous frame-only）；完整FID/FVD/TWE/TDE及正式coverage未测 | 最终smoke的`evaluator_fixture/result.json`；模型图见下节 |
| 13 兼容/隔离 | 独立环境原生train/sample的`--help`退出0；旧项目状态未改，smoke/benchmark拒绝正式树，正式resume拒绝smoke | 环境入口日志、隔离检查、最终52项回归 |

最终统一命令如下，退出0；完整测试为 **52 passed，12 warnings，119.03秒**。警告为上游Apex可选回退、CPU禁用CUDA autocast及旧autocast API等，不代表CUDA算子测试。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
  bash scripts/run_csgo_seen10.sh smoke --experiment csgo_seen10_exp32gen_aligned \
  --run-root outputs/smoke_csgo_seen10_exp32gen_aligned --cpu-only
```

证据：`outputs/smoke_csgo_seen10_exp32gen_aligned/{cpu_tests.log,smoke_summary.json,checks/,evaluator_fixture/}`。首轮独立环境42项及后续局部回归日志仅作修复历史，不与最终52项累加。

## 真实模型的CPU结构验收

实际命令：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python scripts/audit_csgo_seen10_model.py \
  --output-root outputs/implementation_audit/real_model --smoke-steps 2 --verify-resume
.venv/bin/python scripts/plot_csgo_seen10_loss.py \
  --events outputs/implementation_audit/real_model/cpu_probe_events.jsonl \
  --output outputs/implementation_audit/real_model/cpu_probe_loss.png
```

真实base为2,784,546,368参数，注入LoRA后DiT为3,012,510,336，含Gemma/VAE完整bundle为5,710,671,907。可训练227,963,968，占DiT 7.5672%、bundle 3.9919%；所有197目标均进入同一Prodigy lr1/wd.01组。完整审计包含每个参数的名称、形状、数量、冻结状态、职能和optimizer归属。

canonical condition SHA256为 `e437559ce00c800ebfa452f0ce0ff2dc119747a218e5779b036a4b78acdd718d`，注入前后一致。非零合成输入输出max abs=5.57467985，零LoRA前后max差=0。

探针使用真实seen_train前两条、micro/effective batch 1、caption dropout 0、activation checkpointing、clip2、原定Prodigy与constant scheduler。loss分别0.33832714和0.48744825；冻结梯度数始终0，B weight/bias均197组有非零梯度，radar条件B bias为13组；第一步A梯度为0、第二步197组均非零，符合B零初始化语义。此两点仅验证结构和数值，不作为收敛结论。

保存step1后，在同一进程和冻结bundle内恢复并重放step2。样本、loss、591个LoRA张量、Prodigy完整state与scheduler全部逐位相同，共3次micro1更新。**这不是正式128条/update训练，也不是冷进程、BF16/CUDA或多GPU恢复证明。** 结构checkpoint标记smoke，不能用于正式resume。

总215.44秒，加载及初始探针50.44秒，两个更新各约20.79/20.29秒，其余含保存、恢复和重放；并行CPU检查及共享资源影响墙钟时间，不能外推GPU吞吐。实际adapter.pt=912,042,977字节、state.pt=1,936,353,492字节，单checkpoint约2.848GB；五份约14.242GB，另加日志/预测/环境。曲线为 `real_model/cpu_probe_loss.png`，明确标注结构探针。

## 官方基座真实图像的有限推理

使用官方normal完整基座加零初始化LoRA，**未使用结构探针更新后的LoRA，也没有CSGO训练权重**。CPU FP32、B1/VAE1，完整Euler50网格/49 NFE、CFG4/shift6/renorm1/trunc100。推理marker同时保留科学BF16配置和实际CPU/FP32运行精度，输出标记partial。执行脚本为 `run_real_cpu_inference_smoke.py`，其输出位于 `real_cpu_inference/`。

离散任务第一条 `cs_agency/file_num68_frame_421` 已生成，实测原生forward调用49次，拦截到的GT图片读取0次；448×448 RGB JPEG可完整解码。端到端单图489.66秒，随后真实共享discrete smoke退出0。此时间为CPU单图值，未做稳态/GPU速度外推。输出用于验证执行链路，不作为训练后质量或benchmark结论。

连续任务第一条 `cs_agency/file_num3_frame_205` 也已生成，端到端452.06秒；同样49次原生forward、GT读取0次，448RGB JPEG通过。真实共享continuous frame-only smoke退出0，PSNR/SSIM/LPIPS为有限数值；离散还成功计算Boundary_F1。两张JPEG分别50,961/68,645字节，整个探针含加载与两次评测共1000.54秒。只选择一帧，明确未测FID/FVD/TWE/TDE及正式覆盖，不由两张图的大小推算正式预测盘占用保证。证据为 `real_cpu_inference/smoke_summary.json`、各task的`inference_report.json`、`prediction_identity.json`和`shared_evaluator.log`。

## 环境及CPU编译

独立`.venv`为Python3.12.3、torch2.8.0+cu128、torchvision0.23.0+cu128、FlashAttention2.8.3；81包锁文件和`pip check`通过。FlashAttention CUDA扩展仅做CPU-side导入，未调用GPU kernel。证据为 `environment_check.json`、`environment_pip_check.txt`、`environment_lock.txt`。原生sample上游导入不存在且未使用的`read_general`，删除该导入后与train的`--help`均退出0。

首次新环境CPU Inductor发现系统缺Python.h，失败保留于`model_cpu_inductor_venv28_initial_failure.log`。已按Ubuntu签名apt索引取得匹配`libpython3.12-dev=3.12.3-1ubuntu0.17`，SHA256 `c123ddab7763e45199e21b4ac8545e705f20a0315b9ada4c236b52fb088c4e16`，仅在项目`.venv`解包并注册CPATH。setup幂等重跑成功，没有全局安装或借用其他项目环境。

主代理以清空外部CPATH、隐藏CUDA、全新编译缓存运行 `run_cpu_inductor_probe.py`：非零tiny模型B2 fullgraph通过，cold20.38秒，steady约0.0056秒，静态vs原生max8.94e-8，compiled vs静态max1.19e-7，重复误差0。证据 `model_cpu_inductor_independent.json/.log`。系统include目录缺失警告仍存在，实际编译由项目头文件成功完成。历史系统Python及临时借用头文件的检查不作为独立环境结论。

位置回归中真实文本L=5时condition首token为`[5,28,28]`、target首token为`[6,0,0]`；condition14×14、target28×28、caption有效长度5/3、条件调制t=1。改变radar或文本均改变非退化tiny模型输出。

## 三引擎实测范围与选择

| 引擎 | CPU证据 | 真实CUDA完整采样/性能 |
|---|---|---|
| eager | 原生tiny回归；真实forward/backward、恢复、两task各1图的49 NFE通过 | 未测 |
| compiled/default | tiny CPU Inductor fullgraph与数值对照通过；显式关闭cudagraph | 未测 |
| compiled/reduce-overhead | 接口及生命周期保护已实现 | 未测：需要CUDA Graph实测 |

目前无GPU实测支持的最快引擎推荐、B16容量保证、训练小时数或32,800张ETA。手册以eager为待补测基准。不得将tiny CPU时长或旧Lumina加速倍数外推到本模型。

## 共享评测与历史故障范围

最终wrapper生成新项目内常色RGB448 JPEG fixture，仅验证manifest join、编码/尺寸、指标资产和共享CLI。discrete读取1图；continuous只读首clip首帧。逐帧PSNR/SSIM/LPIPS与离散Boundary_F1可运行；FID/FVD/TWE/TDE、正式equal-map coverage未测，fixture marker标明非模型预测。

初期曾发现评测Python路径.resolve丢失venv语义，已修复为保留符号链接。另有过长TMPDIR触发AF_UNIX socket错误，保留 `evaluator_fixture/discrete_socket_failure.log`；仅停止本轮自己的卡住fixture子进程，改用新项目短`.tmp`后成功。公共评测实现未修改。

## GPU补测命令

只在用户选定GPU确有空闲资源时运行，路径和环境变量见 [CSGO_SEEN10.md](CSGO_SEEN10.md)：

```bash
cd /home/jiahao/task/Lumina-Accessory
CUDA_VISIBLE_DEVICES=0 .venv/bin/python train_seen10.py \
  --experiment csgo_seen10_exp32gen_aligned --micro-batch-size 1 \
  --gradient-accumulation-steps 128 --smoke --smoke-steps 2 \
  --run-root outputs/smoke_gpu_csgo_seen10_exp32gen_aligned
CUDA_VISIBLE_DEVICES=0 bash scripts/run_csgo_seen10.sh compare \
  --experiment csgo_seen10_exp32gen_aligned --checkpoint official --task all \
  --batch-size 2 --vae-batch-size 1 --output-root outputs/benchmark_engine_parity_v1
```

compare用eager重复/B1-vs-B2差异预设容差，再比较两种compile的三个完整batch、尾batch和两task；不读取GT调阈值。输出目录必须空，失败仍保存报告，重跑换目录。真实GPU训练、原生/compiled完整推理、FlashAttention/CUDA Graph和多GPU仍需补测；不存在smoke后自动继续正式实验的任务。
