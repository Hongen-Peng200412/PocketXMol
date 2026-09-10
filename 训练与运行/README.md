# PocketXMol 训练与运行

本目录提供 PocketXMol 的共同 CPU 准备、六个无密度实验和候选评价入口。资源申请复用现有通用提交器；实验定义由 `configs/docking/` 中的明确配置决定。

`submit_task.sh → sbatch/task.sbatch → sh/具体任务.sh` 是正式提交链。执行器在实际运行前生成完整代码副本 release，并保存该次命令与资源信息 launch。原 Pocket_Plus 的旧任务脚本保留为历史基础设施，此项目不运行它们。

## 正式命令

工作目录为服务器 `/home/penghongen/My_Project/PocketXMol`。任务脚本使用 `/storage/penghongen/PocketXMol/runtime/venv/bin/python`；该项目环境继承 `pxm_phase1` 的 Torch/PyG/Lightning、RDKit、NumPy、SciPy 等科学依赖，仅在本项目目录补充 W&B 及其依赖，版本见[requirements-docking.txt](../requirements-docking.txt)。原 `pxm_phase1` 未改动。实际环境验收见[无密度实现与准备记录](../日志/第一类实验（不加密度信息）/00-实现与共同数据准备.md)。

### 共同数据准备

四个阶段按顺序完成。一个阶段的全部数组子任务成功后，再提交下一阶段；每任务明确申请 8 核，最多 12 个并发，合计不超过 96 核。

```bash
bash 训练与运行/submit_task.sh --sh prepare.sh --resource cpu --cpus 8 -- index
bash 训练与运行/submit_task.sh --sh prepare.sh --resource cpu --cpus 8 --array 0-11 -- objects
bash 训练与运行/submit_task.sh --sh prepare.sh --resource cpu --cpus 8 --array 0-11 -- samples
bash 训练与运行/submit_task.sh --sh prepare.sh --resource cpu --cpus 8 -- freeze 12
```

`index` 承接原 split 资格并读取实例身份，`objects` 生成每模板一次的手性对称排列，`samples` 检查每个 PDB 的实例资产并拆出单实例标签，`freeze` 汇总四个集合、固定 C5 偏移与三视图。冻结所用分片数必须与此前数组一致，`12` 不代表增加 12 个冻结任务。

源 `/storage/penghongen/AdaLigand/Ori_Data` 只读。清单与统计输出为 `/storage/penghongen/PocketXMol/data`；共享对称及逐实例标签为 `/storage/penghongen/Adaligand_Build/Ori_Data/pocketxmol`。格式和字段见[共同数据说明](../docking/README.md)。不修复坏图、SMILES 或语言向量，不将中间候选文件交给训练。

### 单模型训练

六个实验名称分别为 `B-C-T0-RA`、`B-C-T1-RA`、`B-E-T0-RA`、`B-C-T0-RB`、`B-C-T1-RB`、`B-E-T0-RB`。C为中心模式，E为包络模式；中心T0训练及原val/loss使用C0真实中心，中心T1训练使用动态C5、原val/loss使用冻结C5并保留对应整体平移，包络T0使用E。该选择只读取已有center_translation，RA/RB为两种核酸编码构造。每份配置从同一官方权重开始，完整评价仍保留中心C0/C5和包络E。

在已明确获准申请新 A800 时，可按以下形式提交；它是调用格式，不代表本项目自动拥有新增 GPU 申请权：

```bash
bash 训练与运行/submit_task.sh --sh train_docking.sh --resource a800 --gpus 1 --cpus 16 -- B-C-T0-RA
```

已有获准 allocation 使用其动态命令与锁重用资源，不重复申请。真正执行的短命令为：

```bash
bash 训练与运行/sh/train_docking.sh B-C-T0-RA
```

一个 GPU 顺序执行实验，不在同一卡并行挤入六个训练。获得额外明确 GPU 授权时，不同卡各运行独立实验。无密度正式训练优先使用单卡72、累积1，nominal global batch仍为72；必要时在该模型YAML中成对改为36×2、24×3、18×4或12×6，并记录实际配置。保留原loss和偶发OOM的原reduce_batch，不增加补样本机制。

每 800 次优化器更新计算原 `val/loss`。原 ReduceLROnPlateau 使用 1% 相对改善阈值、patience=5；第三次实际下降立即停止，最多 40000 次更新。best 按最低原验证损失，last 保存恢复状态，定期 checkpoint 均保留。输出为 `/storage/penghongen/PocketXMol/training/<实验名称>/`，包含配置、源码副本、checkpoint、W&B 记录；不把训练退出一概当作科学停止条件完成。

自己的中断检查点恢复示例：

```bash
bash 训练与运行/sh/train_docking.sh B-C-T1-RA --resume /storage/penghongen/PocketXMol/training/B-C-T1-RA/checkpoints/last.ckpt
```

恢复沿用同一实验目录、W&B run id、优化器与调度状态。已因第三次下降或更新上限完成的 checkpoint 不能通过普通 resume 继续。多 worker 从相同均匀有放回分布继续，不宣称预取队列中断前后逐样本完全同序。

科学条件错误后的重训须从官方初始参数开始，不传--resume；使用原入口已有的 `--logdir` 指定独立目录，例如 `/storage/penghongen/PocketXMol/training/B-C-T0-RA-C0`。新的W&B run id由原训练入口创建，旧产物保留并在逐实验日志标记无效。该目录如需正常续训，必须同时显式传相同--logdir和它自己的last.ckpt；不能使用旧错误训练的检查点。

### 完整候选验证与测试

采样和评价的 Python 入口分别为 `scripts/sample_docking.py`、`scripts/evaluate_docking.py`，每次读取一份明确配置。六模型的 best 路径必须在运行时根据实际训练产物写入配置；官方对照固定读取原 pxm 权重，不搜索多个 checkpoint 猜测选择。

每实例 50 个候选、100 步。中心模型评 C0/C5，包络评 E，官方评 C0/C5/E；validation 与 test 分目录，三个测试视图共用候选。正式采样配置及实际命令在训练结果确定后记录到对应实验日志，不能拿 smoke 配置代替。

按用户最新执行顺序，每训练完一个模型，先完成其完整验证／测试采样、评价和结果记录，再启动下一模型。一个模型的验证CPU评价可与它自己的测试GPU采样并行；两集合结果都收口后才切换模型。

官方对照已有两份明确配置 `sample-official-validation.yml` 与 `sample-official-test.yml`。共同数据和模型验收通过后，在获准GPU内按以下短命令生成完整验证候选；测试将后缀换成 `official-test`。随后另提交8核CPU评价任务：

```bash
bash 训练与运行/sh/sample_docking.sh official-validation
bash 训练与运行/submit_task.sh --sh evaluate_docking.sh --resource cpu --cpus 8 -- official-validation
```

此处是正式调用方式，是否已经执行以及实际release和job id，以对应运行日志为准。

## 正式运行产物与控制文件

默认反馈根由项目目录名推导，为 `/home/penghongen/Feedback/PocketXMol`。以下 `<job>` 表示实际 Slurm 作业编号，例如 `400001`；`<attempt>` 表示同一 allocation 的一次实际执行。

```text
/home/penghongen/Feedback/PocketXMol/
├── releases/
│   └── PocketXMol_<发布器内容标识>/
│       ├── manifest.json       # 代码来源、可得Git版本和既有发布器内部内容标识
│       └── PocketXMol/          # 本次实际执行的完整源码与配置副本
├── launches/<job>/<attempt>/
│   ├── launch.json             # 实际release、任务脚本、资源、数组编号与运行标识
│   └── run_cmd.sh              # 执行前保存的真实动态命令
└── allocations/
    ├── pre_lock_<job>          # 仅pre_hold任务出现，删除后开始首次执行
    ├── try_lock_<job>          # after_hold任务执行结束后等待再次执行
    └── <job>/
        ├── run_cmd_<job>.sh    # 下一次执行的命令
        ├── after_lock_<job>    # 保留allocation；未经释放许可不得删除
        ├── kill_lock_<job>     # 明确获准时创建，终止当前命令后由控制器处理
        ├── out                # allocation累计标准输出
        └── err                # allocation累计标准错误
```

控制文件不是实验结果。已启动的旧 allocation 继续使用启动时的控制器和原反馈目录，不能把上述新目录套到旧作业上。接管前逐项核对 job、节点、当前进程、run_cmd、release/launch 和实际锁位置；不使用 `scancel` 替代锁协议，不删除 after_lock，不动其他任务。

代码同步使用项目的安全同步入口，保留远端现有文件和 Git 元数据。既有 release/launch 发布器内部哈希保留；不向科学数据、训练或评价代码添加哈希登记系统。已运行的 Python 来自 release，后续共享源码更新不会热替换它。

## 通用资源参数

| 参数 | 实际作用 |
|---|---|
| `--resource cpu/a100/a800/h100/h200` | 选择通用分区／QOS映射；只有 CPU 与已明确授权的 GPU 资源可用于当前任务 |
| `--cpus N` | 每个 Slurm task 核数；共同 CPU 准备明确为8 |
| `--array 0-11` | 12个数组子任务；科学分片由prepare.sh读取对应环境变量 |
| `--gpus N` | 每节点GPU数；本阶段各模型单GPU独立运行 |
| `--mem VALUE`、`--time VALUE` | 直接传给Slurm，按实际资源和任务记录设置 |
| `--pre_hold` | 启动后等明确操作pre_lock再开始 |
| `--after_hold` | 运行结束保留allocation，进入try_lock |
| `--feedback-root PATH` | 显式覆盖新任务反馈根；默认按当前项目名推导 |
| `--task-root PATH` | 冻结的项目根；所选任务脚本必须在其内部 |

未写 `--cpus` 时通用提交器的 CPU 默认仍是16；本项目的所有共同准备命令显式写8，不因复制基础设施而误用旧值。`--simple` 保留Slurm与锁、跳过release/launch，本阶段正式准备与训练均使用完整模式。跨节点DDP能力仍在通用基础设施中，本阶段不启用。

## 测试与正式运行分开

CPU构造验收使用 `ops/run_docking_checks.sh`。该入口运行pytest，为每个Slurm作业单独使用 `/storage/penghongen/tmp/pocketxmol_checks_<job>/`，不复用其它任务的pytest临时目录。它不训练正式模型、不读取held-out科学样本，不能作为正式实验命令：

```bash
bash 训练与运行/submit_task.sh --sh ops/run_docking_checks.sh --resource cpu --cpus 8 -- tests/test_docking_data.py tests/test_docking_model.py tests/test_docking_training.py tests/test_docking_sampling.py
```

数据、模型、训练控制和后续采样评价各按实际改动完成必要验收。在授权GPU的release／launch中，先检查构造资产与真实官方权重；共同清单冻结后再检查真实train／validation资产：

```bash
bash ops/run_docking_gpu_checks.sh -k official_weights
bash ops/run_docking_gpu_checks.sh -k real_data
```

前者核对RA/RB/T1、bf16原loss、36×2=72、官方参数加载与已停止检查点恢复。后者覆盖中心T0-RA、中心T1-RB和包络T0-RA的输入装配与资源验收，各运行2次更新、1个验证批，再为首条validation生成2个3步候选并完整评价；不设姿态质量阈值，不读取test划分。后者保持实际配置的batch与累积乘积为72，OOM会明确失败，便于在正式训练前按已批准的成对设置调整。两者均关闭W&B，产物只在每次launch独立的pytest临时目录，不能进入正式结果表。

真实资产门控记录整个短fit的耗时和GPU峰值显存；该耗时包含初次读取及验证，不能当作稳定每步速度。全部必要验收通过后才接入正式训练；不因一次epoch或一次验证成功而宣称完整任务完成。

实际正式命令、release、launch、job id、产物、测试结果和失败处理统一写入[总日志](../日志/总日志.md)及其链接的独立实验记录。W&B使用 `pencounkdual-111/PocketXmol_raw`，默认online；私密API key不写入配置或日志。密度实验仍等待无密度完整结果后的用户选择。
