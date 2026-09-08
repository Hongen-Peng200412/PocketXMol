# PocketXMol：给代码 agent 的修改边界与源码核对清单

版本：2026-09-08。配套 `PocketXMol_新版研究计划_v2026-09-08.md`。

本文件是**执行规格模板及本轮已查明事实**。不能把“已有一张清单”误当成实际checkpoint、数据和运行时参数已经核对完成。带 `UNRESOLVED` 的正式实验必须阻塞启动；允许先进行不依赖这些信息的实现/开发验收。

## A. 授权范围与禁止默认

| 范围 | 获准 | 不获准的自动改变 |
|---|---|---|
| 官方原模型O | 原权重加载、项目实例读取、C0/C4/E条件入口、统一评分适配 | 训练/微调、随机NA encoder、新loss/新采样规则 |
| 首轮B-C0/B-C1/B-E | 项目数据、dock-only/free-only、两口袋定义、登记的受体适配；B-C1新增线性平移 | 看测试后改lr/步数/seed/口袋/筛选/排名；新评分网络 |
| 阶段D-C/D-E | 验证集选择四读出、输入通道、RA/RB及登记训练参数 | 隐式改变原loss、用测试选配置、回写原基线结果 |
| 阶段EXT | 另行明确的预测适配、两阶段等 | 作为以上主线前置、未经登记直接全面实现 |

本轮删除的是**新提出的4 Å径向截断**，不是原高斯内部标准差clamp。$w(s)=s$不再搜索。阶段2新增的调参权限不取消“原loss保留”约定。

## B. 正式启动前必须填入的锁定信息

```yaml
spec_version: '2026-09-08'
status: blocked_spec
source:
  reference_repository: UNRESOLVED
  reference_commit: UNRESOLVED
  project_commit: UNRESOLVED
  approved_patch_hash: UNRESOLVED
checkpoint:
  official_source_record: UNRESOLVED
  path: UNRESOLVED
  sha256: UNRESOLVED
  embedded_config_hash: UNRESOLVED
  load_semantics: UNRESOLVED # weights_only / resume_training；不可混用
training:
  resolved_config_hash: UNRESOLVED
  actual_optimizer_defaults: UNRESOLVED
  actual_parameter_groups: UNRESOLVED
  actual_scheduler_state_schema: UNRESOLVED
  local_batch_size: UNRESOLVED
  world_size: UNRESOLVED
  accumulation_steps: UNRESOLVED
  sample_stream_rule: UNRESOLVED
  seed_set: UNRESOLVED
  stop_and_checkpoint_rule: UNRESOLVED
evaluation:
  ranking_pipeline: UNRESOLVED
  scoring_weights_sha256: UNRESOLVED
  num_candidates: UNRESOLVED
  actual_forward_steps: UNRESOLVED
  rng_key_rule: UNRESOLVED
  main_metric: UNRESOLVED
  main_aggregation: UNRESOLVED
  gate_view: UNRESOLVED
  translation_cancel_rule: UNRESOLVED
assets:
  train_manifest_hash: UNRESOLVED
  val_manifest_hash: UNRESOLVED
  test_ALL_hash: UNRESOLVED
  test_CAP10_hash: UNRESOLVED
  test_HF10_TO5_hash: UNRESOLVED
  val_C4_offset_hash: UNRESOLVED
  test_C4_offset_hash: UNRESOLVED
  chemistry_identity_rule: UNRESOLVED
  occurrence_candidate_mapping_hash: UNRESOLVED
  density_geometry_version: UNRESOLVED
```

这里是字段规格，不是可直接运行的配置。若未使用单独评分权重，应显式填 `not_applicable` 并说明采用哪个原排序量，不能空值默认一个新评分器。

## C. 本轮已经核对的源码快照

```text
repo: Hongen-Peng200412/PocketXMol
commit: df1cf846711c3ff7e6284f41cb4b180e8947bac7
```

文件blob SHA：

| 文件 | blob SHA |
|---|---|
| configs/train/train_pxm_reduced.yml | 17d67c714f83072faf370d9ef96d3eefb18fa131 |
| models/loss.py | c5d9aa48c94cb5f06292c895000993bcf2cbcf48 |
| scripts/train_pl.py | a8f2e3bc856f50f9892c6c92ad7951fb912244a1 |
| utils/train.py | 98ff2a0ace1dbf5a52c278db48978fa0cd1716a3 |
| utils/prior.py | 4a406c339c2d739e72f12da794e7d899f93ebb64 |
| configs/sample/examples/dock_smallmol.yml | 8de0e462b85c1c61f8f509a01f82cd47edca9af4 |

另读官方 pengxingang/PocketXMol：README.md、docs/sample_test_sets.md。官方文档给出了Zenodo权重入口、sample_use.py示例及believe.py/rank_pose.py评分流程；**实际checkpoint尚未下载、未加载**。

以下数值仅描述这个公开配置及读取的代码。它们不是已证明匹配用户最终checkpoint的全部训练参数，也不是授权agent不核实就采用的默认。

### C1. 模型结构原值

| 完整字段 | 读取值 |
|---|---|
| model.name | pm_asym_denoiser |
| model.pocket_dim | 128 |
| model.node_dim | 320 |
| model.edge_dim | 96 |
| model.addition_node_features | ['is_peptide'] |
| model.add_output | ['confidence'] |
| model.pocket.edge_dim | 32 |
| model.pocket.hidden_dim | 128 |
| model.pocket.num_blocks | 4 |
| model.pocket.dist_cfg.num_gaussians | 32 |
| model.denoiser.hidden_dim | 320 |
| model.denoiser.num_blocks | 6 |
| model.denoiser.dist_cfg.stop | 15 |
| model.denoiser.dist_cfg.num_gaussians | 64 |
| model.denoiser.gate_dim | 2 |
| model.denoiser.context_cfg.edge_dim | 128 |
| model.denoiser.context_cfg.knn | 32 |
| model.denoiser.context_cfg.dist_cfg.stop | 20 |
| model.denoiser.context_cfg.dist_cfg.num_gaussians | 64 |
| model.denoiser.context_cfg.dist_cfg.type_ | linear |

核查构造函数的额外默认（激活、归一化、dropout、实际边方向等）并展开记录，不能只复制YAML就宣称全部相同。

### C2. Loss原值与**真实优化总键**

| 字段 | 读取值 |
|---|---|
| loss.name | individual_tasks |
| loss.weights.node | 1.5 |
| loss.weights.pos | 2.5 |
| loss.weights.edge | 1.5 |
| loss.weights.dist | 0.0005 |
| loss.weights.dih | 0.0005 |
| loss.weights.fixed_node | 0.1 |
| loss.weights.fixed_pos | 0.1 |
| loss.weights.fixed_edge | 0.1 |
| loss.weights.fixed_dist | 0.0005 |
| loss.confidence.prob_1A | 0.2 |
| loss.confidence.weights.node | 1 |
| loss.confidence.weights.pos | 20 |
| loss.confidence.weights.edge | 1 |

原task列表含denovo/conf/sbdd/dock/maskfill/fbdd/pepdesign。可以只提供dock样本/调度，但不能因此随意改变损失的计算路径。

**本轮查明的高风险细节：** `IndividualTasksLoss.forward()` 先计算各task与mixed的基础总和，然后把ConfidenceLoss的`cfd_total`仅加到`mixed/total`。可选pocket_dist项也加在`mixed/total`。`ModelLightning.training_step()` 优先取`loss`，无此键时使用`mixed/total`。

所以把训练返回值从`mixed/total`改成`dock/total`，会在该实现中漏掉置信度等追加项。不能因为只训练dock就做这个替换。

各项计算必须保留：

| 项 | 源行为 | 禁止简化 |
|---|---|---|
| node/halfedge | 未归约CE，按fixed/unfixed与task掩码再归约 | 固定化学类型后删除分类头/固定项 |
| pos | XYZ逐元素MSE，按相同掩码平均 | 换成每分子RMSD、额外除N或只用坐标 |
| dist | 相同domain/固定距离掩码定义 | 直接改为所有键或所有pair距离损失 |
| dih | 原tor/dihedral元数据与1-cos定义 | 删除字段或换手写角度差 |
| empty mask | 源代码返回0的规则 | 空项nan、改变分母 |
| size_weighted | 当前配置缺省，源码默认False；仍有既有尺寸归一化代码路径 | 自己加按N加权 |
| confidence | 原ConfidenceLoss实现、detach/target/reduction | 自造成功概率loss或遗漏 |
| pocket_dist | 此配置未启用；若所选checkpoint启用需保留 | 默认开启新受体距离约束 |

某项在一个free样本上为零，不等于获准删掉它或其字段。`ConfidenceLoss`完整细节和选定runtime仍需做全项数值/梯度核对。

D3仅追加 $\eta_{\mathrm{seg}}(0.7L_{\mathrm{focal}}+0.3L_{\mathrm{Dice}})$。追加到**真实被反传的总键**，同时保留未追加的原loss日志以验收。

### C3. Dock噪声原值

| 字段 | 读取值 |
|---|---|
| noise.dock.reassign_in | True |
| prior.pos.name | allpos |
| prior.pos.pos.name | gaussian_simple |
| prior.pos.pos.sigma_func | sqrt |
| prior.pos.pos.sigma_max | 1.0 |
| level.name | uniform |
| level.min / max | 0.0 / 1.0 |
| transform.dock.free / flexible | 0.999 / 0.001 |

本项目已批准free=1、flexible=0。配置中inactive translation/rotation/torsional参数不等于free路径使用了它们；可保留未激活字段，不为“清理”改变模型。新T1是独立登记的整体位移扩展，不能误启用原flexible路径。

`GaussianExplodePrior.add_noise()`原行为：

```text
noise = 按原sigma规则得到的逐原子高斯
pert  = x + (1-info_level)*noise
from_prior且info_level==0时: pert = noise
info_level==1时: pert = x
```

sqrt尺度使用原 `sigma[:,None].clamp(min=1)`；保留此操作。不要改用名字相近的`gaussian`，后者具有信号衰减，是另一种prior。

T1普通分支新增$s\,(\mathrm{given\_center}-\operatorname{centroid}(\mathrm{clean\_input}))$，训练clean_input为GT，推理重加噪为当前预测；纯先验分支跳过新增平移。源条件/target不改变。$w(s)=s$，不新增径向clip。按同一生成样本验证step/info_level与所有广播维度。

### C4. 配体/口袋特征原值

| 字段 | 读取值 |
|---|---|
| transforms.featurizer.use_mask_node | True |
| transforms.featurizer.use_mask_edge | True |
| atomic_numbers（顺序不可改） | [6,7,8,9,15,16,17,5,35,53,34] |
| mol_bond_types（顺序不可改） | [1,2,3,4] |
| transforms.featurizer_pocket.knn | 32 |

需实际核查：mask类别编码位置、半边图方向/顺序、显式氢处理、蛋白元素/残基编码顺序、坐标重心逻辑、固定标志、tor/domain/同构置换等元数据。不得把“保持化学图”误解为删除原halfedge完整表示。

源码DataModule先执行FeaturizePocket，再FeaturizeMol，原因是分子要减去pocket center；之后task transform、noiser。新增中心裁剪必须在口袋特征化之前落实，而不是加噪后改一下center字段却不重裁口袋。

### C5. 优化器与训练原值

| 字段 | 读取值 |
|---|---|
| train.seed | 2023 |
| train.batch_size | 40 |
| train.num_workers | 2 |
| train.pin_memory | True |
| train.persistent_workers | True |
| train.max_steps | 180000 |
| train.ckpt_every_n_steps | 1000 |
| train.val_check_interval | 1000 |
| train.precision | bf16-mixed |
| train.gradient_clip_val | 未启用（YAML为注释）；Trainer用缺省None |
| optimizer.type | adamw |
| optimizer.lr | 0.001 |
| optimizer.weight_decay | 0.001 |
| optimizer.beta1 / beta2 | 0.99 / 0.999 |
| scheduler.warmup_step | 1000 |
| scheduler.instance.type | plateau |
| scheduler.factor | 0.75 |
| scheduler.patience | 20 |
| scheduler.min_lr | 0.00001 |
| scheduler.cooldown | 10 |
| scheduler.params.interval | step |
| scheduler.params.frequency | 1000 |
| scheduler.params.monitor | val/loss |

代码实际：`get_optimizer`使用model.parameters()，无旧/新参数分组；实际torch默认eps/amsgrad/fused等需从运行环境记录。`on_before_optimizer_step`在global_step<warmup时按(step+1)/warmup改变lr；不是创建一个新的cosine或替代warmup类。

Trainer读取num_gpus/world_size、max_epochs=1、max_steps、precision、ddp等。batch40为DataLoader每进程设置，不能直接写成全局有效batch40。全局样本数受world_size与accumulation影响。

验证集在源码中使用同一个train-mode transform/noiser；项目固定C0/C4/E完整采样验证是另一个评价入口，不能未授权取代训练scheduler所用val/loss。

`model.pretrained`只加载model.*权重；`--resume`经Trainer恢复完整训练状态，两者语义不同。重训/适配开始时由用户选择，不互换。

源码OOM处理会缩小当前batch。正式运行前做内存验收并记录触发；不能静默把丢掉实例后的训练当作完全预算匹配，也不能未经批准修改有效batch/累积/reduction来“等价修复”。

无墙钟配额不等于无训练停止定义；max_steps或原停止准则必须有明确来源，禁止看测试决定多训多久。

### C6. 中心示例采样原值

| 字段 | dock_smallmol.yml读取值 |
|---|---|
| sample.seed | 2024 |
| sample.batch_size | 50 |
| sample.num_mols | 100 |
| sample.save_traj_prob | 0.05 |
| data.is_pep | False |
| data.pocket_args.radius | 15 |
| transforms.featurizer_pocket.center | 与pocket_coord相同 |
| task.name/transform.name | dock / dock |
| task.transform.settings.free/flexible | 1 / 0 |
| noise.name | dock |
| noise.num_steps | 100 |
| noise.prior | from_train |
| noise.level.name | advance |
| noise.level.min/max | 0 / 1 |
| scale_start / scale_end / width | 0.99999 / 0.00001 / 3 |

这些是示例入口，不等于完整官方benchmark流水线默认。官方benchmark有sample_drug3d.py、believe.py、rank_pose.py等，实际评分权重/聚合必须锁定。不得在C0/C4/E之间偷偷改变候选数。

训练uniform与采样advance是原流程已有的映射差异，不直接把原始循环step当成s。纯先验启动与warm-start也不能混用。

## D. 逐项修改注册表

状态：LOCK=按实际原版继承；ALLOW=已经批准的改变；REGISTER=阶段2须显式登记后可调；CHECK=待核对/决定；DEFER=本轮不开展。

| ID | 项目/作用域 | 规则与边界 | 状态 |
|---|---|---|---|
| A01 | 仓库与依赖 | 锁commit、环境包版本，不默认使用最新版替代 | CHECK |
| A02 | 权重 | 锁checkpoint来源/hash；官方权重不训练 | LOCK/CHECK |
| A03 | 初始化vs恢复 | weights-only与resume分别登记，禁止互换 | CHECK |
| A04 | 任务调度 | 本项目只生成dock样本，不再混其他任务 | ALLOW |
| A05 | free设置 | free=1/flexible=0；不新增tor/rigid噪声 | ALLOW |
| A06 | 化学固定字段 | 原fixed_node/halfedge/pos/half_dist语义与shape不变 | LOCK |
| A07 | 图词表 | 元素、键级、mask token与原索引顺序不变 | LOCK |
| A08 | 蛋白输入 | 原蛋白feature不因NA扩展重排/重新编号 | LOCK |
| A09 | NA | RA/RB；RB复制兼容GNN权重；首轮固定一个 | ALLOW/CHECK |
| A10 | KNN | 首轮同原数量、方向、batch归属；类型配额另登记 | LOCK/REGISTER |
| A11 | 原输出头 | 保留pred_node/pred_halfedge/pred_pos/confidence | LOCK |
| A12 | 原loss | 保留全部实现、权重、掩码、归约、target、总键 | LOCK |
| A13 | 原prior | gaussian_simple、实际sigma分子数规则不变 | LOCK |
| A14 | 原重分配 | reassign_in及调用次序继承，T对照不能不同 | LOCK |
| A15 | 训练数据 | 换为项目合法occurrence清单，split隔离 | ALLOW |
| A16 | 数据抽样 | 重组为项目数据需明确实例频率；不默认PDB均匀/高频抑制 | CHECK |
| A17 | 中心口袋 | 给定中心定义、C模型训练delta4重新抽样 | ALLOW |
| A18 | 包络口袋 | 原GT逐原子包络与完整残基选择函数 | LOCK/ALLOW |
| A19 | 局部原点 | C按给定中心；E沿原入口；地图另做坐标映射 | CHECK |
| A20 | T新增项 | 仅B-C1及被选中D-C，s*delta，共享增强向量 | ALLOW |
| A21 | T推理 | 纯prior不重复加T，后续按预测中心差；无径向clip | ALLOW/CHECK |
| A22 | 中心固定偏移 | val/test按occurrence保存，视图共享 | ALLOW |
| A23 | 缺失原子 | observed图训练、完整图推理、评价mask，不假造坐标 | ALLOW |
| A24 | D1–D4 | 同注入接口，阶段2选择，无密度模型不读density | ALLOW/REGISTER |
| A25 | 密度坐标 | 复用已核实变换，禁止猜origin/spacing | CHECK |
| A26 | 密度输入 | 实验为主；sim/diff/56只用受体，版本固定 | ALLOW/REGISTER |
| A27 | 门控 | alpha与g分开；$w(s)=s$不参与搜索 | ALLOW/REGISTER |
| A28 | D3 loss | 0.7focal+0.3dice，等类别权重；eta验证登记 | ALLOW/REGISTER |
| A29 | D3语言向量 | 仅化学信息；版本/身份/冻结状态核对 | CHECK/REGISTER |
| A30 | optimizer/scheduler | 首轮原生参数组/计划；阶段2每改动单独注册 | LOCK/REGISTER |
| A31 | freeze范围 | 原主干是否训练/部分freeze不能由agent惯例决定 | CHECK/REGISTER |
| A32 | batch/precision | 首轮继承并记录global实际值；不偷偷OOM减批 | LOCK/CHECK |
| A33 | 训练长度/选择ckpt | 使用预定规则；不能看测试延长或择优 | LOCK/CHECK |
| A34 | evaluation sampling | N候选、步数、seed和排名全部预冻结 | LOCK/CHECK |
| A35 | 官方评分 | 复用选定原流程，无新评分器 | LOCK/CHECK |
| A36 | 日志/缓存 | 可增监控与等价缓存，不改变sample stream | ALLOW |
| A37 | 正式测试 | (1)/(1)'/(1)''测试即正式，无效果驱动重训 | ALLOW |
| A38 | 三清单 | 模型无关生成、nested可复用，不当三次独立证据 | ALLOW/CHECK |
| A39 | 两阶段/FFT等 | 先留接口，不抢占主线，不强制三折/新评分 | DEFER |
| A40 | 未列变化 | 必须加入approved_delta并标来源，不以“优化”掩盖 | CHECK |

## E. 运行时行为核对：不能只比较配置文本

每个实验必须展开下列信息，即使值由默认给出：

| 类别 | 逐项记录 |
|---|---|
| 权重 | checkpoint键映射、missing/unexpected key、形状、初始化随机种子、新参数清单 |
| 参数训练性 | requires_grad清单、复制NA权重与独立参数对象验证、共享RA别名验证 |
| 图 | 原子/边词表、mask位置、图连接与方向、半边索引、batch偏移、KNN设置 |
| 坐标 | 世界/模型/体素三个系、原点、中心定义、单位、共同变换、数值dtype |
| 加噪 | per-occurrence实际info_level、sigma、N、delta、from_prior、T触发条件 |
| loss | 每项数值、有效元素数、归约、权重、最终反传键、confidence target |
| optimizer | 每个param group完整内容、参数覆盖、eps等runtime默认、冻结参数是否误入 |
| scheduler | warmup具体调用次序、监控量、step/epoch频率、计数状态、drop触发 |
| 数据流 | 训练实例频率、shuffle、rank/worker分片、种子、重复、batch/accumulation |
| train/val | 各transform及调用顺序、val_loss和val_pose的不同用途 |
| sampler | 纯先验/普通步、预测写回、实际step计划、重分配/固定恢复/后处理 |
| ranking | 所有候选评分函数/权重、是否读density、是否二次改坐标、排序tie-break |
| evaluation | 指定occurrence、化学等价映射、observed_mask、无刚体对齐、失败分母 |
| provenance | manifest版本、固定偏移、完整配置hash、代码diff、结果文件hash |

`strict=False`不能作为泛化兼容策略。可以仅为已登记新增键建立白名单，并断言所有原参数已正确加载；任何其他missing/unexpected/shape mismatch都必须失败。

## F. 三个入口的最小工程结构

以下名字为接口提案，不是已实现命令。

```text
prepare_manifests(spec)
  -> train/val/test occurrence references
  -> ALL/CAP10/HF10_TO5
  -> immutable C4 center offsets

run_original_eval(spec, protocols=[C0,C4,E])
  -> load unchanged official weights
  -> deterministic inputs/candidates/ranking
  -> one full prediction pool per protocol
  -> three view reports

train_registered_experiment(experiment_id, signed_spec)
  -> verify approved_delta
  -> train from declared initialization
  -> save complete states/logs

run_formal_eval(experiment_id, frozen_checkpoint, spec)
  -> reject unresolved fields
  -> fixed candidate budget and ranking
  -> save full scores even if poor

optimize_on_validation(experiment_family, registered_trials)
  -> validation C0/C4 or E only
  -> freeze winner without test access
  -> final formal evaluation
```

阶段依赖由研究计划的DAG管理，不让一键脚本默认执行全部训练或全部测试。一次运行原模型C0/C4/E只是同一原模型的三个输入条件。

## G. 正式运行前验收清单

### G1. 原版函数与训练回归

1. 蛋白-only、D0、T0、相同输入/RNG下，原模型和扩展骨架的所有原输出头逐项一致。
2. 相同batch/outputs下，原与新loss各项、mixed/total、confidence总量一致。
3. 相同权重/噪声下比较原参数梯度；不是只看loss数量级。
4. 受控小样本执行一次优化器更新，核对参数组、warmup、权重衰减的实际作用。
5. task只剩dock不导致使用dock/total漏confidence；fixed化学头仍在图中。
6. domain/tor/dihedral空张量按原约定工作，不出现为了“清理free”而删字段。

### G2. 中心与平移

7. 给定delta后，pocket和map中心相同；无密度与有密度组使用相同定义。
8. $s=0$新增T严格为零；$s=1$普通路径新增$T=\delta$；GT target未变。
9. 新T对同一配体各原子完全相同，不改变内部相对坐标（相对普通Gaussian输入）。
10. 高斯基础随机数不因T开关而改变；中心与原子noise使用隔离RNG/预生成值。
11. from_prior初始步没有重复T；T推理不读取GT delta。
12. 预测中心远于4 Å时不径向截断；不能误把该测试改成将坐标夹回球内。
13. C0协议不意外关闭T1；C4偏移每次评测相同且不随模型改变。
14. s来自实际info_level；测试采样advance下不能直接取raw step。

### G3. 数据与测试清单

15. (pdb,occurrence)唯一，candidate映射显式，重复导出不计多个实例。
16. cap10/HF10_TO5按ALL频数判断；6–10组保持原数；同一随机排序可复现。
17. 全集/子集同occurrence预测不变；索引顺序改变不改变jitter或候选seed。
18. 测试C4与验证C4分别冻结；train不断重采样。
19. 非共价/单残基/词表等筛选与模型结果无关，排除理由可追踪。
20. 程序失败不静默删除分母；共同可执行子集提前定义。

### G4. 地图与密度

21. 非零origin、非单位spacing、轴交换、共同旋转、边界的atom-grid-world往返通过。
22. U-Net/源图归一化只用有效体素；越界不夹边冒充信号。
23. D2并集去重且每原子可读全部并集；D4只读自身邻域。
24. 空密度邻域返回零残差且不nan；部分map覆盖样本保留。
25. Q/K距离偏置等价测试同时比较输出、输入梯度与lambda梯度。
26. 密度分支小门控下有梯度；关闭新增分支能恢复对应D0计算。
27. D3仅用预测top-K；GT label针对当前实例，不把其他同身份副本当作自动替代目标。
28. label I/O优化与feature缓存不改变训练抽样频率/计算图。

### G5. 评价与研究治理

29. 对称、缺失原子、同身份多实例都按预定occurrence评价；无配体单独刚体对齐。
30. 评测层可用GT构造C0/C4/E，模型输入层不能访问其他GT提示。
31. 原排名条件固定；top-of-all明确为oracle不是可部署Top-1。
32. 首轮测试结果一旦打开，参数/清单/种子/排名不得改动；错误修复另留审计。
33. 24个汇总单元均保存，负结果不删；不能把重叠视图当独立重复。
34. T取消规则预先冻结，最终报告说明一次测试驱动选择；无自动“显著性不够再多训”。

## H. 产物与签字

正式运行前交付：

```text
reference_code_manifest.json
checkpoint_manifest.json
reference_effective_config.yaml
approved_delta.yaml
resolved_experiment_config.yaml
runtime_defaults.json
loss_equivalence_report.json
input_and_noise_equivalence_report.json
dataset_and_protocol_manifest.json
preregistered_evaluation_and_gate.yaml
```

每条CHECK填为来源事实或明确用户决定；不适用项写明原因。实验只在其依赖CHECK全部解决后进入ready。

这是可追踪的“完整列出”机制，不是声称本轮已经得到未知服务器/权重事实。当前未执行训练或测试，不能在交付表填入完成日期、GPU成本或成功率。
