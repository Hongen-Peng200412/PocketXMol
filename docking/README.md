# Docking 共同数据与内存输入

本目录提供精确 prepared SMILES 的公共图读取、实例坐标读取、口袋构造及原 PocketXMol free docking 接口。`pdb_id＋candidate_id` 定位沉积实例，`prepared_smiles` 定义化学图身份。公共SMILES芳香键映射为原模型类别4，与固定官方版本65488cf635c856101dbe703ac97e2f10f58e005c的parse_3d_mol相同。历史ligand_object接入直接使用旧单双键数组，其模型类别不作为新链参照；数值验收记录见[SMILES预实验](../日志/预实验（一 --二之间）/1-SMILES构图与GPU验收.md)。

`smiles.py`读取公共图和坐标；`dataset.py`构造模型输入，`sampling.py`与`evaluation.py`共享同一SMILES图。`assets.py`还保留历史read_template，`preparation.py`保留原冻结资产的准备逻辑，均不作为新训练、采样或评价的配体图入口。

## 产物位置

下列SMILES公共包与迁移坐标、清单已在服务器核对存在。正式运行直接读取已有资产。配置中的`smiles_root`、`smiles_coords_root`、`manifest_root`明确选择对应目录。

### 科学接口

```text
/storage/penghongen/AdaLigand/Ori_Data/smiles_assets/
├── smiles_graphs_v1.npz          # 精确字符串对应的重原子和无向键数组
├── smiles_symmetries_v1.npz      # 同一字符串原子顺序的压缩自同构排列
└── SMILE_coords/v1/
    └── 5irx.npz                 # 示例PDB内完整实例的SMILES顺序世界坐标

/storage/penghongen/PocketXMol/data/smiles-v1/frozen/
├── train.jsonl                  # 保留原训练冻结成员和顺序的迁移清单
├── validation.jsonl             # 保留原监督验证冻结成员和顺序
├── calibration.jsonl            # 校准实例，不并入训练
└── test.jsonl                   # 保留原测试成员、三个视图、C5及候选种子

/storage/penghongen/Adaligand_Build/Ori_Data/pocketxmol/ligand_area/
└── 9v7o/0.npy                   # 原occurrence 0的源ZYX体素标签，继续复用
```

### 其他文件

```text
/storage/penghongen/AdaLigand/Ori_Data/smiles_assets/
├── smiles_assets_v1_report.json # Matcher公共包的构图统计（运行统计）
└── SMILE_coords/*.npz           # 根部文件属于首次失败迁移，不是v1坐标接口（计算中间文件）

/storage/penghongen/PocketXMol/data/smiles-v1/
└── {train,validation,calibration,test}.jsonl # 根部四文件属于首次失败迁移（计算中间文件）

/storage/penghongen/tmp/pxm_pre_smiles_20260914/
├── cpu-v1/                     # 化学同构报告与原子映射（外部抽样审计）
├── diagnose-v1/                 # 原模型离散键类别的全量核对（运行诊断）
├── gpu-v1/                      # 预设数值门限及真实GPU比较（运行诊断）
└── bf16-repeat-v1/              # 固定输入的bf16重复性定位（运行诊断）
```

根部原型文件与`v1/`坐标包、`frozen/`清单属于不同尝试，不能混用；具体运行命令和状态只在实验日志维护。旧`ligand_objects`、旧坐标及逐CCD对称文件保留供历史运行和一次迁移审计，不作为本分支Dataset的输入。

## 公共化学图和自同构

### `smiles_graphs_v1.npz`

一份公共包保存K种精确字符串，图顺序由其`smiles`数组确定，不重新规范化输入字符串。N_total和E_total分别是连接后的全部原子数和无向键数。

| 字段 | 类型、形状和切片规则 | 示例 |
|---|---|---|
| `schema_version` | int64标量，包格式版本 | `1` |
| `smiles` | Unicode `(K,)`，精确PreparedLigand顶层字符串 | `CCO` |
| `atom_offsets` | int64 `(K+1,)`，切分element、charge和atom_in_ring；首值0，末值N_total | `[0,3,9]`表示两图分别3和6原子 |
| `element` | int16 `(N_total,)`，原子序数 | `[6,6,8]` |
| `charge` | int8 `(N_total,)`，逐原子形式电荷 | `[0,0,-1]` |
| `atom_in_ring` | bool `(N_total,4)`，3、4、5、6元环标记；原dock特征器不读取此字段 | `[False,False,False,True]` |
| `bond_offsets` | int64 `(K+1,)`，切分bond_index第二维、bond_type和bond_in_ring第一维 | `[0,2,8]` |
| `bond_index` | int32 `(2,E_total)`，每张图内部的局部原子编号，不加全局atom_offsets | `[[0,1],[1,2]]`为两条链式键 |
| `bond_type` | uint8 `(E_total,)`，0/1/2/3/4为单、双、三、配位、芳香 | `4`为芳香键 |
| `bond_in_ring` | bool `(E_total,4)`，与键对齐的3、4、5、6元环标记 | `[False,False,False,True]` |

`read_smiles_graph`将键类别映成原模型的1/2/3/4，配位键拒绝。模型数组直接读取公共包；SDF与RMSD所需RDKit Mol按同一精确SMILES首次解析并缓存，保留`[nH]`显式氢语义，核对原子、形式电荷、端点及键类型顺序。不会每次Dataset索引都解析字符串，也不会通过沉积坐标重建手性。

### `smiles_symmetries_v1.npz`

该包有自己独立的`smiles(K,)`字符串索引，不假设与图包排序一致。`schema_version`为int64标量1；`atom_count`为int32 `(K,)`，保存各图重原子数。

`matches_shape`为int32 `(K,2)`，例如`[2,2]`表示该图有两种排列、两个可交换原子；`matches_offsets`为int64 `(K+1,)`，把一维int32 `matches_iso`切分后恢复为相应形状。恢复的每个值是本图原子编号，例如`[[0,2],[2,0]]`表示原子0和2可以交换。没有可交换原子时为`(1,0)`。

排列沿用公共包的`useChirality=True、maxMatches=10000`和恒等排列补回规则。迁移审计先恢复完整排列，再按新旧原子映射比较集合。该包只用于原训练同构重分配，评价继续沿原`CalcRMS(maxMatches=30000)`，不改用训练排列代替。

## 实例坐标和冻结清单

### `SMILE_coords/v1/{pdb_id}.npz`

一个PDB一份包。K_occ是包内实例数，N_total是所有实例的完整重原子总数。

| 字段 | 类型、形状与对齐 | 示例 |
|---|---|---|
| `schema_version` | int64标量 | `1` |
| `candidate_ids` | int64 `(K_occ,)`，升序的PDB内原实例编号 | `[0,2]` |
| `prepared_smiles` | Unicode `(K_occ,)`，逐实例精确化学身份 | `["CCO","CCO"]` |
| `coord_offsets` | int64 `(K_occ+1,)`，切分coords第一维，首值0、末值N_total | `[0,3,6]` |
| `coords` | float32 `(N_total,3)`，完整沉积重原子的世界XYZ坐标，单位Å | `[12.0,8.0,-1.5]`为一个原子的三个分量 |

`read_smiles_coords`同时核对实例编号和精确字符串，然后返回对应坐标切片；每个切片的原子顺序与公共图一致。包不保存旧模板身份或图缓存编号。一次迁移的新到旧原子映射保存在验收证据中，不参与正式运行。

### 四份冻结JSONL

每个字典对应一个完整实例。原累计条件继续包括`small_molecule`、单残基单CCD、非共价、含碳、元素及键词表支持、语言表征无错误、沉积重原子完整且有限、必要受体和地图资产可用。

| 字段 | 类型与含义 | 格式示例 |
|---|---|---|
| `split` | str，训练、监督验证、校准或测试划分 | `train` |
| `pdb_id` | str，PDB目录名 | `5irx` |
| `candidate_id` | int，原occurrence编号 | `0` |
| `prepared_smiles` | str，精确公共图身份 | `CCO` |
| `n_heavy_atoms` | int，图与坐标共同重原子数 | `3` |
| `center_offset_xyz_A` | 长度3列表，冻结C5世界XYZ偏移，Å | `[1.0,0.0,0.0]` |
| `sampling_seed` | int，逐实例候选种子 | `10831` |
| `views` | list[str]，原冻结测试视图；非测试为空 | `["ALL","CAP10","HF10_TO5"]` |
| `unsupported_smiles_reason` | str，仅明确不支持实例含此字段；训练/val跳过，测试保留失败分母 | `constructed unsupported graph`为构造测试示例 |

构造格式示例：`{"split":"train","pdb_id":"train_demo","candidate_id":0,"prepared_smiles":"CCO","n_heavy_atoms":3,"center_offset_xyz_A":[1.0,0.0,0.0],"sampling_seed":10831,"views":[]}`。示例不代表实际资产编号或种子。

迁移逐项保留原清单的顺序、实例成员、偏移、种子和视图，不重新运行`freeze`。历史CAP10和HF10_TO5按原完整模板身份冻结成员；本次SMILES身份合并不改变这些已冻结集合。C5半径来自原Uniform(0,5)、方向来自均匀球面，仅评测读取；中心训练和val/loss始终C0。

### `ligand_area/{pdb_id}/{candidate_id}.npy`

继续复用已有int32 `(K,3)` 源体素ZYX索引，只包含当前occurrence。例：源`[25,30,40]`减裁块起点`[20,20,20]`得到块内`[5,10,20]`；运行时只保留三分量均在`[0,48)`的索引。K可为0，不补其他实例标签，实际体素尺寸读取地图元数据。

## 历史准备入口

`preparation.py`与`scripts/prepare_docking.py`保留原来的`index、objects、samples、freeze`流程，供追溯旧清单、旧模板对称排列和既有视图定义。其源模板编码不是新模型输入。当前生产入口直接读取`data/smiles-v1/frozen`，不运行历史准备、重新筛选或重新冻结；特别是CAP10与HF10_TO5成员仍由原模板身份分组时冻结的清单决定，不按合并后的SMILES身份重新分组。

## 口袋与原模型链

`OccurrenceDataset`按公共SMILES原子顺序装配图和坐标，再调用原FeaturizeMol、任务变换和dock噪声器。原噪声、同构重分配、固定字段恢复、loss、置信度及self-ranking保持原入口。

受体排除 UNK 后按完整残基重原子质量中心选袋：中心距离严格小于 15 Å，包络距离任一真值配体重原子严格小于 10 Å。中心模型原点为本次给定中心，包络模型为实际选入受体原子的世界坐标均值。官方只读蛋白；RA 读标准蛋白及 RNA/DNA，空蛋白不伪造节点或原点。

RA的E口袋为空时，`OccurrenceDataset.__getitem__` 在求均值前抛出 `EmptyEnvelopePocketError`，例如 `empty_envelope_pocket: 6j3z/11`。训练及val/loss的迭代流捕获该错误和清单明确标记的UnsupportedSmilesError，按划分和实例身份写警告后跳过；训练继续抽样直至组成正常batch，验证沿原路径计算其余有效E的损失。冻结JSONL及其他源资产不改写。正式采样按records逐实例直接索引，已有失败记录保存全部预算候选的preprocess错误，评价仍保留该实例分母；空E不改用配体中心或补入受体。运行时图与字符串不一致、坐标身份不一致等错误直接传播，不按已批准的少量迁移例外静默跳过。

`DataModule.setup`按pocket_mode确定中心C0或包络E，训练与原val/loss条件相同。完整配体世界坐标为X*，g=mean(X*)；中心以g选袋并定模型原点，局部目标X*−g的质心为0。原GaussianExplodePrior加噪后直接执行原同构重分配和固定字段恢复，不增加共享平移，不读取center_translation。

旧运行仍保存含center_translation字段的配置。当前续训、采样和评价的严格配置检查不会把已删除字段自动视为等价；需要读取历史运行时，使用其冻结release与原配置。当前入口服务新的获准运行，使用独立产物目录；现有正确T0结果继续有效，不更改其run.json、checkpoint或W&B。

C0/C5仅是评测提供的实际中心不同，两者使用同一T0推理入口。实际中心c用于选袋及原点C，轨迹中固定；首步原纯高斯先验，后续局部预测Z加s*sigma(N)*epsilon，其中s=1−level_dict['pos']。没有中心相关重新加噪，不强制候选实际质心归零；定位后不读取GT中心或GT偏移，最终只加回C一次。

核酸输入为`(P,15)`：元素C/N/O/P四列、A/C/G/U/DA/DC/DG/DT八列、base/sugar/phosphate三列；蛋白位置为0。蛋白输入保持`(P,25)`，核酸位置为0，`pocket_is_nucleic(P,)`对齐共同原子顺序。RA对蛋白与核酸联合构图，独立投影后使用同一个pocket_encoder；当前无独立核酸编码器或given_center_local字段。

糖组分 sugar 包含 `C1'` 至 `C5'`、`O2'` 至 `O5'`；磷酸组分 phosphate 包含 `P/OP1/OP2/OP3` 及旧名 `O1P/O2P/O3P`；其余标准核苷酸原子归 base。旧原子名的星号转为撇号，例如 `C1* → C1'`。官方空蛋白保留原 `pocket_center(0,3)`，不强制补成正常的 `(1,3)`。

训练对合法实例均匀有放回抽样；有限验证按 worker 编号跨步遍历，正常情况不漏尾部。原偶发 OOM 裁批仍单独记录，不用它解释正常遍历缺失。

历史准备入口为 `scripts/prepare_docking.py`；本次不调用。`tests/test_docking_data.py` 使用构造资产检查筛选、原子编号、几何与视图；SMILES迁移与验收记录见[预实验日志](../日志/预实验（一 --二之间）/1-SMILES构图与GPU验收.md)。本次使用迁移清单，不重新执行freeze。

官方等价验收见[test_docking_official.py](../tests/test_docking_official.py)，参照本地Git提交`65488cf635c856101dbe703ac97e2f10f58e005c`的真实原先验、信息等级、噪声器、采样循环与解码。历史六模型源码和配置保存在`463d59098bca92b7278839992afd0efc84d5db81`；旧RB/T1运行须以其对应历史源码理解，当前入口维护RA＋T0、密度分支与官方蛋白兼容。密度字段保持[密度输入说明](../models/README-density.md)中的48³、ALL56、实际体素间距与模型原点契约。收口证据与历史结果见[非密度记录](../日志/第一类实验（不加密度信息）/总日志&分析/非密度收口记录.md)。
