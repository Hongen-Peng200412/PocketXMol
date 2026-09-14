# official-SMILES

当前状态（2026-09-14）：已获正式执行授权，尚未启动；等待两个无密度训练模型完成。

| 项目 | 当前内容 |
|---|---|
| 资源 | 379402_0，实际JobId379402，gnode09，A800，16CPU |
| 配置 | configs/docking/sample-official-SMILES-test.yml |
| 产物根 | /storage/penghongen/PocketXMol/sampling/official-SMILES/test |
| 测试协议 | C0/C5/E；每实例每协议50候选、100步，batch50 |
| best／W&B／指标 | 尚未产生 |

仅使用规定官方冻结参数，不训练。

## 正式运行命令

以下命令已获准，尚未执行；启动后补充实际release、launch和时间。

```bash
bash 训练与运行/sh/sample_docking.sh official-SMILES-test
bash 训练与运行/sh/evaluate_docking.sh official-SMILES-test
```

## 只读核查与验收依据

前置验收已被接受，不重跑。启动前核对真实作业、GPU UUID、控制目录、after_lock及try_lock；原命令独立备份。

## 之前的准备与尝试

以下为旧分工下尚未启动正式复验时的准备记录，其中378693接管Matcher是当时已发生的事实；当前职责已改为379402。旧编码实验结果继续留在[原历史日志](../第一类实验（不加密度信息）/7-official.md)。

最新执行边界：前置集成及验收收口后先汇报，正式复验等待用户再次明确允许；本次新运行尚未启动。
**当前：官方芳香键编码下的C0/C5/E复验已获准，前置验收完成，等待用户再次允许；尚未启动。** 使用378693和原官方冻结pxm权重，不训练模型。新产物根为`/storage/penghongen/PocketXMol/sampling/official-SMILES/test`，配置为`configs/docking/sample-official-SMILES-test.yml`；新W&B及指标尚未产生。

## 正式运行命令

用户再次明确允许后依次执行`bash 训练与运行/sh/sample_docking.sh official-SMILES-test`与`bash 训练与运行/sh/evaluate_docking.sh official-SMILES-test`，当前未执行。每协议446实例、每实例50候选、100步；使用已有冻结C5和视图。
