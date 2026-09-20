# `models/density` 模块说明

本目录只实现当前获准的`local_cov`密度条件。`docking/density.py`负责读取既有80³源裁块并构造56维`ALL`密度通道；本目录把已选RA受体原子的49维源特征与主链标记拼成50维，硬散射到同一80³几何网格，再让六个去噪块各自读取配体原子周围的11³局部块。

## 数据流

1. `LocalCovConditioner.build_grid`把`density_input (B,56,80,80,80)`与受体硬散射网格拼成`(B,106,80,80,80)`。同一体素内的受体特征求和，越界受体原子忽略。
2. 每个去噪块完成节点更新后，`LocalCovConditioner.forward`使用该层当前配体坐标重新计算home体素。11³窗口保持以实际home为中心，越出80³裁块的位置填零。
3. 六套独立`LocalCubeEncoder`分别执行两层stride-2卷积、一层普通卷积、全局平均池化和64维投影。六层输出共用一套`LayerNorm(64)`。
4. 每层独立的零初始化`FiLMPlusCombine`执行`h * (1 + gamma) + beta`。初始化时`gamma=beta=0`，因此原PocketXMol节点特征保持不变。

训练时，每个原子分块的局部读取与卷积使用非重入激活重计算；验证和推理直接计算。`chunk_size`只允许4096或显存不足时的2048，不改变原子顺序或科学结果。
