# 常见算法题

> 面向机器学习、深度学习、大模型方向的面试手撕代码题总结。每道题包含：题目描述 → 核心思路/公式 → Python/PyTorch 代码 → 常见追问。

---

## 目录

### 一、机器学习基础算法题

| # | 题目 | 关键词 |
|---|------|--------|
| 1 | [手写逻辑回归](./01_逻辑回归.md) | Sigmoid、交叉熵损失、梯度推导 |
| 2 | [手写 Softmax 函数](./02_Softmax.md) | 数值稳定性、exp overflow |
| 3 | [手写 K-Means 聚类](./03_KMeans.md) | 迭代聚类、质心更新 |
| 4 | [手写 KNN](./04_KNN.md) | K-近邻、距离度量 |
| 5 | [手写决策树](./05_决策树.md) | ID3、CART、信息增益、基尼系数 |
| 6 | [手写 AUC 计算](./06_AUC.md) | ROC 曲线、排序法 |
| 7 | [手写交叉熵损失函数](./07_交叉熵损失.md) | Binary CE、Multi-class CE |
| 8 | [手写 L1/L2 正则化](./08_正则化.md) | Lasso、Ridge、权重衰减 |
| 9 | [手写 Dropout](./09_Dropout.md) | 训练/推理模式、inverted dropout |

### 二、深度学习 & PyTorch 算法题

| # | 题目 | 关键词 |
|---|------|--------|
| 10 | [手写多层感知机（MLP）](./10_MLP.md) | 前向传播、PyTorch Module |
| 11 | [手写 BatchNorm / LayerNorm / RMSNorm](./11_Normalization.md) | 归一化对比 |
| 12 | [PreNorm vs PostNorm](./12_PreNorm_PostNorm.md) | Transformer 中的归一化位置 |
| 13 | [手写 Self-Attention](./13_Self_Attention.md) | Scaled Dot-Product Attention |
| 14 | [手写 Multi-Head Attention](./14_Multi_Head_Attention.md) | 多头注意力 |
| 15 | [手写 Transformer Encoder Block](./15_Transformer_Encoder.md) | 完整 Encoder Block |
| 16 | [手写 Positional Encoding](./16_Positional_Encoding.md) | 正弦余弦位置编码 |
| 17 | [手写 CNN 卷积层前向传播](./17_CNN.md) | 卷积、padding、stride |
| 18 | [手写 RNN / LSTM 前向传播](./18_RNN_LSTM.md) | 循环神经网络、门控机制 |
| 19 | [手写 Embedding 层](./19_Embedding.md) | 查表操作 |
| 20 | [手写 Dataset / DataLoader](./20_DataLoader.md) | PyTorch 数据管道 |
| 21 | [手写训练循环](./21_Training_Loop.md) | Training Loop、验证、early stopping |
| 22 | [手写梯度下降优化器](./22_Optimizer.md) | SGD、Momentum、Adam |
| 23 | [手写 Learning Rate Scheduler](./23_LR_Scheduler.md) | Warmup + Cosine Decay |

### 三、大模型相关算法题

| # | 题目 | 关键词 |
|---|------|--------|
| 24 | [手写 RoPE](./24_RoPE.md) | 旋转位置编码 |
| 25 | [手写 KV Cache](./25_KV_Cache.md) | 推理加速、自回归生成 |
| 26 | [手写 Grouped Query Attention](./26_GQA.md) | GQA、MQA |
| 27 | [手写 SwiGLU 激活函数](./27_SwiGLU.md) | GLU 变体 |
| 28 | [手写 Top-K / Top-P 采样](./28_Sampling.md) | Nucleus Sampling、Temperature |
| 29 | [手写 Beam Search](./29_Beam_Search.md) | 束搜索 |
| 30 | [手写 LoRA](./30_LoRA.md) | Low-Rank Adaptation、参数高效微调 |
| 31 | [手写 Flash Attention（简化版）](./31_Flash_Attention.md) | IO-aware、分块计算 |
| 32 | [手写 Tokenizer（BPE）](./32_BPE.md) | Byte Pair Encoding |
| 33 | [手写 RLHF Reward Model](./33_Reward_Model.md) | 奖励模型前向传播 |
| 34 | [二维注意力层（2D Attention）](./2D_Attention.md) | 表格数据、Axial Attention、Cross-Row/Column |

### 四、常见数学 & 概率题

| # | 题目 | 关键词 |
|---|------|--------|
| 35 | [手推反向传播](./34_反向传播.md) | 链式法则、计算图 |
| 36 | [手推 Attention 的梯度](./35_Attention梯度.md) | dQ、dK、dV |
| 37 | [信息熵、交叉熵、KL 散度](./36_信息论.md) | 关系推导 |
| 38 | [Bias-Variance Tradeoff](./37_Bias_Variance.md) | 偏差方差权衡 |
| 39 | [为什么 Transformer 要除以 √d_k](./38_Scaling.md) | 方差分析 |

---

## 使用说明

- 每道题一个独立的 Markdown 文件，方便单独复习
- 代码均使用 Python / PyTorch 实现
- 面向面试复习，注重**手写代码能力**和**原理理解**