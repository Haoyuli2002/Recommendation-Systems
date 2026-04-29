# 二维注意力层（Two-Dimensional Attention Layer）详细解析

## 1. 整体思路：为什么需要"二维"注意力？

这段代码的应用场景是**表格数据（Tabular Data）**。想象你有一张表格：

```
         Col_1    Col_2    Col_3    Col_4
Row_0    [h00]    [h01]    [h02]    [h03]     ← 比如 query 行
Row_1    [h10]    [h11]    [h12]    [h13]     ← context 行
Row_2    [h20]    [h21]    [h22]    [h23]     ← context 行
Row_3    [h30]    [h31]    [h32]    [h33]     ← context 行
```

每个 `[hij]` 是一个 `hidden_size` 维的向量，代表第 `i` 行、第 `j` 列的单元格表示。

普通的 1D Transformer 会把所有单元格拉平成一个长序列做 attention，但这样：
- 序列长度 = `num_rows × num_columns`，计算量是 O((R×C)²)，太大了
- 丢失了表格的二维结构信息

**二维注意力的核心想法：把一次全局 attention 拆成两步——先沿列方向做 attention（同一行内的各列互相看），再沿行方向做 attention（同一列内的各行互相看）。** 这就像先"横着看"，再"竖着看"，分别建模行内关系和列内关系。计算量降低到 O(R×C² + C×R²)。

---

## 2. 整体架构

```
TwoDimensionalAttentionLayer
├── cross_column_layer (TorchRobertaLayer)   ← 第一步：行内attention（同行不同列之间）
└── cross_row_layer (TorchRobertaLayer)      ← 第二步：列内attention（同列不同行之间）
```

每个 `TorchRobertaLayer` 就是一个标准的 Transformer encoder layer：

```
TorchRobertaLayer
├── TorchAttention
│   ├── TorchSelfAttention (Q/K/V 投影 + scaled_dot_product_attention)
│   └── RobertaSelfOutput (残差连接 + LayerNorm)
├── RobertaIntermediate (FFN 第一层，扩展维度)
└── RobertaOutput (FFN 第二层 + 残差连接 + LayerNorm)
```

---

## 3. Forward 流程详解

### 输入

- `hidden_states`: shape `(num_rows, num_columns, hidden_size)` — 表格的每个单元格一个向量
- `attention_mask`: shape `(num_rows, num_rows)`，值为 `0`（允许 attend）或 `-inf`（禁止 attend）

### 第一步：Cross-Column Attention（列间注意力 / 行内注意力）

```python
horizontal_outputs = self.cross_column_layer(hidden_states)
```

**关键理解：** `hidden_states` 的 shape 是 `(num_rows, num_columns, hidden_size)`。当我们把它直接喂给一个标准 Transformer layer 时，PyTorch 的 attention 会把：
- **第 0 维 (`num_rows`)** 当作 **batch 维度**
- **第 1 维 (`num_columns`)** 当作 **序列长度（seq_len）**

所以实际上做的是：**对于每一行（独立地），让该行的各列之间做 self-attention**。

```
Row_0: [h00] ←→ [h01] ←→ [h02] ←→ [h03]   (这4个token之间做full attention)
Row_1: [h10] ←→ [h11] ←→ [h12] ←→ [h13]   (独立的另一个batch)
Row_2: [h20] ←→ [h21] ←→ [h22] ←→ [h23]
...
```

**没有 attention mask**，因为同一行内所有列都可以互相看到（full attention）。

**分块处理（Chunking）：** 代码中用了 `max_rows_per_batch = 8192` 来分块，避免一次性放太多行进 GPU 显存溢出。`batch_step` 根据列数动态调整。

### 第二步：Cross-Row Attention（行间注意力 / 列内注意力）

```python
horizontal_outputs = horizontal_outputs.transpose(0, 1).contiguous()
# 现在 shape: (num_columns, num_rows, hidden_size)

vertical_outputs = self.cross_row_layer(horizontal_outputs, attention_mask)
# 对于每一列（独立地），让该列的各行之间做 self-attention

vertical_outputs = vertical_outputs.transpose(0, 1).contiguous()
# 再转回 shape: (num_rows, num_columns, hidden_size)
```

**关键操作：** 通过 `.transpose(0, 1)` 把 shape 从 `(num_rows, num_columns, hidden_size)` 变成 `(num_columns, num_rows, hidden_size)`。这样：
- **第 0 维 (`num_columns`)** 变成 **batch 维度**
- **第 1 维 (`num_rows`)** 变成 **序列长度**

效果就是：**对于每一列（独立地），让该列的各行之间做 self-attention**。

```
Col_0: [h00] ←→ [h10] ←→ [h20] ←→ [h30]
Col_1: [h01] ←→ [h11] ←→ [h21] ←→ [h31]
...
```

**这里使用了 attention_mask！**

```python
attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
# shape: (1, 1, num_rows, num_rows)
# 第0维broadcast到num_columns（batch），第1维broadcast到num_heads
```

这个 mask 的作用是**控制哪些行可以互相看到**。典型用途：
- 假设 Row_0 是 **query 行**（你想预测/补全的行），Row_1~Row_3 是 **context 行**（提供参考的行）
- 你希望 **context 行不能看到 query 行**（避免信息泄露），但 **query 行可以看到所有 context 行**
- attention_mask 中对应位置设为 `-inf` 就能实现这点

最后 `.transpose(0, 1)` 转回原始 shape `(num_rows, num_columns, hidden_size)`。

---

## 4. 底层组件详解

### TorchSelfAttention

这是标准的 Multi-Head Self-Attention：

```python
def forward(self, hidden_states, attention_mask):
    # hidden_states: (batch, seq_len, hidden_size)
    
    # 1. 线性投影 Q, K, V
    Q = self.query(hidden_states)  # (batch, seq_len, hidden_size)
    K = self.key(hidden_states)
    V = self.value(hidden_states)
    
    # 2. 拆成多头: (batch, num_heads, seq_len, head_dim)
    Q = transpose_for_scores(Q)
    K = transpose_for_scores(K)
    V = transpose_for_scores(V)
    
    # 3. Scaled Dot-Product Attention
    # Attention(Q,K,V) = softmax(QK^T / sqrt(d_k) + mask) V
    output = scaled_dot_product_attention(Q, K, V, attn_mask=attention_mask)
    
    # 4. 合并多头: (batch, seq_len, hidden_size)
    return reshape(output)
```

`transpose_for_scores` 做的事情：

```
(batch, seq_len, hidden_size) 
→ (batch, seq_len, num_heads, head_dim) 
→ (batch, num_heads, seq_len, head_dim)
```

### TorchRobertaLayer

标准的 Transformer Encoder Layer：

```
Input → Self-Attention → Add & LayerNorm → FFN → Add & LayerNorm → Output
```

---

## 5. 总结：数据流图

```
Input: (num_rows, num_columns, hidden_size)
                    │
                    ▼
    ┌──────────────────────────────┐
    │  Step 1: Cross-Column Attn   │  batch=num_rows, seq_len=num_columns
    │  (行内各列互相看, full attn) │  NO mask
    └──────────────────────────────┘
                    │
                    ▼
            transpose(0, 1)
      (num_columns, num_rows, hidden_size)
                    │
                    ▼
    ┌──────────────────────────────┐
    │  Step 2: Cross-Row Attn      │  batch=num_columns, seq_len=num_rows
    │  (列内各行互相看, masked)    │  WITH attention_mask
    └──────────────────────────────┘
                    │
                    ▼
            transpose(0, 1)
      (num_rows, num_columns, hidden_size)
                    │
                    ▼
              Output
```

---

## 6. 应用场景

这种二维注意力机制常见于：

1. **表格理解/补全任务**（如 SAP 的表格模型）：理解表格中单元格之间的关系
2. **类似 TAPAS、TaBERT 的表格预训练模型**：先理解同一行各列的语义关系（比如"姓名"和"年龄"属于同一个实体），再理解同一列各行的语义关系（比如"年龄"列里的不同值的分布和比较）
3. **图像 Patch 的二维注意力**（如 ViT 的 Axial Attention 变体）：分别沿水平和垂直方向做 attention

**优势**：
- 计算复杂度从 O((R×C)²) 降到 O(R×C² + C×R²)
- 保留了表格的二维结构归纳偏置
- attention_mask 允许灵活控制信息流向（如防止 context 看到 query）

---

## 7. 完整源代码

```python
from typing import Optional, Tuple

import torch
from torch import nn
from transformers.models.roberta.modeling_roberta import (
    RobertaIntermediate, RobertaOutput, RobertaSelfOutput
)


class TwoDimensionalAttentionLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        layer_class = TorchRobertaLayer
        self.cross_column_layer = layer_class(config)
        self.cross_row_layer = layer_class(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        num_rows, num_columns, _ = hidden_states.shape

        # Step 1: Cross-column attention (行内各列互相看)
        horizontal_outputs = torch.zeros_like(hidden_states)
        max_rows_per_batch = 8192
        col_fraction = 100.0 / float(num_columns)
        batch_step = int(max_rows_per_batch * col_fraction)

        for i in range(0, num_rows, batch_step):
            end_idx = i + batch_step
            chunk = hidden_states[i:end_idx, :, :]
            chunk_output = self.cross_column_layer(chunk)[0]
            horizontal_outputs[i:i + batch_step] = chunk_output

        # Step 2: Cross-row attention (列内各行互相看, with mask)
        batch_step = 100
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        horizontal_outputs = horizontal_outputs.transpose(0, 1).contiguous()

        vertical_outputs = torch.zeros_like(horizontal_outputs)
        for i in range(0, num_columns, batch_step):
            end_idx = i + batch_step
            chunk = horizontal_outputs[i:end_idx, :, :]
            chunk_output = self.cross_row_layer(chunk, attention_mask)[0]
            vertical_outputs[i:i + batch_step, :, :] = chunk_output

        return vertical_outputs.transpose(0, 1).contiguous()


class TorchRobertaLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.seq_len_dim = 1
        self.attention = TorchAttention(config)
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        return (self.output(intermediate_output, attention_output),)


class TorchAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self_attention = TorchSelfAttention(config)
        self.output = RobertaSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        self_outputs = self.self_attention(hidden_states, attention_mask)
        return self.output(self_outputs, hidden_states)


class TorchSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads

        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = config.attention_probs_dropout_prob

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )

        context_layer = attn_output.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.config.hidden_size,)
        return context_layer.view(new_context_layer_shape)
```
