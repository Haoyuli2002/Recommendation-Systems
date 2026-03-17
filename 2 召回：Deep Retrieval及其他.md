# 2. 召回：Deep Retrieval及其他

Created: March 14, 2026 9:29 PM
Class: 推荐系统

## 整体逻辑

召回：用户 → 路径 → 物品

- 给定用户特征 $x$，用神经网络预测用户对路径 path = [a,b,c]的兴趣，分数记作 $p(\text{path}|x)$
- 用beam search来寻找分数最高的s条路径
- 利用路径到物品的索引，找到那个路径上的物品

训练逻辑：

- 一个物品有 J 条路径， $\text{path}_1$, …, $\text{path}_j$
- 如果用户点击过物品，则更新神经网络参数，使分数变大

$$
\max\sum_{j=1}^Jp(\text{path}_j|x)
$$

- 如果用户对路径的兴趣分数高 $p（\text{path}|x）$较高，且用户点击过物品item，那么item和path具有相关性
- 寻找与iten最相关的J个Path，并且避免一条路径上物品过多。

## 索引

把物品表示成路径。

- 需要确定有多少层，宽度是多少。

![Screenshot 2026-03-14 at 21.30.51.png](2%20%E5%8F%AC%E5%9B%9E%EF%BC%9ADeep%20Retrieval%E5%8F%8A%E5%85%B6%E4%BB%96/Screenshot_2026-03-14_at_21.30.51.png)

1. 物品到路径
    - 一个物品可以有多个路径
2. 路径到物品
    - 一个路径可以对应多个物品
    - 用三个节点表示一条路径：path = [a,b,c]

## 预估模型

- 给定用户特征x，预估用户对节点a的兴趣 $p_1(a|x)$
- 给定x和a，预估用户对节点b的兴趣 $p_2(b|a;x)$
- 以此类推，节点 $p_3(c|a,b;x)$
- 预估用户对 path = [a,b,c]的兴趣：

$$
p(a,b,c|x)=p_1(a|x)\times p_2(b|a;x)\times p_3(c|a,b;x)
$$

- 最优路径 $[a^*, b^*,c^*] = \argmax_{a,b,c} p(a,b,c|x)$
    - 贪心策略，但未必最好

![Screenshot 2026-03-14 at 21.40.37.png](2%20%E5%8F%AC%E5%9B%9E%EF%BC%9ADeep%20Retrieval%E5%8F%8A%E5%85%B6%E4%BB%96/Screenshot_2026-03-14_at_21.40.37.png)

## 线上召回

1. 给定用户特征，用beam search召回一批路径
    - 假如有3层，每层K个节点，那么一共有 $K^3$ 条路径
    - 用beam search来寻找路径

![Screenshot 2026-03-14 at 21.50.03.png](2%20%E5%8F%AC%E5%9B%9E%EF%BC%9ADeep%20Retrieval%E5%8F%8A%E5%85%B6%E4%BB%96/Screenshot_2026-03-14_at_21.50.03.png)

![Screenshot 2026-03-14 at 21.50.51.png](2%20%E5%8F%AC%E5%9B%9E%EF%BC%9ADeep%20Retrieval%E5%8F%8A%E5%85%B6%E4%BB%96/Screenshot_2026-03-14_at_21.50.51.png)

## 训练

### 1. 学习神经网络参数

- 预测用户对哪些路径感兴趣

![Screenshot 2026-03-14 at 22.07.12.png](2%20%E5%8F%AC%E5%9B%9E%EF%BC%9ADeep%20Retrieval%E5%8F%8A%E5%85%B6%E4%BB%96/Screenshot_2026-03-14_at_22.07.12.png)

### 2. 学习物品表征

- 物品应该是哪些路径
- 确定物品与路径的相关性

![Screenshot 2026-03-14 at 22.12.15.png](2%20%E5%8F%AC%E5%9B%9E%EF%BC%9ADeep%20Retrieval%E5%8F%8A%E5%85%B6%E4%BB%96/Screenshot_2026-03-14_at_22.12.15.png)

![Screenshot 2026-03-14 at 22.14.43.png](2%20%E5%8F%AC%E5%9B%9E%EF%BC%9ADeep%20Retrieval%E5%8F%8A%E5%85%B6%E4%BB%96/Screenshot_2026-03-14_at_22.14.43.png)

![Screenshot 2026-03-14 at 22.18.35.png](2%20%E5%8F%AC%E5%9B%9E%EF%BC%9ADeep%20Retrieval%E5%8F%8A%E5%85%B6%E4%BB%96/Screenshot_2026-03-14_at_22.18.35.png)

# 其他召回通道

### 地理位置召回：GeoHash

- 用户可能对附近发生的事感兴趣
- GeoHash：对经纬度编码，地图上的一个长方形区域
- 索引：GeoHash → 优质笔记列表（按时间倒排）
- 这条召回通道没有个性化

### 同城召回

- 城市 → 优质笔记

### 作者召回

索引

- 用户 → 关注的作者
- 作者 → 发布的笔记

召回：

- 用户 → 作者 → 最新笔记

### 有交互的作者召回

索引

- 用户 → 有交互的作者

召回

- 用户 → 有交互的作者 → 最新的笔记

### 相似作者召回

索引：

- 作者 → 相似作者

召回

- 用户 → 感兴趣的作者 → 相似作者 → 最新笔记

### 缓存召回

精排输出几百篇笔记，送入重排。

重排做多样性抽样，选出几十篇。

所以有很多精排的结果没有被曝光，缓存起来，作为一个召回通道。

### 曝光过滤

- 如果用户看过某个物品，则不再把这个物品曝光给这个用户。
- Bloom Filter
    - 判断一个物品ID是否在已经曝光的物品集合中
    - 如果判断为no，那么物品ID一定不在集合中
    - 但是判断为yes，物品很可能在集合中。（不一定肯定在）
- k个哈希函数，哈希table为 m bits。
- 假设有n个物品，误伤概率：

$$
(1-\text{exp}(-\frac{kn}{m}))^k
$$

- k有closed-form solution.