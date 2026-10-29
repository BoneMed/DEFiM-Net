# Yield-Region Loss Design

## 1. Target 编码：`signed_log(strain × 100)`

模型预测的不是物理 strain，而是经过编码的值：

$$
y = \text{sign}(s) \cdot \log(1 + 100 \cdot |s|)
$$

逆变换：

$$
s = \text{sign}(y) \cdot \frac{e^{|y|} - 1}{100}
$$

| 物理 strain `s` | 编码值 `y` |
|---|---|
| 0 | 0 |
| -0.001 | -0.095 |
| -0.005 | -0.405 |
| **-0.019** | **-1.065** |
| -0.030 | -1.386 |
| -0.050 | -1.792 |
| -0.100 | -2.398 |

这个编码把极值压缩进一个对训练友好的范围（典型 [-3, 0]），同时在零附近近似线性、不像 `sqrt` 那样会"抹平"小变化。

## 2. 阈值自动转换

代码里你只填**物理 strain** 单位的阈值（`--yield_threshold -0.019`），Loss 内部把它转到编码空间：

```python
# toolkit/Loss.py
def yield_threshold_to_encoded(strain_threshold, scale=100.0):
    s = float(strain_threshold)
    return sign(s) * log(1 + |s| * scale)
```

`-0.019` 转出来 ≈ **`-1.065`**（encoded space），直接拿这个常数和 pred/target 比较，**不需要在 loss 里反编码**。

带宽 `--yield_temp 0.005`（物理 strain）也按局部导数转：

$$
\text{temp\_enc} = \text{temp\_strain} \cdot \frac{100}{1 + 100 |\text{threshold}|}
$$

`temp=0.005` 在 threshold=-0.019 处 → encoded 带宽 ≈ **`0.172`**。

**验证一下**：用 inv_signed_log(-1.065)：

$$
\text{sign}(-1.065) \cdot (e^{1.065}-1)/100 = -1 \cdot 1.901/100 \approx -0.019 \quad ✓
$$

## 3. Soft Dice 算的什么

对每个 bone 体素：

$$
\text{soft\_gt}(v) = \sigma\!\left(\frac{-1.065 - \text{target}(v)}{0.172}\right)
$$

$$
\text{soft\_pred}(v) = \sigma\!\left(\frac{-1.065 - \text{pred}(v)}{0.172}\right)
$$

- target / pred 在编码空间 ≤ -1.065（深失效）→ sigmoid ≈ 1
- ≥ -1.065（远未失效）→ sigmoid ≈ 0
- ≈ -1.065（边界）→ 0.5，梯度最强

然后 batch 内求 Dice：

$$
L_\text{yield} = 1 - \frac{2 \sum_v \text{soft\_gt}(v) \cdot \text{soft\_pred}(v) \cdot m(v)}
                              {\sum_v \text{soft\_gt}(v) \cdot m(v) + \sum_v \text{soft\_pred}(v) \cdot m(v)}
$$

`m(v)` 是 bone mask。所有运算在编码空间完成，**不解码**。

## 4. 当前总 Loss 结构

```
total = main_L1                                    # 全 bone 体素加权 L1
      + λ_grad · ∂L1_extreme_tail                  # top-2% 压应力区的梯度匹配
      + λ_yield · (1 - SoftDice_yield)             # 失效区软 Dice
```

- `main_L1` 量级 ~0.1-1.0（编码空间）
- `∂L1_extreme_tail` 量级 ~0.5-3.0
- `1 - SoftDice` 量级 0-1

三项加权求和后一起 `backward()`。

## 5. 你的担心："融合在一起学不到"

合理的担心。两种潜在问题：

**问题 1：Dice 信号被 main_L1 稀释**

main_L1 在所有 bone 体素上算（数万体素），每次反传梯度密度高。Soft Dice 的梯度只在 sigmoid 过渡带（threshold ± temp）附近显著，覆盖的体素少得多 —— 几百到几千。绝对梯度幅度可能差几个数量级。

**问题 2：Dice 和 L1 的优化方向偶尔冲突**

L1 让 pred 在每个体素都贴近 target；Dice 让 pred 的"是否过阈值"贴近 target 的。极端情况：某体素 target = -0.020（刚过阈值），pred = -0.015（差一点）。L1 想让 pred 调到 -0.020（小幅调整）。Dice 想让 pred 越过 -0.019（更大幅调整）。两者梯度方向同号但幅度不同。一般不冲突，但偶尔会。

## 6. "用两个 loss" 的可选方案

A 你已经有了一个（SoftDice）。想再加一个**更直接的"失效区加权 L1"**：

### 方案 X：失效区加权 L1（直接补充）

在 main_L1 里加一个 hard-mask 加权：

```python
yield_mask = (target_enc <= yield_thr_enc).float()   # 1 if target 失效, else 0
weight = 1 + yield_focus * yield_mask
loss = (|pred - target| * weight * bone_mask).mean()
```

参数 `yield_focus` 比如设 4.0 → 失效区体素的 L1 权重是普通区的 **5×**。

**对比 SoftDice**：
- SoftDice：通过 sigmoid 软化，整体集合重叠
- 失效区加权 L1：硬掩码，直接放大失效区 per-voxel 误差

**两者互补**：
- SoftDice 不会陷入"per-voxel 完美但集合形状偏移"的局部最优
- 加权 L1 直接放大失效区数值精度

### 方案 Y：失效区单独 backward

```python
loss_main = L1 + λ_grad · ∂L1
loss_yield = SoftDice + 失效区加权L1

opt.zero_grad()
(loss_main + loss_yield).backward()   # 还是一次 backward
opt.step()
```

这其实跟 fused 是一样的（PyTorch 的 backward 是累加梯度，一次还是分两次效果等价，只是数值上略有差异因 chain rule 缓存）。**单纯"分两次 backward"不会让信号更强**。

要让 yield 信号"独立"地训练，得：

- 用**单独的 optimizer**（比如更高 lr），或者
- **轮流训练**（这步 main，下步 yield），或者
- **梯度手术**（PCGrad，给冲突的梯度做投影）

但这些都把代码变复杂，且未必比"调大 λ_yield"或"加方案 X"效果好。

## 7. 推荐 & 怎么调

**第一步**：先用当前的 fused 版本（SoftDice 已加，λ_yield=0.5 默认）跑一轮 tumor finetune，看 epoch log 里的 `Yield-Dice case` 和 `Yield-Dice global` 有没有显著上升。

**如果 Dice 收敛慢或几乎不动**（被 L1 压制）：

1. **先尝试调大 `--lambda_yield`**：试 1.0、2.0
2. **如果还是不够强**，加方案 X（失效区加权 L1）。我可以加一个 `--yield_focus` 参数，0 关闭，5-10 强烈

**如果训练数值变得不稳**（loss spike、Dice 振荡）：

- `--yield_temp` 调大到 0.01（更软的过渡，梯度更平缓）
- 或者降 `--lambda_yield` 到 0.3

## 8. 监控的指标

每个 epoch 现在会打印两个 Dice：

| 指标 | 含义 | 怎么读 |
|---|---|---|
| `Yield-Dice case` | 每个 case 算 hard Dice，再平均 | 反映**典型 case** 的边界对齐质量 |
| `Yield-Dice global` | 所有 case 体素汇集成一个集合算 hard Dice | 反映**整体体素级别**的失效区重叠 |

两者一般 case 平均 ≤ global，因为 case 平均会被失效体素少的 case 拖累（小集合 Dice 容易抖）。

关注 **case 平均上涨**意味着模型在各种 case 上都学会了找失效边界，不是只在大 case 上靠数量碾压。

---

## 总结你看对不对

1. ✅ 阈值 `-0.019` 是物理 strain，代码自动转 `-1.065`（编码空间）
2. ✅ 温度 `0.005` 物理 strain 转 `0.172`（编码空间）
3. ✅ Loss 全程在编码空间算，target/pred 不解码
4. ✅ 当前是 fused 三项 loss
5. ⏳ 是否需要加方案 X（失效区加权 L1）—— 等第一轮跑完看 Dice 收敛再决定

要我现在就加方案 X 吗？还是先跑一轮看 SoftDice 单独效果？
