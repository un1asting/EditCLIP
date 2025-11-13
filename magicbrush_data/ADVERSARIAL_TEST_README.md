# 对抗性测试数据集说明

## 📊 数据集概述

**文件**: `adversarial_dataset.json`

**目的**: 测试EditCLIP和LLM评估方法在**错误指令**下的鲁棒性

**规模**:
- 总样本数: **60个**
  - ✅ 正确样本: 30个（原始正确的编辑+指令）
  - ❌ 错误样本: 30个（同样的图像+**错误的指令**）

**图像**: 复用 `images/` 目录中的原始图像，无需额外存储

---

## 🎯 测试目标

### 问题
现有评估方法（EditCLIP和LLM）是否能正确识别：
- **正确编辑** + **正确指令** → 应该给**高分**
- **正确编辑** + **错误指令** → 应该给**低分**

### 核心假设

**EditCLIP的潜在弱点**:
- 可能只关注"图像是否变化"
- 不深入理解"变化是否匹配指令"
- **预测**: 即使指令错误，只要图像有变化就可能给中等分

**LLM的潜在弱点**:
- 可能被图像的实际变化误导
- 看到"编辑质量好"就给高分，忽略指令不匹配
- **预测**: 应该能识别大部分错误，但可能在细微差异上出错

---

## 📂 数据集结构

### JSON格式

```json
{
  "dataset_info": {
    "total_samples": 60,
    "correct_samples": 30,
    "wrong_samples": 30,
    "error_types": ["color_mismatch", "object_mismatch", ...]
  },
  "samples": [
    {
      "id": "correct_000",
      "original_id": 0,
      "source_image": "images/sample_000_source.jpg",
      "target_image": "images/sample_000_target.jpg",
      "instruction": "change the table for a dog",
      "is_correct": true,
      "error_type": null
    },
    {
      "id": "wrong_000",
      "original_id": 0,
      "source_image": "images/sample_000_source.jpg",
      "target_image": "images/sample_000_target.jpg",
      "instruction": "change the table for a rabbit",  // 错误！
      "is_correct": false,
      "error_type": "object_mismatch",
      "original_instruction": "change the table for a dog"
    }
  ]
}
```

### 字段说明

- **id**: 样本唯一标识 (`correct_XXX` 或 `wrong_XXX`)
- **original_id**: 原始样本ID (0-29)
- **source_image**: 编辑前图像路径
- **target_image**: 编辑后图像路径
- **instruction**: 指令文本
- **is_correct**: 布尔值，true表示正确配对，false表示错误配对
- **error_type**: 错误类型（仅错误样本有）
- **original_instruction**: 原始正确指令（仅错误样本有）

---

## 🔬 错误类型 (6种)

### 1. **颜色不匹配** (color_mismatch) - 6个样本

图像实际变化与指令描述的颜色不符。

**案例**:
```
✅ 正确: "Make the umbrellas blue"
   → 图像: 伞从黄色变成蓝色 ✓

❌ 错误: "Make the umbrellas red"
   → 图像: 伞从黄色变成蓝色
   → 指令说红色，但实际是蓝色 ✗
```

**预期结果**:
- EditCLIP: 可能给中等分（图像确实变了，有部分颜色特征匹配）
- LLM: **应该低分**（能看出是蓝色不是红色）

---

### 2. **对象不匹配** (object_mismatch) - 12个样本

图像中的对象与指令描述的对象不一致。

**案例**:
```
✅ 正确: "change the table for a dog"
   → 图像: 桌子被替换成狗 ✓

❌ 错误: "change the table for a rabbit"
   → 图像: 桌子被替换成狗
   → 指令说兔子，但实际是狗 ✗
```

**预期结果**:
- EditCLIP: 可能给中等分（table→animal有语义重叠）
- LLM: **应该低分**（能区分狗和兔子）

---

### 3. **动作反转** (action_reversal) - 4个样本

指令的动作方向与实际编辑相反。

**案例**:
```
✅ 正确: "remove the mesh screen"
   → 图像: 网格被移除 ✓

❌ 错误: "add the mesh screen"
   → 图像: 网格被移除
   → 指令说添加，但实际是移除 ✗
```

**预期结果**:
- EditCLIP: **可能给高分**（无法理解"添加"vs"移除"的语义差异）
- LLM: **应该低分**（能理解动作方向错误）

---

### 4. **数量不匹配** (quantity_mismatch) - 2个样本

数量相关的描述与实际编辑不符。

**案例**:
```
✅ 正确: "leave words only on half the page"
   → 图像: 半页有文字 ✓

❌ 错误: "leave words only on whole the page"
   → 图像: 半页有文字
   → 指令说整页，但实际是半页 ✗
```

---

### 5. **属性不匹配** (attribute_mismatch) - 1个样本

形状、大小等属性描述错误。

**案例**:
```
✅ 正确: "the window is now square"
   → 图像: 窗户变成方形 ✓

❌ 错误: "the window is now round"
   → 图像: 窗户变成方形
   → 指令说圆形，但实际是方形 ✗
```

---

### 6. **完全无关** (completely_unrelated) - 5个样本

指令与实际编辑完全不相关。

**案例**:
```
✅ 正确: "leave only words on the page"
   → 图像: 只保留文字 ✓

❌ 错误: "add a rainbow in the background"
   → 图像: 只保留文字
   → 指令说彩虹，但图像里没有 ✗
```

**预期结果**:
- EditCLIP: **可能给低分**（指令文本与图像特征完全不匹配）
- LLM: **应该低分**（明显无关）

---

## 🧪 实验流程

### 步骤1: 运行EditCLIP评估

```bash
# 需要修改现有的EditCLIP评估脚本，读取adversarial_dataset.json
python evaluate_editclip_adversarial.py
```

**预期输出**:
- 正确样本的平均分: ?
- 错误样本的平均分: ?
- **理想情况**: 错误样本分数明显低于正确样本

### 步骤2: 运行LLM评估

```bash
# 使用LLM评估对抗样本
export ANTHROPIC_API_KEY='your-key'
python evaluate_llm_adversarial.py
```

**预期输出**:
- 正确样本的平均分: ?
- 错误样本的平均分: ?
- **理想情况**: LLM能正确识别大部分错误

### 步骤3: 对比分析

```python
# 计算准确率
correct_samples = [s for s in results if s['is_correct']]
wrong_samples = [s for s in results if not s['is_correct']]

# EditCLIP的区分能力
editclip_correct_avg = mean([s['editclip_score'] for s in correct_samples])
editclip_wrong_avg = mean([s['editclip_score'] for s in wrong_samples])
editclip_gap = editclip_correct_avg - editclip_wrong_avg

# LLM的区分能力
llm_correct_avg = mean([s['llm_score'] for s in correct_samples])
llm_wrong_avg = mean([s['llm_score'] for s in wrong_samples])
llm_gap = llm_correct_avg - llm_wrong_avg

# 比较
print(f"EditCLIP 区分度: {editclip_gap:.3f}")
print(f"LLM 区分度: {llm_gap:.3f}")
```

---

## 📈 预期结果

### 假设1: EditCLIP在某些错误类型上表现不佳

**可能失败的类型**:
- ✅ **动作反转**: EditCLIP可能无法区分"添加"vs"移除"
- ✅ **对象细分**: EditCLIP可能无法区分"狗"vs"兔子"（都是动物）
- ⚠️ **颜色错误**: 可能部分识别，但不如LLM精确

**可能成功的类型**:
- ✅ **完全无关**: 文本特征完全不匹配，EditCLIP应该给低分

### 假设2: LLM整体表现更好

**优势**:
- 能理解细粒度语义差异（狗vs猫，添加vs移除）
- 能通过视觉检查验证颜色、对象、数量

**可能的弱点**:
- 视觉幻觉：可能误判图像内容
- 对微小差异可能不够敏感

---

## 🎯 评估指标

### 1. 分类准确率

```
准确率 = (正确识别为正确 + 正确识别为错误) / 总样本数
```

**阈值设定**:
- 分数 > 0.5 (EditCLIP) 或 > 5.0 (LLM) → 判定为"正确"
- 分数 ≤ 0.5 (EditCLIP) 或 ≤ 5.0 (LLM) → 判定为"错误"

### 2. 区分度 (Discrimination)

```
区分度 = 正确样本平均分 - 错误样本平均分
```

**理想情况**: 区分度越大越好

### 3. 分错误类型的准确率

```
对每种错误类型计算:
- 该类型错误样本的平均分
- 是否显著低于正确样本？
```

---

## 💡 使用建议

1. **先运行正确样本**: 验证评估方法基本工作正常
2. **再运行错误样本**: 测试鲁棒性
3. **分析失败案例**: 找出哪种错误类型最容易被误判
4. **改进评估方法**: 针对弱点进行优化

---

## 📝 典型案例展示

### 案例1: 颜色不匹配

```
原始ID: 19
图像: sample_019_source.jpg → sample_019_target.jpg

✅ 正确配对:
   指令: "Make the umbrellas blue."
   图像: 黄色伞 → 蓝色伞
   期望: EditCLIP高分, LLM高分

❌ 错误配对:
   指令: "Make the umbrellas red."
   图像: 黄色伞 → 蓝色伞 (实际是蓝色!)
   期望: EditCLIP中等分?, LLM低分
```

### 案例2: 对象不匹配

```
原始ID: 0
图像: sample_000_source.jpg → sample_000_target.jpg

✅ 正确配对:
   指令: "change the table for a dog"
   图像: 桌子 → 狗
   期望: EditCLIP高分, LLM高分

❌ 错误配对:
   指令: "change the table for a rabbit"
   图像: 桌子 → 狗 (实际是狗不是兔子!)
   期望: EditCLIP中等分?, LLM低分
```

### 案例3: 动作反转

```
原始ID: 6
图像: sample_006_source.jpg → sample_006_target.jpg

✅ 正确配对:
   指令: "remove the mesh screen"
   图像: 有网格 → 无网格
   期望: EditCLIP高分, LLM高分

❌ 错误配对:
   指令: "add the mesh screen"
   图像: 有网格 → 无网格 (实际是移除!)
   期望: EditCLIP高分?, LLM低分
```

**这是EditCLIP最可能失败的案例！**

---

## 🔮 预期发现

通过这个对抗性测试，我们期望发现：

1. **EditCLIP的弱点**:
   - 在"动作反转"类错误上区分度低
   - 在"对象细分"上可能被相似语义误导

2. **LLM的优势**:
   - 在大部分错误类型上表现优于EditCLIP
   - 特别是在需要细粒度语义理解的任务上

3. **改进方向**:
   - EditCLIP: 需要更强的语义理解能力
   - LLM: 需要减少视觉幻觉，提高一致性

---

**生成时间**: 2025-11-07
**数据集文件**: `adversarial_dataset.json`
**生成脚本**: `generate_adversarial_dataset.py`
