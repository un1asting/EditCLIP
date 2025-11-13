"""
生成对抗性测试数据集
目的：测试EditCLIP和LLM在错误指令下的鲁棒性
"""

import json
import random

def generate_adversarial_instruction(original_instruction, sample_id):
    """
    为每个正确指令生成对应的错误指令
    错误类型：
    1. 颜色错误 - 改变颜色词
    2. 对象错误 - 改变对象名词
    3. 动作反转 - 添加↔移除
    4. 完全无关 - 随机不相关的指令
    """

    instruction = original_instruction.lower()

    # 类型1: 颜色错误
    color_map = {
        'blue': ['red', 'green', 'yellow'],
        'red': ['blue', 'green', 'purple'],
        'black': ['white', 'red', 'blue'],
        'white': ['black', 'blue', 'red'],
        'yellow': ['blue', 'red', 'green'],
        'ginger': ['blonde', 'brunette', 'black']
    }

    for correct_color, wrong_colors in color_map.items():
        if correct_color in instruction:
            wrong_color = random.choice(wrong_colors)
            adversarial = instruction.replace(correct_color, wrong_color)
            return adversarial, "color_mismatch"

    # 类型2: 对象错误
    object_map = {
        'dog': ['cat', 'bird', 'rabbit'],
        'table': ['chair', 'shelf', 'desk'],
        'car': ['truck', 'bus', 'motorcycle'],
        'airplane': ['helicopter', 'drone', 'jet'],
        'sheep': ['cow', 'goat', 'pig'],
        'umbrella': ['tent', 'canopy', 'awning'],
        'window': ['door', 'wall', 'ceiling'],
        'bags': ['boxes', 'baskets', 'suitcases'],
        'pillows': ['cushions', 'blankets', 'towels'],
        'backpack': ['suitcase', 'handbag', 'briefcase'],
        'laptop': ['tablet', 'desktop', 'monitor'],
        'cellphone': ['tablet', 'camera', 'watch'],
        'microwave': ['oven', 'dishwasher', 'blender'],
        'refrigerator': ['freezer', 'cabinet', 'pantry'],
        'cabinet': ['drawer', 'closet', 'wardrobe'],
        'train': ['bus', 'subway', 'tram']
    }

    for correct_obj, wrong_objs in object_map.items():
        if correct_obj in instruction:
            wrong_obj = random.choice(wrong_objs)
            adversarial = instruction.replace(correct_obj, wrong_obj)
            return adversarial, "object_mismatch"

    # 类型3: 动作反转
    if 'add' in instruction or 'put' in instruction or 'give' in instruction:
        if 'add' in instruction:
            adversarial = instruction.replace('add', 'remove')
        elif 'put' in instruction:
            adversarial = instruction.replace('put', 'take away')
        else:
            adversarial = instruction.replace('give', 'remove')
        return adversarial, "action_reversal"

    if 'remove' in instruction or 'get rid' in instruction:
        if 'remove' in instruction:
            adversarial = instruction.replace('remove', 'add')
        else:
            adversarial = instruction.replace('get rid of', 'add more')
        return adversarial, "action_reversal"

    # 类型4: 形状/属性错误
    shape_map = {
        'square': ['round', 'triangular', 'hexagonal'],
        'round': ['square', 'rectangular', 'oval']
    }

    for correct_shape, wrong_shapes in shape_map.items():
        if correct_shape in instruction:
            wrong_shape = random.choice(wrong_shapes)
            adversarial = instruction.replace(correct_shape, wrong_shape)
            return adversarial, "attribute_mismatch"

    # 类型5: 材质/类型错误
    if 'toy' in instruction:
        adversarial = instruction.replace('toy', 'real')
        return adversarial, "type_mismatch"

    if 'sports car' in instruction:
        adversarial = instruction.replace('sports car', 'sedan')
        return adversarial, "type_mismatch"

    # 类型6: 位置/数量错误
    if 'half' in instruction:
        adversarial = instruction.replace('half', 'whole')
        return adversarial, "quantity_mismatch"

    if 'all' in instruction:
        adversarial = instruction.replace('all', 'some')
        return adversarial, "quantity_mismatch"

    if 'lower' in instruction:
        adversarial = instruction.replace('lower', 'upper')
        return adversarial, "position_mismatch"

    # 类型7: 完全无关的指令（fallback）
    unrelated_instructions = [
        "change the sky to purple",
        "add a rainbow in the background",
        "make everything grayscale",
        "turn day into night",
        "add falling snow",
        "make it look vintage",
        "add a mirror reflection",
        "change the lighting to sunset"
    ]

    adversarial = random.choice(unrelated_instructions)
    return adversarial, "completely_unrelated"

def main():
    # 读取原始数据
    with open('data.json', 'r') as f:
        original_data = json.load(f)

    # 生成对抗性数据集
    adversarial_dataset = []

    print("正在生成对抗性测试数据集...")
    print("="*60)

    for item in original_data:
        sample_id = item['id']
        original_instruction = item['instruction']

        # 添加正确样本
        correct_sample = {
            "id": f"correct_{sample_id:03d}",
            "original_id": sample_id,
            "source_image": item['source_image'],
            "target_image": item['target_image'],
            "instruction": original_instruction,
            "is_correct": True,
            "error_type": None
        }
        adversarial_dataset.append(correct_sample)

        # 生成并添加错误样本
        adversarial_instruction, error_type = generate_adversarial_instruction(
            original_instruction, sample_id
        )

        wrong_sample = {
            "id": f"wrong_{sample_id:03d}",
            "original_id": sample_id,
            "source_image": item['source_image'],
            "target_image": item['target_image'],
            "instruction": adversarial_instruction,
            "is_correct": False,
            "error_type": error_type,
            "original_instruction": original_instruction
        }
        adversarial_dataset.append(wrong_sample)

        # 打印示例
        if sample_id < 5:
            print(f"\n样本 {sample_id}:")
            print(f"  ✅ 正确: {original_instruction}")
            print(f"  ❌ 错误: {adversarial_instruction} [{error_type}]")

    # 保存数据集
    output = {
        "dataset_info": {
            "total_samples": len(adversarial_dataset),
            "correct_samples": len([s for s in adversarial_dataset if s['is_correct']]),
            "wrong_samples": len([s for s in adversarial_dataset if not s['is_correct']]),
            "error_types": list(set([s['error_type'] for s in adversarial_dataset if s['error_type']])),
            "description": "对抗性测试数据集：包含正确和错误的指令配对，用于测试评估方法的鲁棒性"
        },
        "samples": adversarial_dataset
    }

    with open('adversarial_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n" + "="*60)
    print(f"✅ 数据集生成完成！")
    print(f"总样本数: {len(adversarial_dataset)}")
    print(f"  - 正确样本: {output['dataset_info']['correct_samples']}")
    print(f"  - 错误样本: {output['dataset_info']['wrong_samples']}")
    print(f"错误类型: {', '.join(output['dataset_info']['error_types'])}")
    print(f"输出文件: adversarial_dataset.json")

    # 统计错误类型分布
    error_type_count = {}
    for sample in adversarial_dataset:
        if sample['error_type']:
            error_type_count[sample['error_type']] = error_type_count.get(sample['error_type'], 0) + 1

    print("\n错误类型分布:")
    for error_type, count in sorted(error_type_count.items(), key=lambda x: -x[1]):
        print(f"  - {error_type}: {count} 样本")

if __name__ == "__main__":
    main()
