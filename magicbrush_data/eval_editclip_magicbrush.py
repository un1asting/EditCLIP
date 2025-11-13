"""
评估 EditCLIP 在 MagicBrush 数据集上的表现
计算编辑前后图像对的相似度，并保存结果到 JSON 文件
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import sys

# 添加 EditCLIP 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'EditCLIP', 'src'))

try:
    from transformers import CLIPModel, AutoProcessor
    from open_clip import create_model_and_transforms, get_tokenizer
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已安装必要的库:")
    print("pip install transformers torch torchvision pillow tqdm")
    sys.exit(1)


def load_image(image_path, resolution=256):
    """加载并预处理图像"""
    image = Image.open(image_path).convert("RGB")
    image = image.resize((resolution, resolution))
    return image


def pil_to_tensor(pil_image):
    """将 PIL 图像转换为 tensor"""
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(pil_image).unsqueeze(0)


def compute_editclip_score(model, processor, tokenizer, source_image, target_image, instruction, device):
    """
    使用 EditCLIP 计算图像编辑质量分数

    Args:
        model: EditCLIP 模型
        processor: 图像处理器
        tokenizer: 文本分词器
        source_image: 编辑前图像 (PIL Image)
        target_image: 编辑后图像 (PIL Image)
        instruction: 编辑指令文本
        device: 设备

    Returns:
        编辑质量分数 (float) - 图像编辑与文本指令的相似度
    """
    with torch.no_grad():
        # 1. 处理图像对 (6 通道输入)
        inputs1 = processor(images=source_image, return_tensors="pt")
        inputs2 = processor(images=target_image, return_tensors="pt")

        # 获取预处理后的像素值并拼接
        pixel_values1 = inputs1['pixel_values']
        pixel_values2 = inputs2['pixel_values']

        # 拼接成 6 通道
        concatenated_pixel_values = torch.cat([pixel_values1, pixel_values2], dim=1).to(device)

        # 获取图像编辑特征
        image_features = model.get_image_features(pixel_values=concatenated_pixel_values)

        # 2. 处理文本指令
        text_inputs = tokenizer(instruction, return_tensors="pt", padding=True, truncation=True)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        # 获取文本特征
        text_features = model.get_text_features(**text_inputs)

        # 3. 归一化特征
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # 4. 计算图像编辑与文本指令的相似度
        similarity = (image_features @ text_features.T).item()

    return similarity


def compute_clip_similarity(model, processor, image1, image2, device, is_vision_model=False):
    """
    使用标准 CLIP 模型计算两张图像的相似度

    Args:
        model: CLIP 模型
        processor: 图像处理器
        image1: 第一张图像 (PIL Image)
        image2: 第二张图像 (PIL Image)
        device: 设备
        is_vision_model: 是否是 vision_model (CLIPVisionTransformer)

    Returns:
        相似度分数 (float)
    """
    with torch.no_grad():
        # 标准 CLIP：分别处理两张图像
        inputs1 = processor(images=image1, return_tensors="pt")
        inputs1 = {k: v.to(device) for k, v in inputs1.items()}
        inputs2 = processor(images=image2, return_tensors="pt")
        inputs2 = {k: v.to(device) for k, v in inputs2.items()}

        # 获取图像特征
        if is_vision_model:
            # 对于 vision_model，直接调用模型
            features1 = model(**inputs1).pooler_output
            features2 = model(**inputs2).pooler_output
        else:
            # 对于完整的 CLIP 模型
            features1 = model.get_image_features(**inputs1)
            features2 = model.get_image_features(**inputs2)

        # 归一化特征
        features1 = features1 / features1.norm(dim=-1, keepdim=True)
        features2 = features2 / features2.norm(dim=-1, keepdim=True)

        # 计算余弦相似度
        similarity = (features1 @ features2.T).item()

    return similarity


def compute_editclip_features(model, processor, source_image, target_image, device):
    """
    使用 EditCLIP 计算编辑前后图像对的特征

    Args:
        model: EditCLIP 模型
        processor: 图像处理器
        source_image: 编辑前图像 (PIL Image)
        target_image: 编辑后图像 (PIL Image)
        device: 设备

    Returns:
        编辑特征向量
    """
    # 将图像转换为 tensor
    source_tensor = pil_to_tensor(source_image)
    target_tensor = pil_to_tensor(target_image)

    # 拼接图像 (EditCLIP 使用 6 通道输入)
    concatenated = torch.cat([source_tensor, target_tensor], dim=1).to(device)

    # 处理拼接后的图像
    inputs = processor(images=concatenated, return_tensors="pt", do_rescale=False).to(device)

    # 获取特征
    with torch.no_grad():
        features = model.get_image_features(**inputs)

    # 归一化
    features = features / features.norm(dim=-1, keepdim=True)

    return features


def evaluate_magicbrush(
    data_json_path,
    data_dir,
    model_path=None,
    output_json="editclip_results.json",
    resolution=256,
    use_editclip=False
):
    """
    评估 EditCLIP 在 MagicBrush 数据集上的表现

    Args:
        data_json_path: MagicBrush 数据 JSON 文件路径
        data_dir: 数据目录
        model_path: EditCLIP 模型路径 (如果使用 EditCLIP)
        output_json: 输出 JSON 文件路径
        resolution: 图像分辨率
        use_editclip: 是否使用 EditCLIP (否则使用标准 CLIP)
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载模型
    print("正在加载模型...")
    tokenizer = None
    is_vision_model = False

    if use_editclip and model_path:
        try:
            # 加载 EditCLIP 模型 (完整模型，包括文本编码器)
            model = CLIPModel.from_pretrained(model_path)
            processor = AutoProcessor.from_pretrained(model_path)

            # processor 中已经包含了 tokenizer
            tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor

            is_vision_model = False
            print(f"已加载 EditCLIP 模型: {model_path}")
            print("EditCLIP 将计算图像编辑与文本指令的匹配度")
        except Exception as e:
            print(f"加载 EditCLIP 失败: {e}")
            print("回退到标准 CLIP 模型")
            use_editclip = False

    if not use_editclip:
        # 使用标准 CLIP 模型
        model_name = "openai/clip-vit-base-patch32"
        model = CLIPModel.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name)
        is_vision_model = False
        print(f"已加载标准 CLIP 模型: {model_name}")

    model = model.to(device)
    model.eval()

    # 加载数据
    print(f"正在加载数据: {data_json_path}")
    with open(data_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 处理不同的数据格式
    if isinstance(data, dict) and 'samples' in data:
        # 对抗性数据集格式
        data_list = data['samples']
        print(f"检测到对抗性数据集格式")
    elif isinstance(data, list):
        # 原始数据格式
        data_list = data
    else:
        raise ValueError("不支持的数据格式")

    print(f"共有 {len(data_list)} 个样本")

    # 评估结果
    results = []

    print("开始评估...")
    for item in tqdm(data_list):
        sample_id = item['id']
        instruction = item['instruction']

        # 构建图像路径
        source_image_path = os.path.join(data_dir, item['source_image'])
        target_image_path = os.path.join(data_dir, item['target_image'])

        # 检查文件是否存在
        if not os.path.exists(source_image_path) or not os.path.exists(target_image_path):
            print(f"警告: 样本 {sample_id} 的图像文件不存在，跳过")
            continue

        try:
            # 加载图像
            source_image = load_image(source_image_path, resolution)
            target_image = load_image(target_image_path, resolution)

            # 计算分数
            if use_editclip:
                # EditCLIP: 计算图像编辑与文本指令的匹配度
                score = compute_editclip_score(model, processor, tokenizer, source_image, target_image, instruction, device)
                score_name = "editclip_score"
            else:
                # 标准 CLIP: 计算编辑前后图像的相似度
                score = compute_clip_similarity(model, processor, source_image, target_image, device, is_vision_model)
                score_name = "image_similarity"

            # 保存结果
            result = {
                "id": sample_id,
                "instruction": instruction,
                "source_image": item['source_image'],
                "target_image": item['target_image'],
                score_name: float(score),
                "model": "EditCLIP" if use_editclip else "CLIP"
            }
            results.append(result)

        except Exception as e:
            print(f"处理样本 {sample_id} 时出错: {e}")
            continue

    # 计算统计信息
    if len(results) == 0:
        print("\n错误: 没有成功处理任何样本")
        return

    score_key = "editclip_score" if use_editclip else "image_similarity"
    scores = [r[score_key] for r in results]

    stats = {
        "num_samples": len(results),
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
        "model": "EditCLIP" if use_editclip else "CLIP",
        "resolution": resolution,
        "score_type": "图像编辑与文本指令的匹配度" if use_editclip else "编辑前后图像相似度"
    }

    # 保存结果
    output_data = {
        "statistics": stats,
        "results": results
    }

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n评估完成!")
    print(f"模型: {stats['model']}")
    print(f"评估指标: {stats['score_type']}")
    print(f"处理样本数: {stats['num_samples']}")
    print(f"平均分数: {stats['mean_score']:.4f}")
    print(f"标准差: {stats['std_score']:.4f}")
    print(f"最小分数: {stats['min_score']:.4f}")
    print(f"最大分数: {stats['max_score']:.4f}")
    print(f"结果已保存到: {output_json}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="评估 EditCLIP 在 MagicBrush 数据集上的表现")
    parser.add_argument("--data_json", type=str, default="magicbrush_data/data.json",
                        help="MagicBrush 数据 JSON 文件路径")
    parser.add_argument("--data_dir", type=str, default="magicbrush_data",
                        help="数据目录")
    parser.add_argument("--model_path", type=str, default=None,
                        help="EditCLIP 模型路径 (可选)")
    parser.add_argument("--output_json", type=str, default="editclip_results.json",
                        help="输出 JSON 文件路径")
    parser.add_argument("--resolution", type=int, default=256,
                        help="图像分辨率")
    parser.add_argument("--use_editclip", action="store_true",
                        help="使用 EditCLIP 模型 (需要提供 model_path)")

    args = parser.parse_args()

    evaluate_magicbrush(
        data_json_path=args.data_json,
        data_dir=args.data_dir,
        model_path=args.model_path,
        output_json=args.output_json,
        resolution=args.resolution,
        use_editclip=args.use_editclip
    )
