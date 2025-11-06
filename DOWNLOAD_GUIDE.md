# 如何获取真实的评估数据集

由于环境限制无法直接下载HuggingFace数据集，请按以下方式获取真实图片：

## 方案1：在本地机器上下载（推荐）

在您的**本地电脑**上运行以下命令：

```bash
# 安装依赖
pip install datasets huggingface_hub pillow

# 下载脚本（使用项目中的 download_real_samples.py）
python3 download_real_samples.py
```

或者使用Python代码：

```python
from datasets import load_dataset
import json
from PIL import Image
import random

# 加载数据集
dataset = load_dataset('osunlp/MagicBrush', split='train')
# 或者
# dataset = load_dataset('timbrooks/instructpix2pix-clip-filtered', split='train')

# 选择30个样本并保存...
```

然后将生成的 `evaluation_samples/` 文件夹上传到服务器。

## 方案2：使用已有的数据集

如果您已经下载了MagicBrush或InstructPix2Pix数据集，使用以下脚本：

```bash
python3 load_from_local_dataset.py --dataset_path /path/to/your/dataset
```

## 方案3：手动替换图片

保持现有的 `evaluation_samples/samples.json` 结构，手动替换 `evaluation_samples/images/` 中的图片：

1. 根据 `samples.json` 中的指令（instruction）
2. 找到相应的真实图片对
3. 重命名并替换对应的 `XXX_source.jpg` 和 `XXX_edited.jpg`

## 数据集链接

- **MagicBrush**: https://huggingface.co/datasets/osunlp/MagicBrush
- **InstructPix2Pix**: https://huggingface.co/datasets/timbrooks/instructpix2pix-clip-filtered
- **IP2P-1000**: https://huggingface.co/datasets/fusing/instructpix2pix-1000-samples

## 验证图片

替换真实图片后，运行：

```bash
python3 verify_samples.py
```

来验证所有图片是否正确加载。
