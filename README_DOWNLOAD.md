# 如何下载真实的MagicBrush图片

## ❌ 问题
当前环境无法直接访问HuggingFace数据集（网络限制，返回403错误）。

## ✅ 解决方案

### 方法1：在本地机器上运行（推荐）

在您的**本地电脑**上执行以下步骤：

```bash
# 1. 安装依赖
pip install datasets pillow

# 2. 下载这个脚本
# (将 download_magicbrush_50.py 复制到本地)

# 3. 运行下载脚本
python3 download_magicbrush_50.py
```

脚本会：
- 从MagicBrush下载50个样本
- 自动分层采样（每种编辑类型约8-9个）
- 保存到 `evaluation_samples/` 文件夹
- 生成 `samples.json` 元数据文件

完成后：
```bash
# 4. 打包
tar -czf evaluation_samples.tar.gz evaluation_samples/

# 5. 上传到服务器
scp evaluation_samples.tar.gz user@server:/path/to/EditCLIP/

# 6. 在服务器上解压
tar -xzf evaluation_samples.tar.gz

# 7. 验证
python3 verify_samples.py
```

### 方法2：使用Colab或Kaggle

如果本地机器配置困难，可以使用免费的云环境：

**Google Colab:**
```python
!pip install datasets pillow
!git clone https://github.com/un1asting/EditCLIP.git
%cd EditCLIP
!python3 download_magicbrush_50.py

# 下载到本地
from google.colab import files
!tar -czf evaluation_samples.tar.gz evaluation_samples/
files.download('evaluation_samples.tar.gz')
```

**Kaggle Notebook:**
类似的步骤，Kaggle也提供免费GPU环境和网络访问。

### 方法3：使用已下载的数据集

如果您已经下载了完整的MagicBrush数据集：

```bash
python3 load_from_local_dataset.py --dataset_path /path/to/MagicBrush
```

## 🔍 验证下载

下载完成后，运行验证脚本：

```bash
python3 verify_samples.py
```

应该看到：
```
Found 50 samples in samples.json
Verifying images...
✓ [001] Source: 512x512 - RGB
✓ [001] Edited: 512x512 - RGB
...
✅ All samples verified successfully!
```

## 📊 预期结果

- **总样本数**: 50
- **每种类型**: 约8-9个样本
- **编辑类型**:
  - object_add (添加物体)
  - object_remove (移除物体)
  - color_change (颜色变化)
  - style_transfer (风格转换)
  - small_edit (小编辑)
  - large_edit (大型编辑)

## 💡 提示

- 首次下载可能需要5-10分钟（取决于网速）
- 数据集会缓存到 `~/.cache/huggingface/`
- 需要约2-3GB的磁盘空间
- 如果遇到网络问题，可以使用VPN或代理

## 🆘 常见问题

**Q: 下载很慢怎么办？**
A: 可以设置HuggingFace镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

**Q: 提示403错误？**
A: 某些网络环境限制HuggingFace访问，尝试：
- 使用VPN
- 在Colab/Kaggle运行
- 使用镜像站点

**Q: 想要不同数量的样本？**
A: 编辑 `download_magicbrush_50.py` 中的 `total_samples=50` 参数
