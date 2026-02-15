#!/bin/bash

# ========================================================================
# 1_full_download.sh (CPU 服务器 - 严厉版)
# 目标: 下载 Python3.10 + 依赖包 + 数据 + 权重
# 特性: 
# 1. 暴力直链下载 (Wget) 绕过 pip 版本检查
# 2. [新增] 严厉的完整性自检 (Strict Verification)
# ========================================================================

SAVE_DIR="./offline_packages"
PYTHON_DIR="$SAVE_DIR/python_runtime"
WHEEL_DIR="$SAVE_DIR/wheels"
DATA_DIR="$SAVE_DIR/datasets"
WEIGHTS_DIR="$SAVE_DIR/tool_weights"

# 清理旧的 unfinished 下载
mkdir -p $PYTHON_DIR $WHEEL_DIR $DATA_DIR $WEIGHTS_DIR

echo "🚀 [CPU Server] 开始构建全量离线资源 (严厉模式)..."

# =========================================================
# 1. 下载独立 Python (Wget)
# =========================================================
echo "🐍 [1/6] 下载 Python 3.10 Runtime..."
PYTHON_URL="https://github.com/indygreg/python-build-standalone/releases/download/20240224/cpython-3.10.13+20240224-x86_64-unknown-linux-gnu-install_only.tar.gz"
if [ ! -f "$PYTHON_DIR/python-3.10.tar.gz" ]; then
    wget -c -O "$PYTHON_DIR/python-3.10.tar.gz" "$PYTHON_URL" || curl -L -o "$PYTHON_DIR/python-3.10.tar.gz" "$PYTHON_URL"
fi

# =========================================================
# 2. 下载核心框架 (Wget 直链 - 绝对可靠)
# =========================================================
echo "🔥 [2/6] 暴力下载 PyTorch (CUDA 12.1)..."
BASE_URL="https://download.pytorch.org/whl/cu121"
# 显式下载，不给 pip 犯错的机会
wget -nc -P $WHEEL_DIR "$BASE_URL/torch-2.4.1%2Bcu121-cp310-cp310-linux_x86_64.whl"
wget -nc -P $WHEEL_DIR "$BASE_URL/torchvision-0.19.1%2Bcu121-cp310-cp310-linux_x86_64.whl"
wget -nc -P $WHEEL_DIR "$BASE_URL/torchaudio-2.4.1%2Bcu121-cp310-cp310-linux_x86_64.whl"

echo "🤗 [3/6] 下载 Transformers (GitHub Main)..."
wget -nc -O "$WHEEL_DIR/transformers-main.zip" "https://github.com/huggingface/transformers/archive/refs/heads/main.zip"

echo "⚡ [4/6] 下载 Flash Attention 2..."
# 使用 pip download 但加上 --no-deps
pip download flash-attn==2.6.3 --dest $WHEEL_DIR --index-url https://pypi.org/simple --trusted-host pypi.org --no-binary :all: --no-deps

# =========================================================
# 3. 下载通用依赖 (伪装 Py3.10 + No-Deps)
# =========================================================
echo "📚 [5/6] 下载所有依赖 (Nvidia, Numpy, Tools)..."

download_wheel() {
    pip download "$@" \
        --dest $WHEEL_DIR \
        --index-url https://pypi.org/simple \
        --trusted-host pypi.org \
        --trusted-host files.pythonhosted.org \
        --python-version 3.10 \
        --platform manylinux2014_x86_64 \
        --only-binary=:all: \
        --no-deps
}

# 3.1 必须手动列出 NVIDIA 全家桶 (PyTorch 2.x 运行必需)
echo "   -> 下载 NVIDIA CUDA 库..."
download_wheel nvidia-cuda-nvrtc-cu12==12.1.105
download_wheel nvidia-cuda-runtime-cu12==12.1.105
download_wheel nvidia-cuda-cupti-cu12==12.1.105
download_wheel nvidia-cudnn-cu12==9.1.0.70
download_wheel nvidia-cublas-cu12==12.1.3.1
download_wheel nvidia-cufft-cu12==11.0.2.54
download_wheel nvidia-curand-cu12==10.3.2.106
download_wheel nvidia-cusolver-cu12==11.4.5.107
download_wheel nvidia-cusparse-cu12==12.1.0.106
download_wheel nvidia-nccl-cu12==2.20.5
download_wheel nvidia-nvtx-cu12==12.1.105
download_wheel triton==3.0.0

# 3.2 必须手动列出构建工具 (编译 FlashAttn 必需)
echo "   -> 下载构建工具 (Numpy, Ninja)..."
download_wheel "numpy<2.0.0" # 显式版本防止不兼容
download_wheel packaging
download_wheel ninja
download_wheel psutil
download_wheel setuptools
download_wheel wheel

# 3.3 其他常规依赖
echo "   -> 下载常规依赖..."
download_wheel accelerate>=0.27.0 huggingface-hub>=0.23.0 tokenizers>=0.19.1 safetensors>=0.4.1
download_wheel regex requests filelock fsspec pyyaml tqdm
download_wheel sympy networkx jinja2 MarkupSafe typing-extensions mpmath
download_wheel charset-normalizer idna urllib3 certifi
download_wheel datasets sentence-transformers Pillow easyocr scipy
download_wheel termcolor timm rich questionary aiohttp protobuf sentencepiece
download_wheel opencv-python-headless scikit-image python-bidi PyYAML
download_wheel attrs multidict yarl frozenlist aiosignal async-timeout
download_wheel pandas pytz python-dateutil six

# =========================================================
# 4. 严厉的自检模块 (Strict Verification)
# =========================================================
echo "------------------------------------------------"
echo "🕵️  开始执行严厉的完整性检测..."

MISSING_COUNT=0

# 定义核心物资清单 (正则匹配文件名)
# 这些文件如果不存在，GPU 那边 100% 跑不起来
CRITICAL_FILES=(
    "python-3.10.tar.gz"          # Python 本体
    "torch-2.4.1"                 # Torch 本体
    "transformers-main.zip"       # TF 源码
    "flash_attn"                  # FlashAttn 源码
    "numpy"                       # 核心运算库
    "packaging"                   # 编译工具
    "ninja"                       # 编译工具
    "nvidia_cublas_cu12"          # CUDA 数学库 (最容易漏)
    "nvidia_cudnn_cu12"           # CUDA 神经网络库
    "nvidia_cuda_runtime_cu12"    # CUDA 运行时
    "triton"                      # OpenAI Triton
    "filelock"                    # Torch 依赖
    "sympy"                       # Torch 依赖
    "networkx"                    # Torch 依赖
    "jinja2"                      # Torch 依赖
    "pillow"                      # 图像处理
)

# 检查函数
check_file() {
    # 在 WHEEL_DIR 和 PYTHON_DIR 查找文件名包含关键字的文件
    # -iname 忽略大小写
    count=$(find $WHEEL_DIR $PYTHON_DIR -type f -iname "*$1*" | wc -l)
    if [ "$count" -eq 0 ]; then
        echo "❌ [严重缺失] 未找到包: $1"
        MISSING_COUNT=$((MISSING_COUNT + 1))
    else
        echo "✅ 已就位: $1"
    fi
}

for pkg in "${CRITICAL_FILES[@]}"; do
    check_file "$pkg"
done

# 统计 Whl 总数
TOTAL_WHL=$(ls $WHEEL_DIR/*.whl 2>/dev/null | wc -l)
echo "📦 总 Whl 文件数: $TOTAL_WHL"

if [ "$MISSING_COUNT" -gt 0 ]; then
    echo "------------------------------------------------"
    echo "⛔ 自检失败！缺失 $MISSING_COUNT 个关键文件。"
    echo "👉 请检查上方的错误信息，不要拷贝到 GPU 服务器！"
    echo "   可能是网络超时，请重新运行脚本补全下载。"
    exit 1
fi

if [ "$TOTAL_WHL" -lt 50 ]; then
    echo "------------------------------------------------"
    echo "⚠️  警告: 下载的文件数量似乎过少 ($TOTAL_WHL < 50)。"
    echo "   可能有大量依赖未下载成功。建议重新运行。"
    # 这里不强制退出，但给出强警告
fi

echo "------------------------------------------------"
echo "🎉 自检通过！全量包构建成功。"
echo "📂 最终产物路径: $SAVE_DIR"
