#!/bin/bash

# ==========================================
# 步骤 2: 离线安装脚本 (运行在无网的 GPU 服务器)
# 目标: 从共享磁盘 ./offline_packages 安装环境
# ==========================================

ENV_NAME="aurora_env"
PKG_DIR="./offline_packages"

echo "🚀 [GPU Server] 开始离线安装..."

# 1. 检查离线包目录
if [ ! -d "$PKG_DIR" ]; then
    echo "❌ 错误: 未找到离线包目录 $PKG_DIR"
    echo "   请先在 CPU 服务器上运行 1_cpu_download.sh"
    exit 1
fi

# 2. 创建并激活虚拟环境
echo "📦 创建虚拟环境..."
rm -rf $ENV_NAME # 清理旧环境
python3 -m venv $ENV_NAME
source $ENV_NAME/bin/activate

echo "🔌 环境已激活: $(which python)"

# 定义离线安装命令 (关键: --no-index --find-links)
install_offline() {
    pip install "$@" --no-index --find-links=$PKG_DIR
}

# 3. 升级基础工具
echo "🔧 [1/6] 升级 pip/setuptools..."
install_offline --upgrade pip wheel setuptools

# 4. 安装构建依赖 (Flash-Attn 编译必需)
echo "🧱 [2/6] 安装构建工具 (ninja, psutil)..."
install_offline ninja packaging psutil

# 5. 安装 PyTorch (必须先装，Flash-Attn 依赖它)
echo "🔥 [3/6] 安装 PyTorch (CUDA 12.1)..."
install_offline torch torchvision torchaudio

# 6. 编译安装 Flash Attention 2
echo "⚡ [4/6] 编译并安装 Flash Attention 2..."
echo "   (这一步需要调用 nvcc 编译，可能需要几分钟，请耐心等待)"
# --no-build-isolation: 使用当前环境已安装的 torch/ninja 进行编译
install_offline flash-attn --no-build-isolation

# 7. 安装 Transformers (最新版)
echo "🤗 [5/6] 安装 Transformers (Local)..."
install_offline transformers

# 8. 安装其余所有依赖
echo "📚 [6/6] 安装剩余依赖..."
install_offline \
    accelerate \
    datasets \
    huggingface_hub \
    sentence-transformers \
    numpy \
    Pillow \
    easyocr \
    scipy \
    termcolor \
    timm \
    rich \
    questionary \
    aiohttp \
    requests \
    protobuf \
    sentencepiece

echo "------------------------------------------------"
echo "🎉 离线环境安装完成！"
echo "👉 激活命令: source $ENV_NAME/bin/activate"
