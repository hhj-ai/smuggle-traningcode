#!/bin/bash

# ==============================================================================
# AURORA Offline Installer (v2.0)
# ------------------------------------------------------------------------------
# 兼容性：支持 flash_attn 源码包安装。
# ==============================================================================

BASE_DIR="./offline_packages"
WHEEL_DIR="$BASE_DIR/wheels"
INSTALL_ROOT="./aurora_env_root"
VENV_DIR="aurora_env"

echo "🚀 [GPU Server] 开始增量安装流程..."

# 1. 环境准备 (幂等)
if [ ! -d "$INSTALL_ROOT" ]; then
    mkdir -p $INSTALL_ROOT
    tar -xzf "$BASE_DIR/python_runtime/python-3.10.tar.gz" -C $INSTALL_ROOT
fi

if [ -d "$INSTALL_ROOT/python" ]; then EXE_PYTHON="$INSTALL_ROOT/python/bin/python3"; else EXE_PYTHON="$INSTALL_ROOT/bin/python3"; fi

[ ! -d "$VENV_DIR" ] && $EXE_PYTHON -m venv $VENV_DIR
source $VENV_DIR/bin/activate

# 2. 地毯式安装
echo "🧱 [1/3] 安装所有依赖 Wheel..."
# 优先安装核心构建工具
pip install $WHEEL_DIR/packaging-*.whl $WHEEL_DIR/ninja-*.whl $WHEEL_DIR/numpy-*.whl --no-index --find-links=$WHEEL_DIR
# 安装其余所有包
pip install $WHEEL_DIR/*.whl --no-index --find-links=$WHEEL_DIR --no-deps --quiet 2>/dev/null

# 3. 核心框架安装
echo "🔥 [2/3] 安装 PyTorch & Flash Attention..."
pip install torch torchvision torchaudio --no-index --find-links=$WHEEL_DIR

# 编译 Flash Attention
if python -c "import flash_attn" > /dev/null 2>&1; then
    echo "   ✅ Flash Attention 已就绪。"
else
    echo "   ⚡ 正在编译 Flash Attention (可能需要 10 分钟)..."
    FLASH_FILE=$(ls $WHEEL_DIR/flash_attn-*.tar.gz | head -n 1)
    pip install "$FLASH_FILE" --no-index --find-links=$WHEEL_DIR --no-build-isolation
fi

# 4. Transformers
echo "🤗 [3/3] 安装 Transformers..."
if [ -f "$WHEEL_DIR/transformers-main.zip" ]; then
    pip install "$WHEEL_DIR/transformers-main.zip" --no-index --find-links=$WHEEL_DIR
fi

# 5. 自检
echo "------------------------------------------------"
python -c "import torch, transformers, rich, aiohappyeyeballs; print('🎉 环境完美激活，所有组件已就绪！')"
