#!/bin/bash

# ==============================================================================
# AURORA Offline Installer (v3.0 - Zero Compilation)
# ------------------------------------------------------------------------------
# 变更：直接安装预编译的 flash_attn，无需 g++/clang++
# ==============================================================================

BASE_DIR="./offline_packages"
WHEEL_DIR="$BASE_DIR/wheels"
INSTALL_ROOT="./aurora_env_root"
VENV_DIR="aurora_env"

echo "🚀 [GPU Server] 开始安装离线环境..."

# 1. 环境准备
[ ! -d "$INSTALL_ROOT" ] && tar -xzf "$BASE_DIR/python_runtime/python-3.10.tar.gz" -C . # 假设直接解压在当前目录
if [ -d "./aurora_env_root/python" ]; then EXE_PYTHON="./aurora_env_root/python/bin/python3"; else EXE_PYTHON="./aurora_env_root/bin/python3"; fi

[ ! -d "$VENV_DIR" ] && $EXE_PYTHON -m venv $VENV_DIR
source $VENV_DIR/bin/activate

# 2. 地毯式安装所有 Wheel
echo "🧱 [1/2] 安装所有依赖 (含 httpx, hf-xet)..."
# 暴力安装所有下载好的 whl
python -m pip install $WHEEL_DIR/*.whl --no-index --find-links=$WHEEL_DIR --no-deps --quiet 2>/dev/null

# 3. 核心框架安装
echo "🔥 [2/2] 安装核心框架 (秒级完成)..."
# 此时安装 flash_attn 会直接找到预编译的 whl，不再编译
python -m pip install torch torchvision torchaudio flash_attn --no-index --find-links=$WHEEL_DIR

# 4. Transformers
[ -f "$WHEEL_DIR/transformers-main.zip" ] && python -m pip install "$WHEEL_DIR/transformers-main.zip" --no-index --find-links=$WHEEL_DIR

# 5. 最终自检
echo "------------------------------------------------"
python <<EOF
import torch, transformers, flash_attn, httpx
print(f"✅ Torch {torch.__version__}")
print(f"✅ Flash-Attn {flash_attn.__version__} (Installed via pre-built wheel)")
print(f"✅ Transformers {transformers.__version__}")
print(f"🎉 环境完全就绪！")
EOF
