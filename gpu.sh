#!/bin/bash
# ==============================================================================
# AURORA Installer (Final Fix)
# ==============================================================================

BASE_DIR="./offline_packages"
WHEEL_DIR="$BASE_DIR/wheels"
INSTALL_ROOT="./aurora_env_root"
VENV_DIR="aurora_env"

echo "🚀 [GPU Server] 开始极速安装..."

# 1. 清理潜在的干扰项 (非常重要)
# 如果目录下有 flash_attn-xxx.tar.gz，pip 可能会优先尝试编译它，导致报错
# 我们强制删除源码包，只留 .whl
rm -f "$WHEEL_DIR/flash_attn"*.tar.gz

# 2. 环境准备
[ ! -d "$INSTALL_ROOT" ] && mkdir -p $INSTALL_ROOT && tar -xzf "$BASE_DIR/python_runtime/python-3.10.tar.gz" -C $INSTALL_ROOT
if [ -d "$INSTALL_ROOT/python" ]; then EXE_PYTHON="$INSTALL_ROOT/python/bin/python3"; else EXE_PYTHON="$INSTALL_ROOT/bin/python3"; fi
[ ! -d "$VENV_DIR" ] && $EXE_PYTHON -m venv $VENV_DIR
source $VENV_DIR/bin/activate

# 3. 安装依赖 (分批更稳健)
echo "🧱 [1/3] 安装基础依赖 (Numpy, Exceptiongroup)..."
# 优先安装这些底层包
python -m pip install $WHEEL_DIR/numpy*.whl $WHEEL_DIR/exceptiongroup*.whl $WHEEL_DIR/packaging*.whl --no-index --find-links=$WHEEL_DIR

echo "🧱 [2/3] 安装其余所有 Wheel..."
# 安装剩余所有 whl
python -m pip install $WHEEL_DIR/*.whl --no-index --find-links=$WHEEL_DIR --no-deps --quiet 2>/dev/null

# 4. 核心框架验证安装
echo "🔥 [3/3] 验证安装核心框架..."
# 这里不再会有编译过程，因为 .tar.gz 已经被删了，pip 只能用 .whl
python -m pip install torch torchvision torchaudio flash_attn --no-index --find-links=$WHEEL_DIR

# 5. Transformers 源码
if [ -f "$WHEEL_DIR/transformers-main.zip" ]; then
    echo "🤗 安装 Transformers..."
    python -m pip install "$WHEEL_DIR/transformers-main.zip" --no-index --find-links=$WHEEL_DIR
fi

echo "------------------------------------------------"
python <<EOF
import torch, flash_attn, transformers, anyio
print(f"✅ Torch: {torch.__version__}")
print(f"✅ FlashAttn: {flash_attn.__version__}")
print(f"✅ Transformers: {transformers.__version__}")
print(f"✅ AnyIO (requires exceptiongroup): {anyio.__version__}")
print("🎉 环境修复完成！")
EOF
