#!/bin/bash
# ==============================================================================
# AURORA Installer (Zero Compilation)
# ==============================================================================

BASE_DIR="./offline_packages"
WHEEL_DIR="$BASE_DIR/wheels"
INSTALL_ROOT="./aurora_env_root"
VENV_DIR="aurora_env"

echo "🚀 [GPU Server] 开始极速安装..."

# 1. 环境准备
[ ! -d "$INSTALL_ROOT" ] && mkdir -p $INSTALL_ROOT && tar -xzf "$BASE_DIR/python_runtime/python-3.10.tar.gz" -C $INSTALL_ROOT
if [ -d "$INSTALL_ROOT/python" ]; then EXE_PYTHON="$INSTALL_ROOT/python/bin/python3"; else EXE_PYTHON="$INSTALL_ROOT/bin/python3"; fi

[ ! -d "$VENV_DIR" ] && $EXE_PYTHON -m venv $VENV_DIR
source $VENV_DIR/bin/activate

# 2. 地毯式安装 (所有依赖)
echo "🧱 [1/3] 安装底层依赖 (Binary Wheels)..."
# 使用通配符安装所有 .whl，pip 会自动处理拓扑顺序，只要所有依赖都在文件夹里
# --no-deps: 既然我们有信心全下载了，就禁止 pip 联网检查依赖
python -m pip install $WHEEL_DIR/*.whl --no-index --find-links=$WHEEL_DIR --no-deps --quiet

# 3. 验证核心组件
echo "🔥 [2/3] 验证核心组件..."
python -m pip install torch torchvision torchaudio flash_attn --no-index --find-links=$WHEEL_DIR

# 4. 安装 Transformers 源码
echo "🤗 [3/3] 安装 Transformers (Source)..."
if [ -f "$WHEEL_DIR/transformers-main.zip" ]; then
    python -m pip install "$WHEEL_DIR/transformers-main.zip" --no-index --find-links=$WHEEL_DIR
fi

# 5. 最终自检
echo "------------------------------------------------"
python <<EOF
import torch, flash_attn, transformers, rich, cv2, skimage
print(f"✅ Python: {torch.__version__} (CUDA Available: {torch.cuda.is_available()})")
print(f"✅ FlashAttn: {flash_attn.__version__}")
print(f"✅ Transformers: {transformers.__version__}")
print(f"✅ OpenCV & Scikit-Image OK")
print("🎉 环境完整性校验通过！")
EOF
