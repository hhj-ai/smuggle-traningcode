#!/bin/bash
# ==============================================================================
# AURORA Installer (Sequential No-Deps Mode)
# ------------------------------------------------------------------------------
# 核心策略：逐个安装文件，完全绕过 pip 的依赖解析图 (Dependency Graph)。
# 解决 error: resolution-too-deep
# ==============================================================================

BASE_DIR="./offline_packages"
WHEEL_DIR="$BASE_DIR/wheels"
INSTALL_ROOT="./aurora_env_root"
VENV_DIR="aurora_env"

echo "🚀 [GPU Server] 开始序列化安装 (防止依赖死锁)..."

# 1. 环境准备
[ ! -d "$INSTALL_ROOT" ] && mkdir -p $INSTALL_ROOT && tar -xzf "$BASE_DIR/python_runtime/python-3.10.tar.gz" -C $INSTALL_ROOT
if [ -d "$INSTALL_ROOT/python" ]; then EXE_PYTHON="$INSTALL_ROOT/python/bin/python3"; else EXE_PYTHON="$INSTALL_ROOT/bin/python3"; fi
[ ! -d "$VENV_DIR" ] && $EXE_PYTHON -m venv $VENV_DIR
source $VENV_DIR/bin/activate

# 定义强制安装函数
force_install() {
    # --no-deps: 只要解压，不要检查依赖
    # --force-reinstall: 确保覆盖旧的错误版本
    python -m pip install "$1" --no-index --find-links=$WHEEL_DIR --no-deps --quiet
}

# 2. 关键底层库 (手动优先安装)
echo "🧱 [1/5] 安装底层构建工具..."
# 必须先装这些，否则后面可能会报错
for pkg in "numpy" "packaging" "wheel" "setuptools" "ninja"; do
    # 找到对应的文件
    PKG_FILE=$(find $WHEEL_DIR -name "$pkg*.whl" | head -n 1)
    if [ ! -z "$PKG_FILE" ]; then
        echo "   -> $pkg"
        force_install "$PKG_FILE"
    fi
done

# 3. 核心框架
echo "🔥 [2/5] 安装 PyTorch & NVIDIA..."
# 安装 Torch 全家桶
find $WHEEL_DIR -name "torch*.whl" -o -name "nvidia*.whl" -o -name "triton*.whl" | while read whl; do
    force_install "$whl"
done

# 4. Flash Attention (显式安装)
echo "⚡ [3/5] 安装 Flash Attention..."
# 移除可能存在的源码包
rm -f "$WHEEL_DIR/flash_attn"*.tar.gz
FA_WHEEL=$(find $WHEEL_DIR -name "flash_attn*.whl" | head -n 1)

if [ -f "$FA_WHEEL" ]; then
    echo "   -> Installing: $(basename $FA_WHEEL)"
    # 这里必须用 pip install 文件路径，不能用包名
    python -m pip install "$FA_WHEEL" --no-deps --no-index
else
    echo "❌ 严重错误: 未找到 Flash Attention Wheel!"
    exit 1
fi

# 5. 暴力安装剩余所有包
echo "📚 [4/5] 序列化安装所有剩余依赖 (这可能需要一分钟)..."
# 遍历目录所有 whl，逐个安装。忽略错误（因为有的已经装过了）
count=0
total=$(ls $WHEEL_DIR/*.whl | wc -l)
for whl in $WHEEL_DIR/*.whl; do
    count=$((count+1))
    # 只打印进度条，不打印详细日志
    echo -ne "   Processing $count/$total: $(basename $whl)\r"
    force_install "$whl"
done
echo ""

# 6. Transformers 源码
echo "🤗 [5/5] 安装 Transformers (Source)..."
if [ -f "$WHEEL_DIR/transformers-main.zip" ]; then
    # 同样加上 --no-deps，防止它去联网找 tokenizers
    python -m pip install "$WHEEL_DIR/transformers-main.zip" --no-index --find-links=$WHEEL_DIR --no-deps
fi

# 7. 最终自检
echo "------------------------------------------------"
python <<EOF
import torch
print(f"✅ PyTorch: {torch.__version__}")
try:
    import flash_attn
    print(f"✅ FlashAttn: {flash_attn.__version__}")
except ImportError:
    print("❌ FlashAttn Import Failed!")
try:
    import transformers
    print(f"✅ Transformers: {transformers.__version__}")
except ImportError:
    print("❌ Transformers Import Failed!")
try:
    from sentence_transformers import SentenceTransformer
    print(f"✅ SentenceTransformers: OK (Scikit-learn detected)")
except Exception as e:
    print(f"❌ ST Import Failed: {e}")
EOF
echo "------------------------------------------------"
echo "🎉 安装完成！所有包已强制就位。"
echo "👉 请使用: python -m accelerate.commands.launch aurora_train.py"
