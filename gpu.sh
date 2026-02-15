#!/bin/bash

# ========================================================================
# 2_full_install.sh (GPU 服务器)
# 目标: 解压自带的 Python 3.10 -> 创建环境 -> 离线安装
# ========================================================================

BASE_DIR="./offline_packages"
PYTHON_TGZ="$BASE_DIR/python_runtime/python-3.10.tar.gz"
WHEEL_DIR="$BASE_DIR/wheels"
INSTALL_ROOT="./aurora_env_root"  # 安装根目录

echo "🚀 [GPU Server] 开始全离线部署..."

# 1. 检查文件
if [ ! -f "$PYTHON_TGZ" ]; then
    echo "❌ 错误: 未找到 Python 包 $PYTHON_TGZ"
    exit 1
fi

# 2. 解压独立 Python
echo "🐍 [1/5] 部署 Python 3.10..."
rm -rf $INSTALL_ROOT
mkdir -p $INSTALL_ROOT
# 解压到 install_root/python
tar -xzf $PYTHON_TGZ -C $INSTALL_ROOT
# 移动目录结构，确保 $INSTALL_ROOT/python/bin/python3 存在
# 这里的目录结构取决于压缩包，通常解压后是 python/
if [ -d "$INSTALL_ROOT/python" ]; then
    LOCAL_PYTHON="$INSTALL_ROOT/python/bin/python3"
else
    # 有些包解压出来直接就是 bin/ 等，视情况调整
    LOCAL_PYTHON="$INSTALL_ROOT/bin/python3"
fi

echo "   ✅ 独立 Python 路径: $LOCAL_PYTHON"
$LOCAL_PYTHON --version

# 3. 创建虚拟环境 (使用刚才解压的 python)
echo "📦 [2/5] 创建虚拟环境..."
VENV_DIR="aurora_env"
rm -rf $VENV_DIR

$LOCAL_PYTHON -m venv $VENV_DIR
source $VENV_DIR/bin/activate

echo "🔌 当前环境: $(which python)"
# 验证是否指向了虚拟环境
if [[ "$(which python)" != *"$VENV_DIR"* ]]; then
    echo "❌ 虚拟环境激活失败！"
    exit 1
fi

# 定义安装函数
install_pkg() {
    pip install "$@" --no-index --find-links=$WHEEL_DIR
}

# 4. 基础依赖安装
echo "🔧 [3/5] 安装基础依赖..."
install_pkg --upgrade pip setuptools wheel
install_pkg ninja packaging psutil numpy

# 5. 安装核心组件
echo "🔥 [4/5] 安装 PyTorch & FlashAttn..."
install_pkg torch torchvision torchaudio
# 编译 FlashAttn (这一步会调用 GPU 驱动的 nvcc)
echo "   - Compiling Flash Attention (Wait)..."
install_pkg flash-attn --no-build-isolation

# 6. 安装其余包
echo "📚 [5/5] 安装 Transformers & Tools..."
# 此时安装的是 wheel 包，速度极快
install_pkg transformers
install_pkg accelerate huggingface_hub
install_pkg \
    datasets sentence-transformers Pillow easyocr scipy \
    termcolor timm rich questionary aiohttp protobuf sentencepiece

# 7. 生成启动脚本 (方便以后使用)
cat <<EOF > start_aurora.sh
#!/bin/bash
source $(pwd)/$VENV_DIR/bin/activate
echo "✅ Environment Activated!"
exec "\$@"
EOF
chmod +x start_aurora.sh

echo "------------------------------------------------"
echo "🎉 部署完成！"
echo "👉 启动方式: source $VENV_DIR/bin/activate"
