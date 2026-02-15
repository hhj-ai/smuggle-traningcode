import struct

# 你的第一个模型文件路径 (请确认路径无误)
file_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/smuggle-traningcode/models/Qwen3-VL-8B-Instruct/model-00001-of-00004.safetensors"

print(f"正在检查文件: {file_path}")

try:
    with open(file_path, "rb") as f:
        # 读取前8个字节 (Safetensors 的长度字段)
        header_len_bytes = f.read(8)
        
        print(f"🔹 前8字节 (Hex): {header_len_bytes.hex()}")
        
        # 如果全是 0，说明是空洞文件
        if header_len_bytes == b'\x00\x00\x00\x00\x00\x00\x00\x00':
            print("❌ [确诊] 这是一个“空洞文件”！")
            print("   原因：DolphinFS 同步了文件名和大小，但数据全是0。")
            print("   解决：必须删除文件重新拷贝/下载。")
        else:
            # 尝试解析长度
            length = struct.unpack('<Q', header_len_bytes)[0]
            print(f"🔹 解析出的头长度: {length}")
            
            # 读取接下来的数据看看是不是 JSON
            json_preview = f.read(50)
            print(f"🔹 内容预览: {json_preview}")

except Exception as e:
    print(f"❌ 读取报错: {e}")
