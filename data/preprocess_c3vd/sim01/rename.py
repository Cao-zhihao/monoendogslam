import os

# 设置图片文件所在的目录
folder_path = "./color"  # 替换为你的文件路径

# 遍历所有图片并重命名
for i in range(1, 361):
    old_name = f"{folder_path}/{i:04}.png"  # 旧文件名
    new_name = f"{folder_path}/{i - 1:04}_color.png"  # 新文件名

    # 检查旧文件是否存在，防止出错
    if os.path.exists(old_name):
        os.rename(old_name, new_name)
        print(f"重命名: {old_name} -> {new_name}")
    else:
        print(f"文件不存在: {old_name}")
