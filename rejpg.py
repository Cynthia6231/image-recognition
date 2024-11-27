import os

# 指定目标目录路径
directory = "data_photos/line"  # 将 "your_directory_path" 替换为你实际的目录路径

# 遍历文件夹中的所有文件
for filename in os.listdir(directory):
    # 分离文件名和扩展名
    name, ext = os.path.splitext(filename)

    # 如果文件没有扩展名或扩展名不为 .jpg，修改它
    if ext != ".jpg":
        # 构造新的文件名
        new_filename = name + ".jpg"

        # 构造完整的旧文件路径和新文件路径
        old_filepath = os.path.join(directory, filename)
        new_filepath = os.path.join(directory, new_filename)

        # 重命名文件
        os.rename(old_filepath, new_filepath)

print("文件扩展名修改完成。")
