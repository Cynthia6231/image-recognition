import os
import random
import shutil


def split_dataset(src_folder, target_folder, train_ratio=0.8, val_ratio=0.1):
    # 获取源文件夹下的所有子文件夹（每个子文件夹代表一个类别）
    class_folders = [os.path.join(src_folder, d) for d in os.listdir(src_folder) if
                     os.path.isdir(os.path.join(src_folder, d))] #['flower_photos/roses', 'flower_photos/sunflowers', 'flower_photos/daisy']

    # 创建目标文件夹及其子文件夹
    split_names = ['train', 'val', 'test']
    for split_name in split_names:
        split_path = os.path.join(target_folder, split_name)
        os.makedirs(split_path, exist_ok=True)
        for class_folder in class_folders:
            class_split_path = os.path.join(split_path, os.path.basename(class_folder))
            os.makedirs(class_split_path, exist_ok=True)

    # 划分并复制数据
    for class_folder in class_folders:
        # 获取当前类别的文件夹名称
        class_name = os.path.basename(class_folder)

        # 获取当前类别文件夹中的所有图像文件
        images = [f for f in os.listdir(class_folder) if os.path.isfile(os.path.join(class_folder, f))]

        # 随机打乱图像顺序，以确保数据随机性
        random.shuffle(images)

        # 根据划分比例计算每个子集的样本数量
        train_count = int(len(images) * train_ratio)
        val_count = int(len(images) * val_ratio)

        # 划分图像为训练集、验证集和测试集
        train_images = images[:train_count]
        val_images = images[train_count:(train_count + val_count)]
        test_images = images[(train_count + val_count):]

        for image in train_images:
            src_path = os.path.join(class_folder, image)
            dst_path = os.path.join(target_folder, 'train', class_name, image)
            shutil.copy2(src_path, dst_path)

        for image in val_images:
            src_path = os.path.join(class_folder, image)
            dst_path = os.path.join(target_folder, 'val', class_name, image)
            shutil.copy2(src_path, dst_path)

        for image in test_images:
            src_path = os.path.join(class_folder, image)
            dst_path = os.path.join(target_folder, 'test', class_name, image)
            shutil.copy2(src_path, dst_path)


if __name__ == '__main__':
    src_data_folder = r"data_photos"
    target_data_folder = "new_data"
    split_dataset(src_data_folder, target_data_folder)
