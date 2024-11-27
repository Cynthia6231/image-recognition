# -*- coding: utf-8 -*-

import torch
import os
# import PySide2
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import argparse
from models import choose_model
import matplotlib.pyplot as plt

# dirname = os.path.dirname(PySide2.__file__)
# plugin_path = os.path.join(dirname, 'plugins', 'platforms')
# os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

test_transform = transforms.Compose([
    transforms.Resize(519),
    # 从图像中心裁切224x224大小的图片
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def test1(image):
    model.eval()  # 测试模式
    inputs = image.to(device)
    output = model.forward(inputs)
    pre = F.softmax(output, dim=1)
    return pre


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="resnet50")
    parser.add_argument('--weight_path', type=str, default="run/train_241009_142101/weights.pth")
    parser.add_argument('--image_path', type=str, default="data_photos/circle/4.jpg")
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    classes = ["circle",  'line', 'rectangle','triangle']  # 类别名
    # 加载模型结构
    model = choose_model(args.model_name, num_class=args.num_classes, pretrained=False, progress=True)
    # 加载训练好的权重
    model = torch.load(args.weight_path)
    model.to(device)
    # 加载图片
    plt.figure(figsize=(15, 12))
    image = Image.open(args.image_path).convert('RGB')
    img = test_transform(image)
    img = torch.unsqueeze(img, dim=0)
    pre = test1(img)

    pre = pre.cpu().detach().numpy()
    print(pre)
    #print(pre[0][2])

    # 打印每个类别的得分
    for i, score in enumerate(pre[0]):
        class_name = classes[i]
        print("{}: {:0.2f}%".format(class_name, score * 100))

    #plt.text(x=1100.2, y=0, s="Defect:{:0.2f}%, No Defect:{:0.2f}%".format(classes[0]*100, classes[1]*100,classes[2]*100), fontsize=20)
        # 显示图片
    plt.figure(figsize=(15, 12))
    with Image.open(args.image_path).convert('RGB') as image:
            plt.imshow(image)

    # 在图片上添加预测信息
    text = ""
    for i, class_name in enumerate(classes):
        class_score = pre[0][i] * 100
        text += "{}: {:0.2f}%".format(class_name, class_score)
        if i < len(classes) - 1:
            text += ", "

    plt.text(10, 10, text, fontsize=30, color='red', backgroundcolor='black')
    plt.show()
