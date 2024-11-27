# -*- coding: utf-8 -*-
# @Author  : WZS
# @Software: PyCharm
import os
import time
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.utils.data
import torch.nn.functional as F
import torchvision
import argparse
from torchvision import transforms
from models import choose_model
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

train_transform = transforms.Compose([
    transforms.Resize(256),
    # 从图像中心裁切224x224大小的图片
    transforms.CenterCrop(224),
    # 随机裁剪图像，所得图像为原始面积的0.2到1之间，高宽比在3/4和4/3之间。
    # 然后，缩放图像以创建224 x 224的新图像
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
    transforms.RandomHorizontalFlip(),
    # 随机更改亮度，对比度和饱和度
    #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor(),
    # 标准化图像的每个通道
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

val_test_transform = transforms.Compose([
    transforms.Resize(256),
    # 从图像中心裁切224x224大小的图片
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def train(model_name, train_data_loader, val_data_loader, save_dir, model, loss_fn, device, optimizer, TensorBoard, epochs=20):
    train_accuracy = []
    val_accuracy = []
    train_lo = []
    val_lo = []
    val_acc = 0.0
    train_loss = 0.0
    val_loss = 0.0
    for epoch in range(epochs):
        # print(time.localtime())
        num_correct_train = 0
        num_train = 0
        model.train()  # 训练模式
        # loop_t = tqdm(train_data_loader)
        with tqdm(train_data_loader) as loop_t:
            for batch in loop_t:
                # t.set_description('Epoch %i' % epoch)
                optimizer.zero_grad()  # 清空上次计算梯度
                inputs, traget = batch
                inputs, traget = inputs.to(device), traget.to(device)  # 将张量放在gpu上，如果可行
                if model_name == "Googlenet":
                    output, aux2, aux1 = model.forward(inputs)
                else:
                    output = model.forward(inputs)
                loss = loss_fn(output, traget)  # 计算loss
                loss.backward()  # 反向传播
                optimizer.step()  # 更新权重
                train_loss += loss.data.item()
                correct_train = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], traget).view(-1)
                num_correct_train += sum(correct_train).item()
                num_train += correct_train.shape[0]
                # 更新信息
                loop_t.set_description(f'Epoch [{epoch+1}/{epochs}]')
                loop_t.set_postfix(train_acc=num_correct_train / num_train)
            train_loss /= len(train_data)

        model.eval()  # 测试模式
        num_correct_val = 0
        num_val = 0
        loop_v = tqdm(val_data_loader)
        for batch in loop_v:
            inputs, traget = batch
            inputs, traget = inputs.to(device), traget.to(device)
            output = model.forward(inputs)
            loss = loss_fn(output, traget)
            val_loss += loss.data.item()
            correct_val = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], traget).view(-1)
            num_correct_val += sum(correct_val).item()
            num_val += correct_val.shape[0]
            # 更新信息
            loop_v.set_description(f'Epoch [{epoch+1}/{epochs}]')
            loop_v.set_postfix(val_acc=num_correct_val / num_val)
        val_loss /= len(val_data)
        print(
            'Epoch [{}/{}],train_loss:{:.4f},val_loss:{:.4f},train_acc:{:.4f},val_acc:{:.4f}'
                .format(epoch + 1, epochs, train_loss, val_loss, num_correct_train / num_train,
                        num_correct_val / num_val))
        if (num_correct_val / num_val) > val_acc:
            print("\033[1;34mval acc from {:.4f} improve to {:.4f},".format(val_acc, (num_correct_val / num_val)),
                  "save the best model weights\033[0m")
            # print("Save the best model weights")
            val_acc = (num_correct_val / num_val)
            torch.save(model, os.path.join(save_dir, 'weights.pth'))
        # 可视化TensorBoard
        if TensorBoard:
            writer.add_scalar('Train Acc', num_correct_train / num_train, epoch)
            writer.add_scalar('Val Acc', num_correct_val / num_val, epoch)
            writer.add_scalar('Train Loss', train_loss, epoch)
            writer.add_scalar('Val Loss', val_loss, epoch)
            writer.flush()
        train_accuracy.append(num_correct_train / num_train)
        val_accuracy.append(num_correct_val / num_val)
        train_lo.append(train_loss)
        val_lo.append(val_loss)
    writer.close()
    loop_t.close()
    loop_v.close()

    #第一张图
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), train_accuracy)
    plt.plot(range(epochs), val_accuracy)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.grid()
    plt.legend(['train', 'val'], loc='upper left')


    # 保存
    plt.savefig(os.path.join(save_dir, 'model_accuracy.png'))

    # 第二张图
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), train_lo)
    plt.plot(range(epochs), val_lo)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.grid()

    # 保存
    plt.savefig(os.path.join(save_dir, 'model_loss.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="resnet50")
    parser.add_argument('--save_path', type=str, default="run/")
    parser.add_argument('--train_path', type=str, default="new_data/train")
    parser.add_argument('--val_path', type=str, default="new_data/val")
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--TensorBoard', type=bool, default=True)
    parser.add_argument('--log_path', type=str, default="log/")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('开始创建保存文件夹')

    timestamp = time.strftime("%y%m%d_%H%M%S", time.localtime())
    save_dir = os.path.join(os.getcwd(), args.save_path, 'train_' + timestamp)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    print('making model save dir:', save_dir, end='\n')

    if args.TensorBoard:
        if not os.path.isdir(args.log_path):
            os.makedirs(args.log_path + "/" + args.model_name + "_" + timestamp)
        print('making log save dir:', args.log_path + args.model_name + "_" + timestamp, end='\n')

        writer = SummaryWriter(args.log_path + args.model_name + "_" + timestamp)

    print('保存文件夹创建完成')
    print('开始训练')

    train_data = torchvision.datasets.ImageFolder(root=args.train_path, transform=train_transform)
    val_data = torchvision.datasets.ImageFolder(root=args.val_path, transform=val_test_transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    # 加载模型，如果需要预训练模型，将pretrained改为True
    model = choose_model(args.model_name, num_class=args.num_classes, pretrained=False, progress=True)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train(train_data_loader=train_loader, val_data_loader=val_loader, save_dir=save_dir,
          model=model, optimizer=optimizer, loss_fn=torch.nn.CrossEntropyLoss(),
          device=device, epochs=args.epochs, model_name=args.model_name, TensorBoard=args.TensorBoard)









