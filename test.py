# -*- coding: utf-8 -*-
# @Time    : 2022-05-27 17:26
# @Author  : WZS
# @Software: PyCharm
import os
import torch
import numpy as np
import itertools
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn.functional as F
import argparse
from models import choose_model
import torchvision
from scipy import interp
from itertools import cycle
from tqdm import tqdm

test_transform = transforms.Compose([
    transforms.Resize(519),
    # 从图像中心裁切224x224大小的图片
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def test(test_data_loader):
    model.eval()  # 测试模式
    pred = []
    trued = []
    score_list = []  # 存储预测得分
    label_list = []  # 存储真实标签
    top1_correct = 0  # 用于计算Top-1准确率
    top5_correct = 0  # 用于计算Top-5准确率
    total_samples = 0  # 总样本数
    for batch in tqdm(test_data_loader):
        inputs, target = batch
        inputs, target = inputs.to(device), target.to(device)
        output = model.forward(inputs)
        pre = torch.max(F.softmax(output, dim=1), dim=1)[1]
        # pred.append(pre)
        # trued.append(traget)

        # 计算 Top-1 accuracy
        top1_correct += torch.sum(pre == target).item()

        # 计算 Top-5 accuracy
        num_classes = output.size(1)
        k = min(5, num_classes)
        _, top5_pred = torch.topk(output, k, dim=1)
        top5_correct += torch.sum(top5_pred == target.view(-1, 1)).item()

        total_samples += target.size(0)

        pred += list(pre.cpu().numpy())
        trued += list(target.cpu().numpy())

        score_tmp = output  # (batchsize, nclass)

        score_list.extend(score_tmp.detach().cpu().numpy())
        label_list.extend(target.cpu().numpy())

    top1_accuracy = top1_correct / total_samples
    top5_accuracy = top5_correct / total_samples

    print(f'Top-1 Accuracy: {top1_accuracy:.4f}')
    print(f'Top-5 Accuracy: {top5_accuracy:.4f}')

    return pred, trued, score_list, label_list


def plot_confusion_matrix(cm, target_names,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens,  # 这个地方设置混淆矩阵的颜色主题
                          normalize=True):
    # np.trace返回对角线的和////np.sum(cm)返回类别数
    # 绘制热力图
    plt.figure(figsize=(15, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    # 显示坐标轴刻度的名字
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    # 写数值
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    #画图
    #自动调整子图参数，使之填充整个图像区域
    plt.tight_layout()
    save_dir = ""
    plt.savefig(os.path.join(save_dir, 'Confusion Matrix.jpg'))
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="resnet50")
    parser.add_argument('--weight_path', type=str, default="run/train_230827_183917/weights.pth")  # 保存的权重的路径
    parser.add_argument('--test_path', type=str, default="new_data/test")
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    classes = ["circle",  'line', 'rectangle','triangle']# 类别名
    # 加载模型结构
    model = choose_model(args.model_name, num_class=args.num_classes, pretrained=False, progress=True)
    # 加载训练好的权重
    model = torch.load(args.weight_path)
    model.to(device)
    # 导入测试集
    test_data = torchvision.datasets.ImageFolder(root=args.test_path, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, drop_last=True)
    # 模型推理
    pred, trued, score_list, label_list = test(test_data_loader=test_loader)

    predictions = [classes[i] for i in pred]
    true = [classes[i] for i in trued]
    # 画混淆矩阵
    conf_mat = confusion_matrix(y_true=true, y_pred=predictions)
    plot_confusion_matrix(conf_mat, normalize=False, target_names=classes, title='Confusion Matrix')

    # 画PR曲线
    score_array = np.array(score_list)
    # 将label转换成onehot形式
    label_tensor = torch.tensor(label_list)
    label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
    label_onehot = torch.zeros(label_tensor.shape[0], args.num_classes)
    label_onehot.scatter_(dim=1, index=label_tensor, value=1)
    label_onehot = np.array(label_onehot)

    # 调用sklearn库，计算每个类别对应的precision和recall
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(args.num_classes):
        precision[i], recall[i], _ = precision_recall_curve(label_onehot[:, i], score_array[:, i])
        average_precision[i] = average_precision_score(label_onehot[:, i], score_array[:, i])

    precision["micro"], recall["micro"], _ = precision_recall_curve(label_onehot.ravel(), score_array.ravel())
    average_precision["micro"] = average_precision_score(label_onehot, score_array, average="micro")

    plt.plot(recall['micro'], precision['micro'],
             label="average P_R(area={0:0.2f})".format(average_precision["micro"]))
    for i in range(args.num_classes):
        plt.plot(recall[i], precision[i], label="P_R curve of class{0}(area={1:0.2f})".format(i, average_precision[i]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower right")
    plt.savefig("test_pr_curve.jpg")
    # =================================================================================================
    # 调用sklearn库，计算每个类别对应的fpr和tpr
    fpr_dict = dict()
    tpr_dict = dict()
    roc_auc_dict = dict()
    for i in range(args.num_classes):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(label_onehot[:, i], score_array[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
    # micro
    fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(label_onehot.ravel(), score_array.ravel())
    roc_auc_dict["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])

    # macro
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(args.num_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(args.num_classes):
        mean_tpr += interp(all_fpr, fpr_dict[i], tpr_dict[i])
    # Finally average it and compute AUC
    mean_tpr /= args.num_classes
    fpr_dict["macro"] = all_fpr
    tpr_dict["macro"] = mean_tpr
    roc_auc_dict["macro"] = auc(fpr_dict["macro"], tpr_dict["macro"])

    # 绘制所有类别平均的roc曲线
    plt.figure()
    lw = 2
    plt.plot(fpr_dict["micro"], tpr_dict["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc_dict["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr_dict["macro"], tpr_dict["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc_dict["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(args.num_classes), colors):
        plt.plot(fpr_dict[i], tpr_dict[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc_dict[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig('set113_roc.jpg')
    plt.show()

    # =================================================================================================
    # 打印其他评价指标
    print(classification_report(true, predictions, target_names=classes))
