import os
import sys

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import device, optim
from tqdm import tqdm

from model import MobileNetV2
from dataload import train_loader, test_loader


def device_initialize():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        device = torch.device("cuda:0")
        print(f"Using {num_gpus} GPU(s).")
        print(f"Selected GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("No GPU available, using CPU.")
    return device


def model_train_and_test(model, train_loader, test_loader, criterion, optimizer, save_path, log_interval, n_epochs, device):
    train_avg_losses = []
    test_avg_losses = []
    model_accuracies = []

    for epoch in range(1, n_epochs + 1):
        train_losses = []
        test_losses = []
        # train
        model.train()

        batch_i = 0
        for batch_index, (inputs, labels) in enumerate(train_loader):
            batch_i = batch_index
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            if batch_index % log_interval == 0:
                print("正在训练第 {} 个epoch >> [{}/{}] >> loss: {:.6f}".format(
                    epoch, batch_index * len(inputs), len(train_loader.dataset), loss.item()))

        print("第{}个epoch训练完成！！！".format(epoch))
        train_avg_losses.append(sum(train_losses) / batch_i)

        # test
        model.eval()
        correct = 0
        batch_i = 0
        with torch.no_grad():
            for batch_index, (inputs, targets) in enumerate(test_loader):
                batch_i = batch_index

                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_losses.append(loss.item())
                predicts = outputs.argmax(dim=1, keepdim=True)
                correct += predicts.eq(targets.view_as(predicts)).sum().item()

        accuracy = 100. * correct / len(test_loader.dataset)
        test_avg_losses.append(sum(test_losses) / batch_i)

        model_accuracies.append(accuracy)

        print('\nTest集的平均损失：{:.4f}，模型准确率：{}/{} ({:.0f}%)\n'.format(
        test_avg_losses[-1], correct, len(test_loader.dataset), accuracy))

        # 保存训练好的权重
        torch.save(model.state_dict(), save_path)

    return train_avg_losses, test_avg_losses, model_accuracies


def visualize(train_losses, test_losses, model_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 4))

    # 绘制训练损失和测试损失
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'o-', label='Train Loss')
    plt.plot(epochs, test_losses, 'o-', label='Test Loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制模型准确率
    plt.subplot(1, 2, 2)
    plt.plot(epochs, model_accuracies, 'o-', color='green', label='Model Accuracy')
    for i, acc in enumerate(model_accuracies):
        plt.text(epochs[i], acc, f"{acc:.2f}", ha='center', va='bottom')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()




def run(model, device, n_epochs):
    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)
    save_path = './MobileNetV2.pth'
    log_interval = 10

    train_losses, test_losses, model_accuracies = model_train_and_test(model, train_loader, test_loader,
                                                                       criterion, optimizer,
                                                                       save_path, log_interval, n_epochs, device)

    visualize(train_losses, test_losses, model_accuracies)


if __name__ == '__main__':
    # 模型配置
    model = MobileNetV2(num_classes=10)
    # 加载 mobileNetV2 的预训练权重
    # download url: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
    model_weight_path = "./mobilenet_v2-pre.pth"
    assert os.path.exists(model_weight_path), "预训练模型权重：file {} dose not exist.".format(model_weight_path)
    pre_weights = torch.load(model_weight_path, map_location='cpu')

    # 遍历预训练权重字典（pre_weights），只保留那些与当前模型中同名参数具有相同尺寸的键-值对，并将它们保存在pre_dict中
    pre_dict = {k: v for k, v in pre_weights.items() if model.state_dict()[k].numel() == v.numel()}
    # 将上一步筛选出的pre_dict中的权重加载到模型中，strict=False表示允许加载不完全匹配的权重，可能会有一些不匹配的权重被忽略
    missing_keys, unexpected_keys = model.load_state_dict(pre_dict, strict=False)

    # 冻结了网络中的特征提取器（features）的权重，使其在训练过程中不再更新
    for param in model.features.parameters():
        param.requires_grad = False

    device = device_initialize()
    model.to(device)

    run(model, device, n_epochs=10)

