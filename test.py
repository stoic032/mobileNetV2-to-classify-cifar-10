import os

from PIL import Image, ImageDraw, ImageFont
import torch

from torchvision import transforms
from model import MobileNetV2


def main():
    classes = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 定义数据转换
    data_transform = transforms.Compose([
        transforms.Resize([32, 32]),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载图片
    img_path = 'airplane.jpg'
    assert os.path.exists(img_path), "file: '{}' does not exist.".format(img_path)
    img = Image.open(img_path)

    # 应用转换
    img = data_transform(img)
    # 扩展批次维度
    img = torch.unsqueeze(img, dim=0)

    # 加载模型
    model = MobileNetV2(num_classes=10).to(device)
    weights_path = "./MobileNetV2.pth"
    assert os.path.exists(weights_path), "file: '{}' does not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    model.eval()
    with torch.no_grad():
        # 预测
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    # 输出结果
    print("Predicted class: {}   Probability: {:.3f}".format(classes[predict_cla], predict[predict_cla].numpy()))

    # 可选：在图像上绘制结果
    img = Image.open(img_path)  # 重新加载原始图像，因为之前的图像已经被转换了
    draw = ImageDraw.Draw(img)
    text = "Class: {}   Prob: {:.3f}".format(classes[predict_cla], predict[predict_cla].numpy())
    draw.text((10, 10), text, fill='red')
    img.show()

    # 打印所有类别的概率
    for i in range(len(predict)):
        print("Class: {:10}   Prob: {:.3f}".format(classes[i], predict[i].numpy()))

if __name__ == '__main__':
    main()
