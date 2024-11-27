import os
import uuid  # 用于生成唯一文件名
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F
from models import choose_model

app = Flask(__name__)

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# 图像预处理
test_transform = transforms.Compose([
    transforms.Resize(519),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# 选择模型
# 直接加载整个模型
def load_model(weight_path, model_name='resnet50', num_classes=4):
    # 定义模型架构
    model = choose_model(model_name, num_class=num_classes, pretrained=False, progress=True)

    model = torch.load(weight_path, map_location=torch.device('cpu'))  # 使用 CPU 加载模型
    model.eval()  # 设置为评估模式
    return model




# 模型推断
def predict_image(image, model):
    image = test_transform(image).unsqueeze(0)  # 预处理图片并扩展维度
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
    return probabilities

# 判断文件扩展名是否合法
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# 上传图片处理
@app.route("/", methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # 检查是否上传了图片
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # 为文件生成唯一的文件名，保存到 static 目录下
            unique_filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
            filepath = os.path.join('static', unique_filename)
            file.save(filepath)

            # 打开图片并转换为 RGB 格式
            try:
                image = Image.open(filepath).convert('RGB')
            except Exception as e:
                return f"Error loading image: {str(e)}"

            # 加载模型并进行预测
            model = load_model('run/train_241009_171950/weights.pth')
            probabilities = predict_image(image, model).cpu().numpy()

            # 传回前端显示结果
            classes = ["circle", 'line', 'rectangle', 'triangle']
            class_scores = {classes[i]: round(probabilities[0][i] * 100, 2) for i in range(len(classes))}


            # 传递相对路径给前端
            return render_template('result.html', filepath=unique_filename, class_scores=class_scores)

    return render_template('upload.html')


if __name__ == "__main__":
    app.run(debug=True)
