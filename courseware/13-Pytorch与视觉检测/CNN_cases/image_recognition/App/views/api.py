import io
from io import BytesIO
from PIL import Image
import base64
import pandas as pd
import os
from flask import Blueprint, request,render_template,redirect,json,Response
import torch
from App.models import *
from torchvision import transforms
import pandas as pd

# 加载预训练模型
model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet152', pretrained=True)
model.eval()

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
api = Blueprint("api", __name__, url_prefix='/')
@api.route("")
def home():
    return redirect("/image")


@api.route("/image", methods=["GET", "POST"])
def index():
    if request.method == 'POST':
        # 读取文件
        file = request.files.get('file')
        if not file:
            return
        figfile = io.BytesIO(file.read())
        input_image = image = Image.open(figfile)

        # 读取添加的信息read added information
        user_input = request.form.get("name")
        #print(user_input)
        print(ROOT_PATH+'/imagenet_class.csv')

        # 使用Pytorch模型对图片进行识别
        #res_img = model.forward(image)
        # 待检测图片
        classes = pd.read_csv(ROOT_PATH+'/imagenet_class.csv', sep=',', header=None)
        preprocess = transforms.Compose([
        	transforms.Resize(256),
        	transforms.CenterCrop(224),
        	transforms.ToTensor(),
        	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

        # 判断是否使用GPU
        if torch.cuda.is_available():
        	input_batch = input_batch.to('cuda')
        	model.to('cuda')
        with torch.no_grad():
        	output = model(input_batch)
        # 输出classID
        result = classes[classes[0]==int(output[0].argmax())]
        # 显示分类名称
        print(result.iloc[0][1])

        # 保存在本地
        output_buffer = BytesIO()  # for PIL.Image
        image.save(output_buffer, format='JPEG')
        img = base64.b64encode(output_buffer.getvalue()).decode('ascii')  
        return render_template('result.html', user_input=user_input, img=img, class_id = result.iloc[0][0], class_name=result.iloc[0][1])

    return render_template('index.html')






