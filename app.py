from flask import Flask, render_template, request
import cv2
import os

app = Flask(__name__)

# 加载人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    # 获取上传的文件
    file = request.files['imageInput']

    # 保存上传的文件
    file_path = 'static/' + file.filename
    file.save(file_path)

    # 加载图像
    image = cv2.imread(file_path)

    # 将图像转换为灰度
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 进行人脸检测
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 对检测到的人脸进行模糊处理
    for (x, y, w, h) in faces:
        # 提取人脸区域
        face_roi = image[y:y+h, x:x+w]
        
        # 对人脸区域进行模糊处理
        blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
        
        # 将模糊处理后的人脸区域放回原图像
        image[y:y+h, x:x+w] = blurred_face

    # 保存处理后的图像
    output_path = 'static/blurred_image.jpg'
    cv2.imwrite(output_path, image)

    return render_template('result.html', input_image=file_path, output_image=output_path)

if __name__ == '__main__':
    app.run(debug=True)