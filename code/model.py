# model.py
import torch
from flask import Flask, request, jsonify
import model_loading_script  # 导入加载模型的脚本
import MFCC
import os
import tempfile

# 在 predict 函数外部创建临时文件夹
TEMP_FOLDER = tempfile.mkdtemp()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 创建一个字典，将预测的类别编号映射到鸟类编号和种类
bird_classes = {
    0: {"编号": 17, "种类": "大天鹅"},
    1: {"编号": 34, "种类": "绿头鸭"},
    2: {"编号": 114, "种类": "雉鸡"},
    3: {"编号": 180, "种类": "苍鹭"},
    4: {"编号": 202, "种类": "普通鸬鹚"},
    5: {"编号": 298, "种类": "黑翅长脚鹬"},
    6: {"编号": 300, "种类": "凤头麦鸡"},
    7: {"编号": 368, "种类": "红脚鹬"},
    8: {"编号": 370, "种类": "林鹬"},
    9: {"编号": 1331, "种类": "麻雀"}
}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './weights'
app.config['ALLOWED_EXTENSIONS'] = {'pth'}
# app.secret_key = 'supersecretkey'  # 为了使用 flash 消息


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/predict', methods=['POST'])
def predict():
    # 获取上传的文件
    uploaded_file = request.files['file']

    # 检查是否上传了文件
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    # 生成临时文件路径
    temp_file_path = os.path.join(TEMP_FOLDER, 'uploaded_audio.wav')

    # 保存上传的文件到临时文件夹
    # temp_file_path = '/tmp/uploaded_audio.wav'
    uploaded_file.save(temp_file_path)

    # 调用 MFCC 函数处理上传的文件
    MFCC_audio_file = MFCC.get_MFCC(temp_file_path)

    # 删除临时文件
    os.remove(temp_file_path)

    # 将 numpy 数组转换为 PyTorch 张量
    MFCC_audio_tensor = torch.from_numpy(MFCC_audio_file).unsqueeze(0).float()

    # 加载模型
    model = model_loading_script.load_model_and_weights("./weights/best_model.pth")
    if model is None:
        return jsonify({'error': 'Failed to load model'}), 500

    # 在模型中进行预测
    with torch.no_grad():
        model.eval()
        prediction = model(MFCC_audio_tensor)

    print(prediction)

    # 获取预测结果的索引
    predicted_class = torch.argmax(prediction, dim=1).item()

    # 将预测结果转换为鸟类编号和种类
    if predicted_class in bird_classes:
        bird_info = bird_classes[predicted_class]
        bird_number = bird_info["编号"]
        bird_species = bird_info["种类"]
    else:
        bird_number = -1
        bird_species = "Unknown"

    return jsonify({'prediction': predicted_class, 'no': bird_number, 'specie': bird_species})


@app.route('/upload_model', methods=['POST'])
def upload_model():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = 'best_model.pth'
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return jsonify({'message': 'Model uploaded and replaced successfully'}), 200
    else:
        return jsonify({'error': 'File type not allowed'}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
