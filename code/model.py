# model.py
import shutil

import torch
from flask import Flask, request, jsonify
import model_loading_script  # 导入加载模型的脚本
import MFCC
import os
import tempfile

from Save_MFCC import save_MFCC

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


# 预测
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


# 上传模型
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


# 提取MFCC特征并且保存为npy文件
@app.route('/save_MFCC', methods=['POST'])
def extract_and_save_features():
    try:
        # 调用 save_MFCC 函数处理和保存特征
        save_MFCC()
        # 返回成功消息
        return jsonify({'message': 'Feature extraction and saving completed successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# 上传保存鸟类音频文件到数据集里
@app.route('/save_audio', methods=['POST'])
def save_audio():
    try:
        # 获取上传的文件
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']

        # 获取编号参数
        bird_number = request.form.get('number')
        if not bird_number:
            return jsonify({'error': 'No number provided'}), 400

        # 检查编号是否在指定列表中
        valid_numbers = ["0017", "0034", "0114", "0180", "0202", "0298", "0300", "0368", "0370", "1331"]
        if bird_number not in valid_numbers:
            return jsonify({'error': 'Invalid number provided'}), 400

        # 检查文件是否允许的类型
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # 生成保存路径
        save_dir = os.path.join('..', 'data', 'BirdsSong-10spec-graduation', bird_number)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, file.filename)
        file.save(file_path)

        return jsonify({'message': 'Audio file saved successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# 复制npy文件，用新的训练集替换原来的训练集
@app.route('/copy_MFCC', methods=['POST'])
def copy_mfcc():
    try:
        source_file = '../data/MFCC_train/MFCC_train_combine.npy'
        destination_file = '../data/MFCC_train_combine.npy'

        # 检查源文件是否存在
        if not os.path.exists(source_file):
            return jsonify({'error': 'Source file does not exist'}), 400

        # 复制文件
        shutil.copyfile(source_file, destination_file)

        source_file = '../data/MFCC_train/MFCC_train_label_combine.npy'
        destination_file = '../data/MFCC_train_label_combine.npy'

        # 检查源文件是否存在
        if not os.path.exists(source_file):
            return jsonify({'error': 'Source file does not exist'}), 400

        # 复制文件
        shutil.copyfile(source_file, destination_file)

        return jsonify({'message': 'File copied successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
