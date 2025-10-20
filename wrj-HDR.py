import os
import sys
from ctypes import cdll, c_char_p, c_size_t, c_int

# ====== 基础路径 ======
BASE_PATH = "/data2/home/user/27/Algorithm"

def check_license():
    """
    调用 /data1/libs/librunner.so 完成 License 校验
    """
    lib_path = "/data1/libs/librunner.so"
    lic_path = "/data1/libs/lic.txt"

    # ===== 检查文件是否存在 =====
    if not os.path.exists(lib_path):
        print(f"[License Error] Missing library file: {lib_path}")
        sys.exit(1)
    if not os.path.exists(lic_path):
        print(f"[License Error] Missing license file: {lic_path}")
        sys.exit(1)

    try:
        # ===== 加载动态库 =====
        lib = cdll.LoadLibrary(lib_path)

        # ===== 定义函数原型 =====
        lib.getHashCode.argtypes = [c_char_p, c_size_t]
        lib.getHashCode.restype = c_int
        lib.validFile.argtypes = [c_char_p]
        lib.validFile.restype = c_int

        # ===== 调用校验函数 =====
        file_path_bytes = lic_path.encode('utf-8')
        result = lib.validFile(file_path_bytes)

        if result != 0:
            print("[License Error] License validation failed. Please check lic.txt.")
            sys.exit(1)
        else:
            print("[License] License validation passed. Authorization confirmed")

    except Exception as e:
        print(f"[License Exception] {e}")
        sys.exit(1)


# ====== 在算法启动前执行 License 校验 ======
check_license()

import json
import time
import logging
import argparse
import numpy as np
import cv2
from flask import Flask, request, jsonify
import torch
import os.path as osp

# 添加项目代码路径
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'codes'))

# 导入项目模块
import codes.options.options as option
import codes.utils.util as util
from codes.models import create_model
from codes.data import create_dataset
import codes.data.util as data_util
import tempfile
import shutil

app = Flask(__name__)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 全局变量
model = None
opt = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init_model():
    """初始化模型"""
    global model, opt, device
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('-opt', type=str, default='codes/options/test/test_HDRUNet.yml',
                            help='Path to options YAML file.')
        args = parser.parse_args(['-opt', 'codes/options/test/test_HDRUNet.yml'])

        opt = option.parse(args.opt, is_train=False)
        opt = option.dict_to_nonedict(opt)

        # 修改模型路径为latest_G.pth
        opt['path']['pretrain_model_G'] = 'pretrained_models/latest_G.pth'

        # 创建模型
        model = create_model(opt)
        logger.info('Model initialized successfully')

    except Exception as e:
        logger.error(f"Service initialization failed: {str(e)}")
        raise RuntimeError("500: Internal Server Error: Service initialization failed") from e


def pad_image(img):
    """对图像进行padding，使其尺寸能被64整除"""
    h, w = img.shape[:2]
    pad_h = (64 - h % 64) % 64
    pad_w = (64 - w % 64) % 64

    if pad_h > 0 or pad_w > 0:
        img_padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
        return img_padded, (h, w)

    return img, (h, w)


def process_image(input_path, output_dir, file_base):
    """处理图像并保存结果 - 使用原始数据处理管道"""
    try:
        # 保持原有的padding逻辑
        img_LQ = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img_LQ is None:
            return False, f"Internal Server Error: Image file not found at path: {input_path}", None

        # 确保图像是3通道的
        if len(img_LQ.shape) == 2:
            img_LQ = cv2.cvtColor(img_LQ, cv2.COLOR_GRAY2BGR)
        elif img_LQ.shape[2] == 1:
            img_LQ = cv2.cvtColor(img_LQ, cv2.COLOR_GRAY2BGR)
        elif img_LQ.shape[2] == 4:
            img_LQ = cv2.cvtColor(img_LQ, cv2.COLOR_BGRA2BGR)

        original_size = img_LQ.shape[:2]
        img_LQ_padded, original_size = pad_image(img_LQ)
        
        # 创建临时目录和文件，使用原始数据处理管道
        temp_dir = tempfile.mkdtemp()
        try:
            # 保存padded图像到临时文件，使用与原始数据处理一致的格式
            temp_image_path = os.path.join(temp_dir, "temp_input.png")
            # 确保保存为uint8格式，这样read_img函数会正确处理
            if img_LQ_padded.dtype != np.uint8:
                img_LQ_padded_uint8 = (img_LQ_padded * 255).astype(np.uint8)
            else:
                img_LQ_padded_uint8 = img_LQ_padded
            cv2.imwrite(temp_image_path, img_LQ_padded_uint8)
            
            # 创建临时数据集配置
            dataset_opt = {
                'name': 'temp_test',  # 添加缺失的name字段
                'mode': 'LQ_condition',
                'dataroot_LQ': temp_dir,
                'condition': 'image',  # 使用image condition，与test配置一致
                'data_type': 'img'
            }
            
            # 使用原始数据处理管道
            dataset = create_dataset(dataset_opt)
            data = dataset[0]  # 获取预处理后的数据
            
            # 添加batch维度，因为模型期望batch输入
            data['LQ'] = data['LQ'].unsqueeze(0)  # 添加batch维度 [C,H,W] -> [1,C,H,W]
            data['cond'] = data['cond'].unsqueeze(0)  # 添加batch维度
            
            # 模型推理
            model.feed_data(data, need_GT=False)
            model.test()
            visuals = model.get_current_visuals(need_GT=False)

            # 后处理：使用原始的tensor2numpy函数
            sr_img = util.tensor2numpy(visuals['SR'])
            
            # 保持原有的去padding逻辑
            sr_img = sr_img[:original_size[0], :original_size[1], :]

            # 保持原有的保存逻辑
            os.makedirs(output_dir, exist_ok=True)
            output_image_path = os.path.join(output_dir, f"{file_base}_result.png")
            alignratio_path = os.path.join(output_dir, f"{file_base}_result_ratio.npy")
            util.save_img_with_ratio(output_image_path, sr_img, alignratio_path)

            # 返回相对路径
            output_image_rel = os.path.relpath(output_image_path, BASE_PATH)
            return True, "Image processed successfully", output_image_rel
            
        finally:
            # 清理临时目录
            shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        logger.error(f"Internal Server Error: Failed to process image - {str(e)}")
        return False, f"Internal Server Error: Failed to process image - {str(e)}", None


@app.errorhandler(404)
def not_found_error(error):
    return jsonify({
        'code': 404,
        'message': "error,Not Found: The requested URL was not found on the server",
        'output_image': None
    }), 404


@app.errorhandler(405)
def method_not_allowed_error(error):
    return jsonify({
        'code': 405,
        'message': "error,Method Not Allowed: Only POST requests are accepted",
        'output_image': None
    }), 405


@app.errorhandler(415)
def unsupported_media_type_error(error):
    return jsonify({
        'code': 415,
        'message': "error,Unsupported Media Type: Request must be JSON format",
        'output_image': None
    }), 415


@app.route('/wrj_HDR', methods=['POST'])
def process():
    """API接口处理函数"""
    try:
        if not request.is_json:
            return jsonify({
                'code': 415,
                'message': "error,Unsupported Media Type: Request must be JSON format",
                'output_image': None
            }), 415

        data = request.get_json()

        if not data or 'image' not in data:
            return jsonify({
                'code': 400,
                'message': "error,Bad Request: Must provide 'image' (file path)",
                'output_image': None
            }), 400

        # ===== 拼接成绝对路径 =====
        relative_path = data['image']
        input_path = os.path.join(BASE_PATH, relative_path)

        # ===== 输出目录 =====
        output_dir = os.path.join(BASE_PATH, os.path.dirname(relative_path).replace("INPUT", "OUTPUT"))
        file_base = os.path.splitext(os.path.basename(relative_path))[0]
        os.makedirs(output_dir, exist_ok=True)

        # ===== 处理图像 =====
        success, message, output_image = process_image(input_path, output_dir, file_base)

        if success:
            return jsonify({
                'code': 200,
                'output_image': output_image,
                'message': f"success,{message}"
            })
        else:
            return jsonify({
                'code': 500,
                'message': f"error,{message}",
                'output_image': None
            }), 500

    except Exception as e:
        logger.error(f"Internal Server Error: {str(e)}")
        return jsonify({
            'code': 500,
            'message': f"error,Internal Server Error: {str(e)}",
            'output_image': None
        }), 500


if __name__ == '__main__':
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("503: GPU unavailable")
        init_model()
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)

    app.run(host='0.0.0.0', port=7651, debug=False)