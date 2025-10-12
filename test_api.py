import requests
import json
import os
import argparse

def test_api(image_path):
    """测试API接口"""
    url = "http://127.0.0.1:7651/process"
    
    # 准备请求数据
    data = {
        "image": image_path
    }
    
    # 发送请求
    response = requests.post(url, json=data)
    
    # 打印响应
    print("Status Code:", response.status_code)
    print("Response:")
    print(json.dumps(response.json(), indent=4, ensure_ascii=False))
    
    return response.json()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HDR API")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    args = parser.parse_args()
    
    # 确保输入图像存在
    if not os.path.exists(args.image):
        print(f"Error: Input image {args.image} does not exist")
        exit(1)
    
    # 测试API
    test_api(args.image)