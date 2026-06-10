<div class="title" align=center>
    <img src="./doc/logo.png" width=500>
    <br>
    <p>
        <img src="https://img.shields.io/badge/License-Apache%202.0-red.svg">
        <img src="https://img.shields.io/badge/python-≥3.6-blue">
        <img src="https://img.shields.io/github/stars/XJTU-WXY/RapidOCRServer?style=social">
    </p>

</div>

## 🚩 简介
一个基于 RapidOCR 官方 API 项目（[RapidAI/RapidOCRAPI](https://github.com/RapidAI/RapidOCRAPI)）重构的 RapidOCR 推理后端，在完全兼容原版 API 规范的基础上，相比原版增加了如下特性：
- [x] 可通过配置文件指定 RapidOCR 所使用的推理引擎（ONNX Runtime / OpenVINO / PaddlePaddle / PyTorch / MNN），以支持 GPU 或 NPU 等硬件加速。
- [x] 支持 RapidOCR 在推理过程中可自定义的全部参数（use_det / use_cls / use_rec / return_word_box / return_single_char_box / text_score / box_thresh / unclip_ratio）。
- [x] 支持模型闲时卸载，在指定时长内未接受到请求时暂时卸载模型以节省内存或显存。

## 📥 部署
```bash
git clone https://github.com/XJTU-WXY/RapidOCRServer
cd RapidOCRServer
pip install .
```
默认仅支持`onnxruntime`推理引擎，如需支持其他推理引擎，则需额外安装支持库，具体参见 RapidOCR 官方文档[使用不同推理引擎](https://rapidai.github.io/RapidOCRDocs/main/install_usage/rapidocr/how_to_use_infer_engine/)。

## 🔑 使用方法
### ▶️ 启动服务端
#### 1. 以默认配置启动
```bash
# 默认配置启动（RapidOCR 默认模型与参数，监听 0.0.0.0:9003）
rapidocr_server

# 指定端口、进程数量、空闲卸载超时时长（以分钟为单位）
rapidocr_server --listen 0.0.0.0 --port 9005 --workers 2 --idle-timeout 15
```
#### 2. 以指定配置文件启动（使用其他推理引擎、指定推理参数）
```bash
# 先生成默认配置文件
rapidocr config
```
这会在当前工作目录生成`default_rapidocr.yaml`文件，配置方法详见 RapidOCR 官方文档[参数介绍](https://rapidai.github.io/RapidOCRDocs/main/install_usage/rapidocr/parameters/)。

``` bash
# 指定配置文件启动
rapidocr_server --config default_rapidocr.yaml

# 指定端口、进程数量、空闲卸载超时时长（以分钟为单位）
rapidocr_server --config default_rapidocr.yaml --listen 0.0.0.0 --port 9005 --workers 2 --idle-timeout 15
```
### 🔌 调用接口

以 POST 方式调用，与官方 API 调用方式兼容，可接受文件或 base64 图片数据。具体可访问`http://{IP}:{Port}/docs`查看 FastAPI 交互式 API 文档。
#### Python 调用示例
```python
import requests

url = 'http://localhost:9003/ocr'
img_path = 'tests/test_files/ch_en_num.jpg'

# ---- 以文件形式发送 -----------------
with open(img_path, 'rb') as f:
    file_dict = {'image_file': (img_path, f, 'image/png')}
    response = requests.post(url, files=file_dict, timeout=60)

print(response.json())

# ---- 以 base64 形式发送 ------------
with open(img_path, 'rb') as fa:
    img_str = base64.b64encode(fa.read())

payload = {'image_data': img_str}
response = requests.post(url, data=payload, timeout=60)

print(response.json())

# ---- 自定义推理参数 -----------------
with open(img_path, 'rb') as f:
    data = {"use_det": False, "use_cls": True, "text_score": 0.7, "box_thresh": 0.7}
    response = requests.post(url, files=file_dict, data=data, timeout=60)
print(response.json())
```
### 📑 返回值
识别正常时，返回值与官方 API 格式一致：
```json
{
    "0": {
        "rec_txt": "正品促销", # 识别的文本
        "dt_boxes": [  # 依次为左上角 → 右上角 → 右下角 → 左下角
            [0,0],
            [322,1],
            [322,147],
            [0,110]
        ],
        "score": 0.99459    # 置信度
    },
    "1": {
        "rec_txt": "大桶装更划算",
        "dt_boxes": [
            [68,94],
            [256,94],
            [256,129],
            [68,129]
        ],
        "score": 0.99859
    }
}
```
识别出错时，返回错误原因：
```json
{
    "error": "ImageDecodeError",
    "message": "Unrecognized image format: cannot identify image file <tempfile.SpooledTemporaryFile object at 0x0000020A699E0A30>"
}
```
## ⚖ 开源协议
本项目基于 [RapidAI/RapidOCRAPI](https://github.com/RapidAI/RapidOCRAPI) 重构，并基于相同的 [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) 协议开源。
  
*Open source leads the world to a brighter future!*