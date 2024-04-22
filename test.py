import cv2
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import concurrent.futures 
from concurrent.futures import ThreadPoolExecutor
from paddleocr import PaddleOCR, draw_ocr
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from image_processing import preprocess_image, detect_table_edges, detect_table_corners
import time

# Khởi tạo OCR
ocr = PaddleOCR(lang='en', use_gpu=False)
texts = []
def process_roi(img, box):
    # Cắt và lấy ROI từ ảnh dưới dạng numpy array
    cropped_image = img[box[0][1]:box[1][1], box[0][0]:box[1][0]]
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    cropped_image = Image.fromarray(cropped_image)
    rec_result = detector.predict(cropped_image)
    return rec_result

def process_image_multithreaded(img, boxes):
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Dùng dictionary comprehension để lưu trữ các future object
        future_to_box = {executor.submit(process_roi, img, box): box for box in boxes}
        for future in concurrent.futures.as_completed(future_to_box):
            results.append(future.result())
    return results


# Khởi tạo và cấu hình VietOCR
config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = './weight/vgg_transformer.pth'
config['cnn']['pretrained'] = False
config['device'] = 'cpu'
detector = Predictor(config)

# Đọc và xử lý ảnh gốc
img_path = './assets/don-khoi-kien-vu-an-hanh-chinh-9418.png'
img_ori = cv2.imread(img_path)
img = preprocess_image(img_ori)
corners, rotation_angle = detect_table_corners(img)

# Xoay ảnh nếu cần
if rotation_angle != 0:
    center = (img.shape[1] // 2, img.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    img = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))

# Phát hiện vùng quan tâm và nhận dạng ký tự bằng đa luồng
detection_result = ocr.ocr(img, cls=False, det=True, rec=False)
boxes = [[[[int(line[0][0]), int(line[0][1])], [int(line[2][0]), int(line[2][1])]] for line in detection_result[0]]][0]

for box in boxes:
    box[0][0] -= 5
    box[0][1] -= 5
    box[1][0] += 5
    box[1][1] += 5

texts = process_image_multithreaded(img, boxes)

# Hiển thị và lưu ảnh kết quả
plt.figure(figsize=(6, 5))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('OCR Final')
# for text, box in zip(texts, boxes):
#     plt.text(box[0][0], box[0][1], str(text), ha='left', va='top', color='red', fontdict={'family': 'sans-serif', 'size': 7})
plt.axis('off')
plt.savefig('ocr_final_image_with_boxes.jpg', bbox_inches='tight', pad_inches=0, dpi=300)
plt.show()
