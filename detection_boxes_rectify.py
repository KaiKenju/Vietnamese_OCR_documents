import cv2
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg


import os
from PIL import Image
import matplotlib.pyplot as plt
from image_processing import is_image_sharp, sharpen_image, preprocess_image, detect_table_edges, detect_table_corners, calculate_rotation_angle, process_and_save, convert_pdf_to_docx

from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
import time

# Khởi tạo PaddleOCR với model mặc định
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Đọc ảnh đầu vào
image_path = './assets/quote_rotation.png'
image = cv2.imread(image_path)
result = ocr.ocr(image_path, cls=True)
# Sử dụng PaddleOCR để phát hiện và nhận diện văn bản
config = Cfg.load_config_from_file('./config/config_after_trainer.yml')
config['weights'] = './weight/transformerocr.pth'
config['cnn']['pretrained'] = False
config['device'] = 'cpu'

detector = Predictor(config)

# Chuyển đổi bounding boxes thành dạng numpy array
boxes = [line[0] for line in result[0]]

# Chức năng để sửa bounding boxes
def rectify_box(box, image):
    box = np.array(box).astype(np.float32)
    width = int(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3])))
    height = int(max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2])))

    # Điểm đích để wrap bounding box
    dst_pts = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    # Tính ma trận biến đổi
    M = cv2.getPerspectiveTransform(box, dst_pts)
    # Warp vùng chứa bounding box
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped

# Sửa bounding boxes và lưu kết quả
rectified_images = []
for idx, box in enumerate(boxes):
    rectified_image = rectify_box(box, image)
    rectified_images.append(rectified_image)
    #cv2.imwrite(f'rectified_box_{idx}.jpg', rectified_image)

# Chuyển đổi bounding boxes sang dạng phù hợp
#boxes = [[[[int(line[0][0]), int(line[0][1])], [int(line[2][0]), int(line[2][1])]] for line in result[0]]][0]

# Thay đổi kích thước bounding box
EXPAND = 5
for box in boxes:
    box[0][0] -= EXPAND
    box[0][1] -= EXPAND
    box[1][0] += EXPAND
    box[1][1] += EXPAND

texts = []
for idx, box in enumerate(boxes):
    cropped_image = rectified_images[idx]
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    cropped_image = Image.fromarray(cropped_image)
    rec_result = detector.predict(cropped_image)  # Sử dụng PaddleOCR để nhận diện văn bản từ ảnh đã cắt
    texts.append(rec_result)

# Hiển thị kết quả
for idx, text in enumerate(texts):
    print(f'Text from rectified box {idx}: {text}')

c = canvas.Canvas("ocr_detect.pdf", pagesize=(image.shape[1], image.shape[0]))
pdfmetrics.registerFont(TTFont('Times New Roman', 'times.ttf'))
image_height = image.shape[0]

#add text & bbox in image
for text, box in zip(texts, boxes):
    x1, y1 = box[0]
    x2, y2 = box[1]
    x1 += 5  # shift right 5 pixel
    x2 += 5  # shift right 5 pixel
    y1 += 10  # shift down 10 pixel 
    y2 += 10
    
   
    y1, y2 = image_height - y1, image_height - y2
    c.setFont("Times New Roman", 11)
    c.drawString(x1, y1, text)
c.save()

# Gọi hàm chuyển đổi
pdf_file = 'ocr_detect.pdf'
docx_file = 'ocr_detect.docx'
convert_pdf_to_docx(pdf_file, docx_file)