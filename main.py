from paddleocr import PaddleOCR, draw_ocr
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

import cv2
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from image_processing import is_image_sharp, sharpen_image, preprocess_image, detect_table_edges, detect_table_corners, calculate_rotation_angle, process_and_save, convert_pdf_to_docx

from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
import time
# Initialize PaddleOCR
# ocr = PaddleOCR(lang='en')
start_time_total = time.time()
ocr = None
def initialize_ocr():
    global ocr
    if ocr is None:
        ocr = PaddleOCR(lang='en', use_gpu=False)

# Gọi hàm này khi cần sử dụng OCR
start_time_initialize_ocr = time.time()
initialize_ocr()
end_time_initialize_ocr = time.time()
execution_time_initialize_ocr = end_time_initialize_ocr - start_time_initialize_ocr

# Specifying output path and font path.
out_path = './output'
font = './simfang.ttf'

# Initialize VietOCR
# use vgg_seq2seq
# config = Cfg.load_config_from_name('vgg_seq2seq.pth') # load internet
# config['weights'] = 'https://vocr.vn/data/vietocr/vgg_seq2seq.pth'
# # config['pretrain'] = 'https://vocr.vn/data/vietocr/vgg_seq2seq.pth'
# config['cnn']['pretrained']=False
config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = './weight/vgg_transformer.pth' #use weihgt local
# config['weights'] = 'https://vocr.vn/data/vietocr/vgg_transformer.pth'
config['cnn']['pretrained'] = False
config['device'] = 'cpu' #comment this line if you want to use GPU

detector = Predictor(config)

start_time_processing = time.time()
# image path
img_path = './assets/don-khoi-kien-vu-an-hanh-chinh-9418.png'

img_ori = cv2.imread(img_path)

img = preprocess_image(img_ori)

corners, rotation_angle  = detect_table_corners(img)
print("Góc xoay của ảnh là:", rotation_angle, "độ")
# Xoay ảnh
if rotation_angle != 0:
    center = (img.shape[1] // 2, img.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    img = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))

# Thực hiện phát hiện vùng quan tâm bằng PaddleOCR
detection_result = ocr.ocr(img, cls=False, det=True, rec=False)
boxes = [[[[int(line[0][0]), int(line[0][1])], [int(line[2][0]), int(line[2][1])]] for line in detection_result[0]]][0]

# change bounding box size
EXPAND = 5
for box in boxes:
    box[0][0] -= EXPAND
    box[0][1] -= EXPAND
    box[1][0] += EXPAND
    box[1][1] += EXPAND

masked_img, edges_on_white  = detect_table_edges(img) # edge recognition


texts = []
for box in boxes:
    cropped_image = masked_img[box[0][1]:box[1][1], box[0][0]:box[1][0]]
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    cropped_image = Image.fromarray(cropped_image)
    rec_result = detector.predict(cropped_image)
    texts.append(rec_result)

plt.figure(figsize=(6, 5))
plt.imshow(cv2.cvtColor(edges_on_white, cv2.COLOR_BGR2RGB))
plt.title('Image (rotation, Blur, edge)')
plt.axis('off')
plt.savefig('edge_image.jpg', bbox_inches='tight', pad_inches=0,dpi=300)


plt.figure(figsize=(6, 5))
plt.imshow(cv2.cvtColor(edges_on_white, cv2.COLOR_BGR2RGB))
plt.title('OCR Final')

end_time_processing = time.time()
execution_time_processing = end_time_processing - start_time_processing
c = canvas.Canvas("ocr_final.pdf", pagesize=(img.shape[1], img.shape[0]))
pdfmetrics.registerFont(TTFont('Arial', 'Arial.ttf'))
image_height = img.shape[0]

#add text & bbox in image
for text, box in zip(texts, boxes):
    x1, y1 = box[0]
    x2, y2 = box[1]
    x1 += 5  # shift right 5 pixel
    x2 += 5  # shift right 5 pixel
    y1 += 10  # shift down 10 pixel 
    y2 += 10
    
    plt.text(x1, y1, text, ha='left', va='top', wrap=True, linespacing=0.75, fontdict={'family': 'serif', 'size': 7})

    y1, y2 = image_height - y1, image_height - y2
    c.setFont("Arial", 10)
    c.drawString(x1, y1, text)
c.save()
plt.axis('off')
plt.tight_layout()
plt.savefig('ocr_final_image_with_boxes.jpg', bbox_inches='tight', pad_inches=0,dpi=300)

# Gọi hàm chuyển đổi
pdf_file = 'ocr_final.pdf'
docx_file = 'ocr_final_word.docx'
convert_pdf_to_docx(pdf_file, docx_file)

 #Post-processing, analysis 
# # texts_1 = texts[::-1]
# # Specify the folder where you want to save the cropped images
save_folder = 'cropped_images'

# Ensure the folder exists
os.makedirs(save_folder, exist_ok=True)

# Process and save each cropped image
count = 0
for box in reversed(boxes):
    cropped_image = img[box[0][1]:box[1][1], box[0][0]:box[1][0]]
    process_and_save(cropped_image, save_folder, count)
    count += 1

end_time_total = time.time()
execution_time_total = end_time_total - start_time_total
print("Thời gian khởi tạo OCR:", execution_time_initialize_ocr, "giây")
print("Thời gian xử lý ảnh và dự đoán:", execution_time_processing, "giây")
print("Tổng thời gian thực thi:", execution_time_total, "giây")
plt.show()
