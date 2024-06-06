from paddleocr import PaddleOCR, draw_ocr

from image_processing import is_image_sharp, sharpen_image, preprocess_image, detect_table_edges, detect_table_corners, calculate_rotation_angle, process_and_save, convert_pdf_to_docx, rotate_image, deskew, brightness, enhance_brightness

from paddleocr import PaddleOCR
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import cv2
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
import time
import concurrent.futures
from reportlab.lib.pagesizes import letter
# # Khởi tạo PaddleOCR

ocr = None
def initialize_ocr():
    global ocr
    if ocr is None:
        ocr = PaddleOCR(lang='en', use_gpu=False)
start_init = time.time()
initialize_ocr()
end_init = time.time()
execution_time_init = end_init - start_init
# # Hàm nhận diện văn bản từ một phần của ảnh
def recognize_text(box, masked_img):
    cropped_image = masked_img[box[0][1]:box[1][1], box[0][0]:box[1][0]]
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    cropped_image = Image.fromarray(cropped_image)
    rec_result = detector.predict(cropped_image)
    return rec_result

# # Khởi tạo VietOCR
# config = Cfg.load_config_from_name('vgg_transformer')
# config['weights'] = './weight/vgg_transformer.pth'
config = Cfg.load_config_from_file('./config/config_after_trainer.yml')
config['weights'] = './weight/transformerocr.pth'
config['cnn']['pretrained'] = False
config['device'] = 'cpu'#'cuda:0' if use gpu
#config['device'] = 'cuda:0' 
detector = Predictor(config)


def main():
    start_time_total = time.time()

    img_path = './assets/anh12.png'
    img = cv2.imread(img_path)
    start_time_processing = time.time()
    # Tăng cường độ sáng nếu cần
    img = enhance_brightness(img, 240)
    img = preprocess_image(img)
    img = deskew(img) # check rotation 
    img = preprocess_image(img)

    detection_result = ocr.ocr(img, cls=False, det=True, rec=False)
    boxes = [[[[int(line[0][0]), int(line[0][1])], [int(line[2][0]), int(line[2][1])]] for line in detection_result[0]]][0]

    # Mở rộng kích thước của bounding box
    EXPAND = 7
    for box in boxes:
        box[0][0] -= EXPAND
        box[0][1] -= EXPAND
        box[1][0] += EXPAND
        box[1][1] += EXPAND


    masked_img, edges_on_white = detect_table_edges(img)

    end_time_processing = time.time()
    execution_time_processing = end_time_processing - start_time_processing


    start_time_recognition = time.time()
    # Recognition with multithreading
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Gửi các công việc nhận diện văn bản đến các luồng khác nhau
        results = [executor.submit(recognize_text, box, masked_img) for box in boxes]

    # Thu thập kết quả từ các luồng
    texts = [result.result() for result in results]
    end_time_recognition = time.time()
    execution_time_recognition = end_time_recognition - start_time_recognition
    
    page_width, page_height = letter
    c = canvas.Canvas("ocr_final.pdf", pagesize=(page_width, page_height))
    pdfmetrics.registerFont(TTFont('Times New Roman', 'times.ttf'))
    # image_height = letter

    for text, box in zip(texts, boxes):
        x1, y1 = box[0]
        x2, y2 = box[1]
        # x1 += 10  # dịch phải 5 pixel 30
        # x2 += 5  # dịch phải 5 pixel
        y1 += 10  # dịch xuống 10 pixel 
        y2 += 10

        # if x2 > page_width:
        # # Điều chỉnh lại vị trí của bounding box để nằm trong giới hạn của trang PDF
        #     x2 = page_width
        # # Tính lại vị trí x của điểm đỉnh bên trái của bounding box
        #     # x1 = x2 - (box[1][0] - box[0][0])
        #     # x1 += 100
            
        #     x1 = max(0, x2 - (x2 - x1))
          
        y1, y2 = page_height - y1, page_height - y2
        c.setFont("Times New Roman", 12)
        c.drawString(x1, y1, text)
    c.save()
    pdf_file = 'ocr_final.pdf'
    docx_file = 'ocr_final_word.docx'
    convert_pdf_to_docx(pdf_file, docx_file)
   
    end_time_total = time.time()
    execution_time_total = end_time_total - start_time_total
    
    print("Paddle load:", execution_time_init, "s")
    print("Image processing time :", execution_time_processing, "s")
    print("Recognition time:", execution_time_recognition, "s")
    
    print("Total:", execution_time_total, "s")
    
if __name__ == "__main__":
    main()
    
