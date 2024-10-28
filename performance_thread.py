import gridfs
from pymongo import MongoClient
from paddleocr import PaddleOCR, draw_ocr
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import cv2
import os
import numpy as np
from PIL import Image
import time
import concurrent.futures
from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.pagesizes import letter
from image_processing import (
    is_image_sharp, sharpen_image, preprocess_image, detect_table_edges,
    detect_table_corners, calculate_rotation_angle, process_and_save,
    convert_pdf_to_docx, rotate_image, deskew, brightness, enhance_brightness
)

class OCRProcessor:
    def __init__(self):
        self.ocr = self.initialize_ocr()
        self.detector = self.initialize_vietocr()
        self.client = MongoClient('mongodb://localhost:27017/')  # Kết nối tới MongoDB
        self.db = self.client['OCRdata']  # Thay thế với tên cơ sở dữ liệu của bạn
        self.fs = gridfs.GridFS(self.db)

    def initialize_ocr(self):
        ocr = PaddleOCR(lang='en', use_gpu=False)
        return ocr

    def initialize_vietocr(self):
        config = Cfg.load_config_from_file('./config/config_after_trainer.yml')
        config['weights'] = './weight/transformerocr.pth'
        config['cnn']['pretrained'] = False
        config['device'] = 'cpu'
        detector = Predictor(config)
        return detector

    def recognize_text(self, box, masked_img):
        cropped_image = masked_img[box[0][1]:box[1][1], box[0][0]:box[1][0]]
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        cropped_image = Image.fromarray(cropped_image)
        rec_result = self.detector.predict(cropped_image)
        return rec_result

    def process_image(self, img_path):
        start_time_total = time.time()
        img = cv2.imread(img_path)

        # Tăng cường độ sáng nếu cần
        img = enhance_brightness(img, 240)
        img = preprocess_image(img)
        img = deskew(img)
        img = preprocess_image(img)

        start_time_detection = time.time()
        detection_result = self.ocr.ocr(img, cls=False, det=True, rec=False)
        end_time_detection = time.time()
        self.execution_time_detection = end_time_detection - start_time_detection

        boxes = [[[[int(line[0][0]), int(line[0][1])], [int(line[2][0]), int(line[2][1])]] for line in detection_result[0]]][0]

        # Mở rộng kích thước của bounding box
        EXPAND = 7
        for box in boxes:
            box[0][0] -= 0
            box[0][1] -= 0
            box[1][0] += EXPAND
            box[1][1] += EXPAND

        masked_img, edges_on_white = detect_table_edges(img)

        return boxes, masked_img, img

    def recognize_texts(self, boxes, masked_img):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = [executor.submit(self.recognize_text, box, masked_img) for box in boxes]
        texts = [result.result() for result in results]
        return texts

    def create_pdf(self, texts, boxes, img_shape, pdf_path):
        page_width, page_height = letter
        c = canvas.Canvas(pdf_path, pagesize=(page_width, page_height))
        pdfmetrics.registerFont(TTFont('Times New Roman', 'times.ttf'))

        for text, box in zip(texts, boxes):
            x1, y1 = box[0]
            x2, y2 = box[1]
            y1 += 10
            y2 += 10

            y1, y2 = page_height - y1, page_height - y2
            c.setFont("Times New Roman", 12)
            c.drawString(x1, y1, text)
        c.save()

    def convert_pdf_to_docx1(self, pdf_file, docx_file):
        convert_pdf_to_docx(pdf_file, docx_file)

    def save_image_to_mongodb(self, img_path):
        with open(img_path, 'rb') as img_file:
            img_id = self.fs.put(img_file, filename=os.path.basename(img_path))  # Lưu ảnh và lấy ID
            print(f"Image saved with ID: {img_id}")
            
    
    def execute(self, img_path, pdf_path, docx_path):
        start_time_total = time.time()
        boxes, masked_img, img = self.process_image(img_path)

        # Lưu hình ảnh vào MongoDB
        self.save_image_to_mongodb(img_path)

        start_time_recognition = time.time()
        texts = self.recognize_texts(boxes, masked_img)
        end_time_recognition = time.time()

        self.create_pdf(texts, boxes, img.shape, pdf_path)
        self.convert_pdf_to_docx1(pdf_path, docx_path)
        self.save_file_to_mongodb(docx_path)
        self.save_file_to_mongodb(pdf_path)

        end_time_total = time.time()

        print(f"Thời gian phát hiện của PaddleOCR: {self.execution_time_detection:.2f} s")
        print(f"Thời gian nhận diện: {end_time_recognition - start_time_recognition:.2f} s")
        print(f"Tổng thời gian: {end_time_total - start_time_total:.2f} s")

    def save_file_to_mongodb(self,pdf_path):
        with open(pdf_path, 'rb') as file:
            file_id = self.fs.put(file, filename=pdf_path.split('/')[-1])  # Lưu tệp và lấy ID
            print(f"File saved with ID: {file_id}")
if __name__ == "__main__":
    ocr_processor = OCRProcessor()
    ocr_processor.execute(
        img_path='./assets/test_ne.png',
        pdf_path='ocr_final.pdf',
        docx_path='ocr_final_word.docx'
    )
