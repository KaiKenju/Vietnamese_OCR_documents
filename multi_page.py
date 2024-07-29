from paddleocr import PaddleOCR
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import io
import concurrent.futures
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from image_processing import (convert_pdf_to_docx)
import time

class OCRProcessor:
    def __init__(self):
        self.ocr = self.initialize_ocr()
        self.detector = self.initialize_vietocr()

    def initialize_ocr(self):
        return PaddleOCR(lang='en', use_gpu=False)

    def initialize_vietocr(self):
        config = Cfg.load_config_from_file('./config/config_after_trainer.yml')
        config['weights'] = './weight/transformerocr.pth'
        config['cnn']['pretrained'] = False
        config['device'] = 'cpu'
        return Predictor(config)

    def recognize_text(self, box, masked_img):
        cropped_image = masked_img[box[0][1]:box[1][1], box[0][0]:box[1][0]]
        if cropped_image.size == 0:
            return ""
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        cropped_image = Image.fromarray(cropped_image)
        return self.detector.predict(cropped_image)

    def process_image(self, img):
        if img is None:
            raise ValueError("Image not loaded properly")

        detection_result = self.ocr.ocr(img, cls=False, det=True, rec=False)
        boxes = [[[[int(line[0][0]), int(line[0][1])], [int(line[2][0]), int(line[2][1])]] for line in detection_result[0]]][0]

        EXPAND = 7
        for box in boxes:
            box[0][0] -= EXPAND
            box[0][1] -= EXPAND
            box[1][0] += EXPAND
            box[1][1] += EXPAND

        return boxes, img

    def recognize_texts(self, boxes, masked_img):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = [executor.submit(self.recognize_text, box, masked_img) for box in boxes]
        return [result.result() for result in results]

    def pdf_to_images(self, pdf_path):
        pdf_document = fitz.open(pdf_path)
        images = []
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes()))
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            if img_cv is None or img_cv.size == 0:
                continue
            images.append(img_cv)
        return images

    def create_blank_pdf_with_texts(self, images, boxes_list, texts_list, pdf_path):
        page_width, page_height = letter
        c = canvas.Canvas(pdf_path, pagesize=(page_width, page_height))
        pdfmetrics.registerFont(TTFont('Times New Roman', 'times.ttf'))

        for page_num, img in enumerate(images):
            for box, text in zip(boxes_list[page_num], texts_list[page_num]):
                x0, y0 = box[0]
                x1, y1 = box[1]
                c.setFont("Times New Roman", 11)
                c.drawString(x0, page_height - y1, text)
            c.showPage()

        c.save()
    
    def execute(self, pdf_path, output_pdf_path, output_docx_path):
        start_time_total = time.time()
        images = self.pdf_to_images(pdf_path)
        if not images:
            raise ValueError("No images were loaded from the PDF")

        boxes_list = []
        texts_list = []
        for img in images:
            boxes, masked_img = self.process_image(img)
            boxes_list.append(boxes)
            texts = self.recognize_texts(boxes, masked_img)
            texts_list.append(texts)

        self.create_blank_pdf_with_texts(images, boxes_list, texts_list, output_pdf_path)
        convert_pdf_to_docx(output_pdf_path, output_docx_path)
        
        end_time_total = time.time()
        print(f"Tổng thời gian: {end_time_total - start_time_total:.2f} s")
        
if __name__ == "__main__":
    ocr_processor = OCRProcessor()
    ocr_processor.execute(
        pdf_path='New_54.2014.QH13.pdf',
        output_pdf_path='Multi_page_afterOCR.pdf',
        output_docx_path='Multi_page_afterOCR.docx'
    )
