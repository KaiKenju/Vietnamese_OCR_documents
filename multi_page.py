# from paddleocr import PaddleOCR, draw_ocr
# from vietocr.tool.predictor import Predictor
# from vietocr.tool.config import Cfg
# import cv2
# import numpy as np
# from PIL import Image
# import time
# import concurrent.futures
# import fitz  # PyMuPDF
# from docx import Document
# import io
# from image_processing import (
#     preprocess_image, detect_table_edges, deskew, enhance_brightness, convert_pdf_to_docx
# )

# class OCRProcessor:
#     def __init__(self):
#         self.ocr = self.initialize_ocr()
#         self.detector = self.initialize_vietocr()

#     def initialize_ocr(self):
#         return PaddleOCR(lang='en', use_gpu=False)

#     def initialize_vietocr(self):
#         config = Cfg.load_config_from_file('./config/config_after_trainer.yml')
#         config['weights'] = './weight/transformerocr.pth'
#         config['cnn']['pretrained'] = False
#         config['device'] = 'cpu'
#         return Predictor(config)

#     def recognize_text(self, box, masked_img):
#         cropped_image = masked_img[box[0][1]:box[1][1], box[0][0]:box[1][0]]
#         cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
#         cropped_image = Image.fromarray(cropped_image)
#         return self.detector.predict(cropped_image)

#     def process_image(self, img):
#         img = enhance_brightness(img, 240)
#         img = preprocess_image(img)
#         img = deskew(img)
#         img = preprocess_image(img)

#         start_time_detection = time.time()
#         detection_result = self.ocr.ocr(img, cls=False, det=True, rec=False)
#         end_time_detection = time.time()
#         self.execution_time_detection = end_time_detection - start_time_detection

#         boxes = [[[[int(line[0][0]), int(line[0][1])], [int(line[2][0]), int(line[2][1])]] for line in detection_result[0]]][0]

#         EXPAND = 7
#         for box in boxes:
#             box[0][0] -= EXPAND
#             box[0][1] -= EXPAND
#             box[1][0] += EXPAND
#             box[1][1] += EXPAND

#         masked_img, edges_on_white = detect_table_edges(img)
#         return boxes, masked_img

#     def recognize_texts(self, boxes, masked_img):
#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             results = [executor.submit(self.recognize_text, box, masked_img) for box in boxes]
#         return [result.result() for result in results]

#     def pdf_to_images(self, pdf_path):
#         pdf_document = fitz.open(pdf_path)
#         images = []
#         for page_num in range(pdf_document.page_count):
#             page = pdf_document.load_page(page_num)
#             pix = page.get_pixmap()
#             img = Image.open(io.BytesIO(pix.tobytes()))
#             img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
#             images.append(img_cv)
#         return images

#     def execute(self, pdf_path, docx_path):
#         start_time_total = time.time()
#         images = self.pdf_to_images(pdf_path)

#         doc = Document()

#         for page_num, img in enumerate(images):
#             boxes, masked_img = self.process_image(img)

#             start_time_recognition = time.time()
#             texts = self.recognize_texts(boxes, masked_img)
#             end_time_recognition = time.time()

#             for text in texts:
#                 doc.add_paragraph(text)

#             doc.add_page_break()

#             print(f"Processed page {page_num + 1}/{len(images)}")
#             print(f"Detection time: {self.execution_time_detection:.2f} s")
#             print(f"Recognition time: {end_time_recognition - start_time_recognition:.2f} s")

#         doc.save(docx_path)

#         end_time_total = time.time()
#         print(f"Total time: {end_time_total - start_time_total:.2f} s")

# if __name__ == "__main__":
#     ocr_processor = OCRProcessor()
#     ocr_processor.execute(
#         pdf_path='699-ttg.signed.pdf',
#         docx_path='multipage.docx'
#     )

from paddleocr import PaddleOCR
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import cv2
import numpy as np
from PIL import Image
import time
import concurrent.futures
import fitz  # PyMuPDF
from docx import Document
import io
from image_processing import (
    preprocess_image, detect_table_edges, deskew, enhance_brightness, convert_pdf_to_docx
)

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

        img = enhance_brightness(img, 240)
        img = preprocess_image(img)
        img = deskew(img)
        img = preprocess_image(img)

        start_time_detection = time.time()
        detection_result = self.ocr.ocr(img, cls=False, det=True, rec=False)
        end_time_detection = time.time()
        self.execution_time_detection = end_time_detection - start_time_detection

        boxes = [[[[int(line[0][0]), int(line[0][1])], [int(line[2][0]), int(line[2][1])]] for line in detection_result[0]]][0]

        EXPAND = 7
        for box in boxes:
            box[0][0] -= EXPAND
            box[0][1] -= EXPAND
            box[1][0] += EXPAND
            box[1][1] += EXPAND

        masked_img, edges_on_white = detect_table_edges(img)
        return boxes, masked_img

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

    def execute(self, pdf_path, docx_path):
        start_time_total = time.time()
        images = self.pdf_to_images(pdf_path)
        if not images:
            raise ValueError("No images were loaded from the PDF")

        doc = Document()

        for page_num, img in enumerate(images):
            boxes, masked_img = self.process_image(img)

            start_time_recognition = time.time()
            texts = self.recognize_texts(boxes, masked_img)
            end_time_recognition = time.time()

            for text in reversed(texts):
                doc.add_paragraph(text)

            doc.add_page_break()

            print(f"Processed page {page_num + 1}/{len(images)}")
            print(f"Detection time: {self.execution_time_detection:.2f} s")
            print(f"Recognition time: {end_time_recognition - start_time_recognition:.2f} s")

        doc.save(docx_path)

        end_time_total = time.time()
        print(f"Total time: {end_time_total - start_time_total:.2f} s")

if __name__ == "__main__":
    ocr_processor = OCRProcessor()
    ocr_processor.execute(
        pdf_path='VanBanGoc_54.2014.QH13.pdf',
        docx_path='multi_page_output.docx'
    )
