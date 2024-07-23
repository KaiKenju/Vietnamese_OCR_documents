# from paddleocr import PaddleOCR
# from vietocr.tool.predictor import Predictor
# from vietocr.tool.config import Cfg
# import cv2
# import numpy as np
# from PIL import Image
# import time
# import concurrent.futures
# import fitz  # PyMuPDF
# from reportlab.pdfgen import canvas
# from reportlab.pdfbase.ttfonts import TTFont
# from reportlab.pdfbase import pdfmetrics
# from reportlab.lib.pagesizes import letter
# from docx import Document
# import io

# from image_processing import preprocess_image, detect_table_edges, deskew, enhance_brightness, convert_pdf_to_docx


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
#         return boxes, masked_img, img

#     def recognize_texts(self, boxes, masked_img):
#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             results = [executor.submit(self.recognize_text, box, masked_img) for box in boxes]
#         texts = [result.result() for result in results]
#         return texts

#     def create_pdf(self, texts, boxes, img_shape, pdf_path):
#         page_width, page_height = letter
#         c = canvas.Canvas(pdf_path, pagesize=(page_width, page_height))
#         pdfmetrics.registerFont(TTFont('Times New Roman', 'times.ttf'))

#         for text, box in zip(texts, boxes):
#             x1, y1 = box[0]
#             x2, y2 = box[1]
#             y1 += 10
#             y2 += 10

#             y1, y2 = page_height - y1, page_height - y2
#             c.setFont("Times New Roman", 12)
#             c.drawString(x1, y1, text)
#         c.save()

#     def convert_pdf_to_docx1(self, pdf_file, docx_file):
#         convert_pdf_to_docx(pdf_file, docx_file)

#     def execute(self, pdf_path, docx_path):
#         start_time_total = time.time()
#         pdf_document = fitz.open(pdf_path)
#         for page_num in range(len(pdf_document)):
#             page = pdf_document.load_page(page_num)
#             pix = page.get_pixmap()
#             img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#             img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

#             boxes, masked_img, img = self.process_image(img_cv)
#             texts = self.recognize_texts(boxes, masked_img)
#             self.create_pdf(texts, boxes, img.shape, f'page_{page_num}.pdf')

#         # Merge all page PDFs into one
#         merged_pdf = fitz.open()
#         for page_num in range(len(pdf_document)):
#             merged_pdf.insert_pdf(fitz.open(f'page_{page_num}.pdf'))
#         merged_pdf.save("ocr_final.pdf")
#         self.convert_pdf_to_docx1("ocr_final.pdf", docx_path)

#         end_time_total = time.time()
#         print(f"Total time: {end_time_total - start_time_total:.2f} s")

# if __name__ == "__main__":
#     ocr_processor = OCRProcessor()
#     ocr_processor.execute(
#         pdf_path='vanbanmulti.pdf',
#         docx_path='vanbanmulti_word.docx'
#     )
import fitz  # PyMuPDF
from paddleocr import PaddleOCR
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image, ImageDraw
import io
from docx import Document

# Khởi tạo PaddleOCR và VietOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Sử dụng ngôn ngữ phù hợp với bạn
config = Cfg.load_config_from_file('./config/config_after_trainer.yml')
config['weights'] = './weight/transformerocr.pth'
config['device'] = 'cpu'  # Hoặc 'cuda' nếu bạn sử dụng GPU
detector = Predictor(config)

def pdf_to_images(pdf_path):
    pdf_document = fitz.open(pdf_path)
    images = []
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes()))
        images.append(img)
    return images

def ocr_and_draw_bounding_boxes(images):
    ocr_results = []
    for img in images:
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        result = ocr.ocr(img_byte_arr, cls=True)
        
        draw = ImageDraw.Draw(img)
        for line in result:
            for bbox in line:
                (x1, y1), (x2, y2), (x3, y3), (x4, y4) = bbox[0]
                text_bbox = [x1, y1, x3 - x1, y3 - y1]
                cropped_img = img.crop((x1, y1, x3, y3))
                text = detector.predict(cropped_img)
                draw.rectangle([x1, y1, x3, y3], outline='red', width=2)
                draw.text((x1, y1 - 10), text, fill='red')
                ocr_results.append({'bbox': (x1, y1, x3 - x1, y3 - y1), 'text': text})
    return ocr_results

def save_to_pdf(images, output_pdf_path):
    pdf_document = fitz.open()
    for img in images:
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        pdf_page = pdf_document.new_page(width=img.width, height=img.height)
        pdf_page.insert_image(pdf_page.rect, stream=img_byte_arr)
    pdf_document.save(output_pdf_path)

def pdf_to_word(pdf_path, word_path):
    doc = Document()
    pdf_document = fitz.open(pdf_path)
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text = page.get_text()
        doc.add_paragraph(text)
    doc.save(word_path)

def main(input_pdf, output_pdf, output_word):
    images = pdf_to_images(input_pdf)
    ocr_results = ocr_and_draw_bounding_boxes(images)
    save_to_pdf(images, output_pdf)
    pdf_to_word(output_pdf, output_word)

# Đường dẫn tới file PDF đầu vào và đầu ra
input_pdf = 'vanbanmulti.pdf'
output_pdf = 'vanbanmulti_after_OCR.pdf'
output_word = 'vanbanmulti_after_OCR.docx'

main(input_pdf, output_pdf, output_word)
