

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
from paddleocr import PaddleOCR
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image
import time
import concurrent.futures
from image_processing import (
    is_image_sharp, sharpen_image, preprocess_image, detect_table_edges,
    detect_table_corners, calculate_rotation_angle, process_and_save,
    convert_pdf_to_docx, rotate_image, deskew
)
from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")

class OCRProcessor:
    def __init__(self):
        self.ocr = PaddleOCR(lang='en', use_gpu=False)
        self.config = Cfg.load_config_from_name('vgg_transformer')
        self.config['weights'] = './weight/vgg_transformer.pth'
        # config = Cfg.load_config_from_file('./config/config_after_trainer.yml')
        # config['weights'] = './weight/transformerocr.pth'
        self.config['cnn']['pretrained'] = False
        self.config['device'] = 'cpu'
        self.detector = Predictor(self.config)
    
    def recognize_text(self, box, masked_img):
        cropped_image = masked_img[box[0][1]:box[1][1], box[0][0]:box[1][0]]
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        cropped_image = Image.fromarray(cropped_image)
        rec_result = self.detector.predict(cropped_image)
        return rec_result
    
    def process_image(self, img):
        img = preprocess_image(img)
        img = deskew(img)
        img = preprocess_image(img)
        detection_result = self.ocr.ocr(img, cls=False, det=True, rec=False)
        boxes = [[[[int(line[0][0]), int(line[0][1])], [int(line[2][0]), int(line[2][1])]] for line in detection_result[0]]][0]
        
        EXPAND = 5
        for box in boxes:
            box[0][0] -= EXPAND
            box[0][1] -= EXPAND
            box[1][0] += EXPAND
            box[1][1] += EXPAND
        
        masked_img, edges_on_white = detect_table_edges(img)
        return boxes, masked_img, img

ocr_processor = OCRProcessor()

@app.get("/")
def main(request: Request):
    return templates.TemplateResponse("c.html", {"request": request})

@app.post("/process_image/")
async def process_image(image: UploadFile = File(...)):
    try:
        start_time_total = time.time()
        
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        start_time_processing = time.time()
        boxes, masked_img, processed_img = ocr_processor.process_image(img)
        end_time_processing = time.time()
        execution_time_processing = end_time_processing - start_time_processing
        
        start_time_recognition = time.time()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = [executor.submit(ocr_processor.recognize_text, box, masked_img) for box in boxes]
        texts = [result.result() for result in results]
        end_time_recognition = time.time()
        execution_time_recognition = end_time_recognition - start_time_recognition

        c = canvas.Canvas("./folder_file_api/ocr_final2.pdf", pagesize=(processed_img.shape[1], processed_img.shape[0]))
        pdfmetrics.registerFont(TTFont('Times New Roman', 'times.ttf'))
        image_height = processed_img.shape[0]

        for text, box in zip(texts, boxes):
            x1, y1 = box[0]
            x2, y2 = box[1]
            x1 += 30
            x2 += 30
            y1 += 10
            y2 += 10
            
            y1, y2 = image_height - y1, image_height - y2
            c.setFont("Times New Roman", 11)
            c.drawString(x1, y1, text)
        c.save()
        pdf_file = './folder_file_api/ocr_final2.pdf'
        docx_file = './folder_file_api/ocr_final_word2.docx'
        convert_pdf_to_docx(pdf_file, docx_file)
        
        end_time_total = time.time()
        execution_time_total = end_time_total - start_time_total
    except Exception as e:
        import traceback; traceback.print_exc()

    result = {
        # "processing_time": execution_time_processing,
        # "recognition_time": execution_time_recognition,
        # "total_time": execution_time_total,
        "texts": texts
    }
    
    return JSONResponse(content=result)

@app.get("/download_file/")
async def download_file():
    file_path = "./folder_file_api/ocr_final_word2.docx"
    return FileResponse(file_path, media_type='application/octet-stream', filename='output.pdf')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000)
