from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse
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
from image_processing import is_image_sharp, sharpen_image, preprocess_image, detect_table_edges, detect_table_corners, calculate_rotation_angle, process_and_save, convert_pdf_to_docx, rotate_image, deskew
from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
app = FastAPI()
templates = Jinja2Templates(directory="static")

# Khởi tạo PaddleOCR và VietOCR

ocr = PaddleOCR(lang='en', use_gpu=False)


config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = './weight/vgg_transformer.pth'
# config = Cfg.load_config_from_file('./config/config_after_trainer.yml')
# config['weights'] = './weight/transformerocr.pth'
config['cnn']['pretrained'] = False
config['device'] = 'cpu'
detector = Predictor(config)

# Hàm nhận diện văn bản từ một phần của ảnh
def recognize_text(box, masked_img):
    cropped_image = masked_img[box[0][1]:box[1][1], box[0][0]:box[1][0]]
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    cropped_image = Image.fromarray(cropped_image)
    rec_result = detector.predict(cropped_image)
    return rec_result


@app.get("/")
def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Route POST để xử lý ảnh và nhận diện văn bản
@app.post("/process_image/")
async def process_image(image: UploadFile = File(...)):
    print(image.filename)
    try:
        start_time_total = time.time()

        # Đọc ảnh từ request
        contents = await image.read()
        
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        start_time_processing = time.time()
        img = preprocess_image(img)
        img = deskew(img)
        img = preprocess_image(img)

        detection_result = ocr.ocr(img, cls=False, det=True, rec=False)
        boxes = [[[[int(line[0][0]), int(line[0][1])], [int(line[2][0]), int(line[2][1])]] for line in detection_result[0]]][0]
        # Mở rộng kích thước của bounding box
        EXPAND = 5
        for box in boxes:
            box[0][0] -= EXPAND
            box[0][1] -= EXPAND
            box[1][0] += EXPAND
            box[1][1] += EXPAND

        masked_img, edges_on_white = detect_table_edges(img)

        end_time_processing = time.time()
        execution_time_processing = end_time_processing - start_time_processing

        start_time_recognition = time.time()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = [executor.submit(recognize_text, box, masked_img) for box in boxes]
        texts = [result.result() for result in results]
        end_time_recognition = time.time()
        execution_time_recognition = end_time_recognition - start_time_recognition

        
        c = canvas.Canvas("./folder_file_api/ocr_final2.pdf", pagesize=(img.shape[1], img.shape[0]))
        pdfmetrics.registerFont(TTFont('Times New Roman', 'times.ttf'))
        image_height = img.shape[0]

        for text, box in zip(texts, boxes):
            x1, y1 = box[0]
            x2, y2 = box[1]
            x1 += 30  # dịch phải 5 pixel
            x2 += 30  # dịch phải 5 pixel
            y1 += 10  # dịch xuống 10 pixel 
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
        import traceback; traceback.print_exc();
    
    result = {
        # "processing_time": execution_time_processing,
        # "recognition_time": execution_time_recognition,
        # "total_time": execution_time_total,
        "texts": texts
    }
    
    return JSONResponse(content=result)
from fastapi.responses import FileResponse
@app.get("/download_file/")
async def download_file():
    # Đường dẫn đến tệp cần tải xuống
    file_path = "./folder_file_api/ocr_final2.pdf"
    # Trả về phản hồi FileResponse với đường dẫn đến tệp
    return FileResponse(file_path, media_type='application/octet-stream', filename='output.pdf')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000)

