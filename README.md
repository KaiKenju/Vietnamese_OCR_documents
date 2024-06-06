# OCR-CORE

## Cài Đặt

- Clone  this project:

```[bash]
git clone https://github.com/KaiKenju/Vietnamese_OCR_documents
```

- Initial enviromment with Miniconda:

```[bash]
conda create -n <env_name> python=3.8
```
- Activate conda
```[bash]
conda activate <env_name> 
cd Vietnamese_OCR_documents
```

Chạy lần lượt các lệnh:

```[bash]
pip install -r requirements.txt
pip install reportlab
<!-- cd PaddleOCR
pip install -e . -->
```

## Khởi chạy
* if you want to understand how the system works, please run:
```[bash]
python main.py
```
* else, best performance:
```[bash]
python performance_thread.py
```
## Kết quả
<table>
  <tr>
    <td><img src="assets/don-khoi-kien-vu-an-hanh-chinh-9418.png" alt="don-khoi-kien-vu-an-hanh-chinh-9418" style="width: 800px; height: 600px;"></td>
    <td><img src="ocr_final_image_with_boxes.jpg" alt="ocr_final_image_with_boxes" style="width: 800px; height: 600px;"></td>
  </tr>
</table>


