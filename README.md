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

Chạy lần lượt các lệnh:

```[bash]
pip install -r requirements.txt
cd PaddleOCR
pip install -e .
```

## Khởi chạy

```[bash]
python main.py
```
## Kết quả
<table>
  <tr>
    <td><img src="assets/don-khoi-kien-vu-an-hanh-chinh-9418.png" alt="don-khoi-kien-vu-an-hanh-chinh-9418" style="width: 50%;"></td>
    <td><img src="ocr_final_image_with_boxes.jpg" alt="ocr_final_image_with_boxes" style="width: 50%;"></td>
  </tr>
</table>

