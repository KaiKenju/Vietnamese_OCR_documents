🌎 English | [Vietnamese](README_vn.md)

## 🛠️ Setup

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
- Run the commands:
```[bash]
pip install -r requirements.txt
pip install reportlab
```

## ▶️ Run
* 🔥 if you want to understand how the system works, please run:
```[bash]
python main.py
```
* ✅ else, best performance:
```[bash]
python performance_thread.py
```
## 📝 Result
<table>
  <tr>
    <td><img src="assets/don-khoi-kien-vu-an-hanh-chinh-9418.png" alt="don-khoi-kien-vu-an-hanh-chinh-9418" style="width: 800px; height: 600px;"></td>
    <td><img src="ocr_final_image_with_boxes.jpg" alt="ocr_final_image_with_boxes" style="width: 800px; height: 600px;"></td>
  </tr>
</table>

## Video Demo
https://github.com/m1guelpf/readme-with-video/assets/94727276/2d434abb-7b71-4bca-9e0f-a3baad53a02c



## 🚀 Structure Project
```[bash]
Vietnamese_OCR_Documents/
          ├── assets/                   # contains image to OCR
          ├── config/                   #  configuration files and options for OCR system
          ├── cropped_images/           # Images are cropped for recognition purposes
          ├── folder_file_api/          # file pdf,word after OCR for web-app using Fast-api
          ├── weight/                   # The weight  of system
          ├── PaddleOCR/                # Paddle repositories
          ├── static/                   # front-end 
          ├── app.py                    # demo web(local-host) using FastAPI 
          ├── Core_OCR.ipynb            # notebook paddleOCR + vietOCR
          ├── image_processing.py       # image processing
          ├── main.py 
          ├── performance_thread.py     # performance optimization (faster main.py using thread)
          ├── Pretrained_vietOCR.ipynb  # training VietOCR
          ├── requirements.txt     
          ├── README.md                 # english version
          ├── README_vn.md              # vietnamese version
```
## 🚀 PaddleOCR
[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/tree/main) is an open source framework developed by Baidu PaddlePaddle to support the recognition and extraction of information from images. Initially, PaddleOCR supported recognition of English, Chinese, numbers, and processing of long texts. Currently, it has expanded its support to more languages ​​such as Japanese, Korean, German,... However, PaddleOCR does not currently support Vietnamese.

🌟 Features:

[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/main/README_en.md) support a variety of cutting-edge algorithms related to OCR, and developed industrial featured models/solution PP-OCR、 PP-Structure and PP-ChatOCR on this basis, and get through the whole process of data production, model training, compression, inference and deployment.

![image](https://github.com/KaiKenju/Vietnamese_OCR_documents/assets/94727276/75d28e4d-c8cd-4738-bd8e-8fb20643026a)

In the paddle's configuration file, [DB](https://arxiv.org/pdf/1911.08947) (Differentiable Binarization) is often used to detect text
![image](https://github.com/KaiKenju/Vietnamese_OCR_documents/assets/94727276/a59ae091-80e7-40e7-8ddb-0d7e52e91b07)

## 🚀 VietOCR

[VietOCR](https://github.com/pbcquoc/vietocr) is a combination of AttentionOCR and TransformerOCR

[AttentionOCR](https://arxiv.org/pdf/1706.03762)
![image](https://github.com/KaiKenju/Vietnamese_OCR_documents/assets/94727276/c1350449-14b0-4a8c-81fe-c1740e1a6880)

[TransformerOCR](https://pbcquoc.github.io/transformer/)
![image](https://github.com/KaiKenju/Vietnamese_OCR_documents/assets/94727276/83a37c72-b84e-400c-bd7c-289dafc91149)

[VietOCR](https://pbcquoc.github.io/vietocr/) library was built by me with the purpose of supporting you to use it to solve problems related to OCR in industry. The library provides both AtentionOCR and TransformerOCR architectures. Although the TransformerOCR architecture works quite well in NLP, in my opinion, the accuracy does not have a significant improvement compared to AttentionOCR and the prediction time is much slower.