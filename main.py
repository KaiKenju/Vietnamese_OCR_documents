from paddleocr import PaddleOCR, draw_ocr
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

import cv2
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from image_processing import is_image_sharp, sharpen_image, preprocess_image, detect_table_edges, detect_table_corners, calculate_rotation_angle, process_and_save

# Initialize PaddleOCR
ocr = PaddleOCR(lang='en')

# Specifying output path and font path.
out_path = './output'
font = './simfang.ttf'

# Initialize VietOCR
# use vgg_seq2seq
# config = Cfg.load_config_from_name('vgg_vgg_seq2seq.pth')
# config['weights'] = 'https://vocr.vn/data/vietocr/vgg_seq2seq.pth'
# # config['pretrain'] = 'https://vocr.vn/data/vietocr/vgg_seq2seq.pth'
# config['cnn']['pretrained']=False
config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = 'https://vocr.vn/data/vietocr/vgg_transformer.pth'
config['cnn']['pretrained'] = False
config['device'] = 'cpu' #comment this line if you want to use GPU

detector = Predictor(config)

# image path
img_path = './assets/rotation_thuvien.png'

img_ori = cv2.imread(img_path)

img = preprocess_image(img_ori)

corners, rotation_angle  = detect_table_corners(img)
print("Góc xoay của ảnh là:", rotation_angle, "độ")
# Xoay ảnh
if rotation_angle != 0:
    center = (img.shape[1] // 2, img.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    img = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))

# Thực hiện phát hiện vùng quan tâm bằng PaddleOCR
detection_result = ocr.ocr(img, cls=False, det=True, rec=False)
boxes = [[[[int(line[0][0]), int(line[0][1])], [int(line[2][0]), int(line[2][1])]] for line in detection_result[0]]][0]

# change bounding box size
EXPAND = 5
for box in boxes:
    box[0][0] -= EXPAND
    box[0][1] -= EXPAND
    box[1][0] += EXPAND
    box[1][1] += EXPAND

masked_img, edges_on_white  = detect_table_edges(img) # edge recognition


texts = []
for box in boxes:
    cropped_image = masked_img[box[0][1]:box[1][1], box[0][0]:box[1][0]]
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    cropped_image = Image.fromarray(cropped_image)
    rec_result = detector.predict(cropped_image)
    texts.append(rec_result)

plt.figure(figsize=(6, 5))
plt.imshow(cv2.cvtColor(edges_on_white, cv2.COLOR_BGR2RGB))
plt.title('Image (rotation, Blur, edge)')
plt.axis('off')
plt.savefig('edge_image.jpg', bbox_inches='tight', pad_inches=0,dpi=300)


plt.figure(figsize=(6, 5))
plt.imshow(cv2.cvtColor(edges_on_white, cv2.COLOR_BGR2RGB))
plt.title('OCR Final')


#add text & bbox in image
for text, box in zip(texts, boxes):
    x1, y1 = box[0]
    x2, y2 = box[1]
    x1 += 5  # shift right 5 pixel
    x2 += 5  # shift right 5 pixel
    y1 += 10  # shift down 10 pixel 
    y2 += 10
    plt.text(x1, y1, text, ha='left', va='top', wrap=True, linespacing=0.75, fontdict={'family': 'serif', 'size': 7})

plt.axis('off')
plt.tight_layout()
plt.savefig('ocr_final_image_with_boxes.jpg', bbox_inches='tight', pad_inches=0,dpi=300)

#Post-processing, analysis 
texts_1 = texts
# Specify the folder where you want to save the cropped images
save_folder = 'cropped_images'

# Ensure the folder exists
os.makedirs(save_folder, exist_ok=True)

# Process and save each cropped image
count = 0
for box in reversed(boxes):
    cropped_image = img[box[0][1]:box[1][1], box[0][0]:box[1][0]]
    process_and_save(cropped_image, save_folder, count)
    count += 1

output = ""
for text in reversed(texts):
    output += text + " "

print(output)
plt.show()
