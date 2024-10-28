# DISPLAY IMAGE FROM MONGODB
from bson.objectid import ObjectId
from pymongo import MongoClient
import gridfs
from PIL import Image
import io

client = MongoClient('mongodb://localhost:27017/')
db = client['OCRdata']
fs = gridfs.GridFS(db)

def get_image_from_mongodb(img_id):
    try:
        img_data = fs.get(img_id).read()  # Lấy dữ liệu hình ảnh
        image = Image.open(io.BytesIO(img_data))  # Mở hình ảnh bằng PIL
        return image
    except gridfs.errors.NoFile:
        print(f"No file found with ID: {img_id}")

# Sử dụng ID của hình ảnh

image_id = ObjectId('671f5047fb910eef2fae9d71')  # Chuyển đổi chuỗi thành ObjectId
image = get_image_from_mongodb(image_id)
if image:
    image.show()  # Hiển thị hình ảnh