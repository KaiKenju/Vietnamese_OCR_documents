from bson.objectid import ObjectId
from pymongo import MongoClient
import gridfs
from PIL import Image
import io

client = MongoClient('mongodb://localhost:27017/')
db = client['OCRdata']
fs = gridfs.GridFS(db)

file_id = ObjectId('671f529a9ce1a948a58f6de8') # id object
# display file
def retrieve_file_from_mongodb(file_id, output_path):
    # Lấy tệp từ GridFS
    file_data = fs.get(file_id).read()
    
    # Lưu tệp vào hệ thống tệp cục bộ
    with open(output_path, 'wb') as output_file:
        output_file.write(file_data)
    print(f"File retrieved and saved to {output_path}")

# Ví dụ truy xuất tệp bằng ID
retrieve_file_from_mongodb(file_id, './mongodb_export/retrieved_file.pdf')  # file_id là ID tệp đã lưu trước đó
# or
retrieve_file_from_mongodb(file_id, './mongodb_export/retrieved_file.docx')  # file_id là ID tệp đã lưu trước đó