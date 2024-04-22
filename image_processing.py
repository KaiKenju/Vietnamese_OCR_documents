import os
import cv2
import numpy as np
from pdf2docx import Converter

def is_image_sharp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return variance > 5000  # Ngưỡng có thể điều chỉnh tùy theo yêu cầu 6000


def sharpen_image(img):
    sharpen_kernel = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
    return cv2.filter2D(img, -1, sharpen_kernel)


def preprocess_image(img):
    if not is_image_sharp(img):
        print("Ảnh mờ, đang làm sắc nét...")
        img = sharpen_image(img)
    return img

# phát hiện cạnh/bảng của văn bản và mask văn bản


def detect_table_edges(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    img_dilate = cv2.dilate(thresh, kernel, iterations=1)
    img_erode = cv2.erode(img_dilate, kernel, iterations=1)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    detected_lines_horizontal = cv2.morphologyEx(
        img_erode, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    detected_lines_vertical = cv2.morphologyEx(
        img_erode, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    table_mask = cv2.addWeighted(
        detected_lines_horizontal, 0.5, detected_lines_vertical, 0.5, 0.0)
    edges_on_white = np.ones_like(img) * 255
    edges_on_white[table_mask == 0] = (255, 255, 255)
    edges_on_white[table_mask != 0] = (0, 0, 0)
    edges_on_white_gray = cv2.cvtColor(edges_on_white, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(edges_on_white_gray, 10, 255, cv2.THRESH_BINARY)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    return masked_img, edges_on_white

# Tính góc xoay cần điều chỉnh để làm thẳng ảnh
def calculate_rotation_angle(corners):
    if len(corners) < 2:
        return 0
    x1, y1 = corners[0]
    x2, y2 = corners[1]
    angle = np.arctan2(y2 - y1, x2 - x1)
    degrees = np.degrees(angle)
    if -80 > degrees > -90:
        # Nếu góc nhỏ hơn -90 độ, xoay về phía bên phải để làm thẳng đứng
        print("Góc xoay khi xoay:", degrees)
        degrees = 90+ degrees
       
    elif degrees > 80:
        # Nếu góc lớn hơn 90 độ, xoay về phía bên trái để làm thẳng đứng
        print("Góc xoay khi xoay:", degrees)
        degrees = degrees - 90
        
    
    return degrees

# phát hiện góc của bảng để xoay
def detect_table_corners(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 170, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                            threshold=100, minLineLength=100, maxLineGap=10)

    corners = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            angle = np.arctan2(y2 - y1, x2 - x1)
            angle_deg = np.degrees(angle)

            if abs(angle_deg) > 45:  # Vertical line
                corners.append((x1, y1))
                corners.append((x2, y2))
            else:  # Horizontal line
                corners.append((x1, y1))
                corners.append((x2, y2))

    rotation_angle = calculate_rotation_angle(
        corners)  # Calculate rotation angle here

    return corners, rotation_angle

def process_and_save(image, save_folder, count):
    # Create the save folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)
    
    # Save the cropped image to the specified folder
    cv2.imwrite(os.path.join(save_folder, f'cropped_image_{count}.jpg'), image)

def convert_pdf_to_docx(pdf_file, docx_file):
    cv = Converter(pdf_file)
    
    # Chuyển đổi PDF thành DOCX
    cv.convert(docx_file, start=0, end=None)
    
    # Đóng Converter
    cv.close()