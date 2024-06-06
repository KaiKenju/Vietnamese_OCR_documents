import os
import cv2
import numpy as np
from pdf2docx import Converter

def brightness(image):
    # Chuyển đổi hình ảnh sang không gian màu YUV
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    # Trích xuất kênh Y (Luminance) từ không gian màu YUV
    luminance = yuv_image[:,:,0]

    # Tính toán giá trị trung bình của kênh Y
    mean_brightness = luminance.mean()

    return mean_brightness

def enhance_brightness(image, target_brightness, increase_factor=1.2):
    current_brightness = brightness(image)
    if current_brightness < target_brightness:
        # Tăng cường độ sáng bằng cách nhân với hệ số tăng
        adjusted_image = cv2.convertScaleAbs(image, alpha=increase_factor, beta=0)
        print("đã tăng cường độ sáng...")
        return adjusted_image
    else:
        return image
def is_image_sharp(image):
    '''
    - Checks if the image is sharp by calculating the variance of Laplacian.
    - Converts the image to grayscale, applies Laplacian filter,
    and calculates the variance. Returns True if variance is greater than 500,
    indicating sharpness. Threshold can be adjusted as needed.
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return variance > 100  # Ngưỡng có thể điều chỉnh tùy theo yêu cầu 6000
def blur_image(img):
    '''
    - Blurs the input image using Gaussian blur.
    - Utilizes a 3x3 kernel for Gaussian blur.
    '''
    blurred_image = cv2.GaussianBlur(img, (3, 3), 0)  # Sử dụng kernel 5x5 cho bộ lọc Gaussian
    return blurred_image

def sharpen_image(img):
    '''
    - Sharpens the input image using a specific sharpening kernel.
    - Applies a 3x3 sharpening kernel which enhances edges.
    '''
    sharpen_kernel = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
    return cv2.filter2D(img, -1, sharpen_kernel)


def preprocess_image(img):
    '''
    Combination 2 function above
    '''
    if not is_image_sharp(img):
        print("Ảnh mờ, đang làm sắc nét...")
        img = sharpen_image(img)
        print("Đang làm mờ ảnh để dễ nhìn hơn...")
        img = blur_image(img)
    return img


def detect_table_edges(img):
    '''
    - Detects edges of tables in the input image using morphological operations.
    - Converts the image to grayscale, applies morphological transformations to detect horizontal and vertical lines,
    combines the results, and creates a mask. Returns the masked image with detected table edges.
    '''
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

def calculate_rotation_angle(corners):
    if len(corners) < 1:
        return 0  # Nếu không đủ điểm góc, không xoay ảnh
    angles = []
    for i in range(0, len(corners), 2):
        x1, y1 = corners[i]
        x2, y2 = corners[i + 1]
        angle = np.arctan2(y2 - y1, x2 - x1)
        degrees = np.degrees(angle)
        if abs(degrees) > 10:  # Chỉ tính các góc lớn hơn 10 độ
            angles.append(degrees)
    if not angles:
        return 0
    average_angle = sum(angles) / len(angles)
    if abs(average_angle) > 45:
        average_angle = (90 + average_angle) if average_angle < 0 else (average_angle - 90)
    else: 
        average_angle = 0 
    return average_angle

def detect_table_corners(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=10)
    corners = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            corners.extend([(x1, y1), (x2, y2)])
    return corners, calculate_rotation_angle(corners)

def rotate_image(img, angle):
    if angle == 0:
        return img
    center = (img.shape[1] // 2, img.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))

def deskew(img):
    '''
    - Deskews the input image to straighten any skewed text or objects.
    - Converts the image to grayscale, detects edges using Canny edge detector,
    performs Hough transform to detect lines, calculates the median angle from the detected lines,
    and rotates the image to align it with the median angle.
    - Returns the deskewed image.
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        angles.append(angle)
    
    median_angle = np.median(angles)
    

    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), median_angle, 1)
    img_rotated = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
    print(f"Ảnh đã được xoay {median_angle:.2f} độ để thẳng đứng.")

    return img_rotated

def process_and_save(image, save_folder, count):
    '''
    - Processes the image and saves it to the specified folder with a given count.
    - Creates the save folder if it doesn't exist, then saves the processed image with a unique filename.
    '''
    os.makedirs(save_folder, exist_ok=True)
    cv2.imwrite(os.path.join(save_folder, f'cropped_image_{count}.jpg'), image)

def convert_pdf_to_docx(pdf_file, docx_file):
    '''
    - Converts a PDF file to a DOCX file using the `pdf2docx` library.
    - Initializes a converter, converts the PDF file to DOCX, and closes the converter.
    '''
    cv = Converter(pdf_file)
    cv.convert(docx_file, start=0, end=None)
    cv.close()

