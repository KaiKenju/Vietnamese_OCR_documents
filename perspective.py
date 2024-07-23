# import cv2
# import numpy as np


# def biggest_contour(contours):
#     biggest = np.array([])
#     max_area = 0
#     for i in contours:
#         area = cv2.contourArea(i)
#         if area > 1000:
#             peri = cv2.arcLength(i, True)
#             approx = cv2.approxPolyDP(i, 0.015 * peri, True)
#             if area > max_area and len(approx) == 4:
#                 biggest = approx
#                 max_area = area
#     return biggest

# img = cv2.imread('assets/dewarp4marked.png')
# img_original = img.copy()

# # Image modification
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = cv2.bilateralFilter(gray, 20, 30, 30)
# edged = cv2.Canny(gray, 10, 20)

# # Contour detection
# contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours = sorted(contours, key=cv2.contourArea, reverse=True)[:300]

# biggest = biggest_contour(contours)


# cv2.drawContours(img, [biggest], -1, (0, 255, 0), 3)

# # Pixel values in the original image
# points = biggest.reshape(4, 2)
# input_points = np.zeros((4, 2), dtype="float32")

# points_sum = points.sum(axis=1)
# input_points[0] = points[np.argmin(points_sum)]
# input_points[3] = points[np.argmax(points_sum)]

# points_diff = np.diff(points, axis=1)
# input_points[1] = points[np.argmin(points_diff)]
# input_points[2] = points[np.argmax(points_diff)]

# (top_left, top_right, bottom_right, bottom_left) = input_points
# bottom_width = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
# top_width = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
# right_height = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))
# left_height = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))

# # Output image size
# max_width = max(int(bottom_width), int(top_width))
# # max_height = max(int(right_height), int(left_height))
# max_height = int(max_width * 1.414)  # for A4

# # Desired points values in the output image
# converted_points = np.float32([[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]])

# # Perspective transformation
# matrix = cv2.getPerspectiveTransform(input_points, converted_points)
# img_output = cv2.warpPerspective(img_original, matrix, (max_width, max_height))

# # Image shape modification for hstack
# gray = np.stack((gray,) * 3, axis=-1)
# edged = np.stack((edged,) * 3, axis=-1)

# # img_hor = np.hstack((img_original, gray, edged, img))
# # cv2.imshow("Contour detection", img_hor)
# cv2.imshow("Warped perspective", img_output)

# # cv2.imwrite('output/document.jpg', img_output)

# cv2.waitKey(0)


# import cv2
# import numpy as np



# img = cv2.imread('./assets/dewarp4marked.png')
# img_original = img.copy()



# # Image modification
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = cv2.bilateralFilter(gray, 20, 30, 30)
# edged = cv2.Canny(gray, 10, 20)

# # Contour detection
# contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]


# for i in contours:
#     cv2.drawContours(img, [i], -1, (0,255,0), 3)




# # Image shape modification for hstack
# gray = np.stack((gray,) * 3, axis=-1)
# edged = np.stack((edged,) * 3, axis=-1)

# img_hor = np.hstack((img_original, gray, edged, img))
# cv2.imshow("Contour detection", img_hor)
# # cv2.imshow("Warped perspective", img_output)

# # cv2.imwrite('output/document.jpg', img_output)

# cv2.waitKey(0)

# import cv2
# import numpy as np

# # Đọc hình ảnh
# img = cv2.imread('assets/image_test_perspective.jpg')
# img_original = img.copy()

# # Chuyển đổi sang ảnh xám và tách biên
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = cv2.bilateralFilter(gray, 20, 30, 30)
# edged = cv2.Canny(gray, 10, 20)

# # Phát hiện các đường thẳng bằng phương pháp Hough
# lines = cv2.HoughLinesP(edged, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

# # Tạo một mảng numpy chứa các điểm 2D hợp lệ
# points = []
# for line in lines:
#     x1, y1, x2, y2 = line[0]
#     points.append((x1, y1))
#     points.append((x2, y2))
# points = np.array(points, dtype=np.int32)

# # Tìm hình chữ nhật bao quanh văn bản
# rect = cv2.minAreaRect(points)
# box = cv2.boxPoints(rect)
# box = np.int0(box)

# # Tạo src_pts và dst_pts từ hình chữ nhật
# src_pts = box.astype("float32")
# dst_pts = np.array([[0, 0], [img.shape[1], 0], [img.shape[1], img.shape[0]], [0, img.shape[0]]], dtype="float32")

# # Tính toán ma trận biến đổi perspective
# M = cv2.getPerspectiveTransform(src_pts, dst_pts)

# # Áp dụng biến đổi perspective lên hình ảnh gốc
# img_output = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))

# # Hiển thị hình ảnh
# cv2.imshow("Original Image", img_original)
# cv2.imshow("Warped Perspective", img_output)

# # Lưu hình ảnh đầu ra (nếu cần)
# # cv2.imwrite('output/document.jpg', img_output)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #---------------------------------------------------------------
# import cv2
# import numpy as np

# def biggest_contour(contours):
#     biggest = np.array([])
#     max_area = 0
#     for i in contours:
#         area = cv2.contourArea(i)
#         if area > 1000:
#             peri = cv2.arcLength(i, True)
#             approx = cv2.approxPolyDP(i, 0.015 * peri, True)
#             if area > max_area and len(approx) >= 3:
#                 biggest = approx
#                 max_area = area
#     return biggest, max_area

# def process_image(img):
#     img_original = img.copy()

#     # Image modification
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray = cv2.bilateralFilter(gray, 20, 30, 30)
#     edged = cv2.Canny(gray, 10, 20)

#     # Contour detection
#     contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)[:300]
#     biggest, max_area = biggest_contour(contours)
    
#     if len(biggest) >= 4:  # Nếu có đủ 4 góc
#         cv2.drawContours(img, [biggest], -1, (0, 255, 0), 3)

#         # Xác định tọa độ 4 điểm của văn bản
#         points = biggest.reshape(4, 2)
#         input_points = np.zeros((4, 2), dtype="float32")
#         points_sum = points.sum(axis=1)
#         input_points[0] = points[np.argmin(points_sum)]
#         input_points[3] = points[np.argmax(points_sum)]
#         points_diff = np.diff(points, axis=1)
#         input_points[1] = points[np.argmin(points_diff)]
#         input_points[2] = points[np.argmax(points_diff)]

#         # Xác định kích thước đầu ra
#         (top_left, top_right, bottom_right, bottom_left) = input_points
#         bottom_width = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
#         top_width = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
#         max_width = max(int(bottom_width), int(top_width))
#         max_height = int(max_width * 1.414)  # for A4

#         # Tạo tọa độ đầu ra
#         converted_points = np.float32([[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]])

#         # Tính toán ma trận biến đổi perspective
#         matrix = cv2.getPerspectiveTransform(input_points, converted_points)
#         img_output = cv2.warpPerspective(img_original, matrix, (max_width, max_height))

#         # Hiển thị hình ảnh đầu ra
#         cv2.imshow("Warped Perspective", img_output)
#         cv2.imwrite('assets/after_perspective.png', img_output)
    

#     else:  # Nếu không đủ 4 góc
#         # Tạo hình chữ nhật gần đúng nhất từ các điểm có sẵn
#         rect = cv2.minAreaRect(biggest)
#         box = cv2.boxPoints(rect)
#         box = np.int0(box)
#         cv2.drawContours(img, [box], -1, (0, 255, 0), 3)

#         # Tạo src_pts và dst_pts từ hình chữ nhật
#         src_pts = box.astype("float32")
#         dst_pts = np.array([[0, 0], [img.shape[1], 0], [img.shape[1], img.shape[0]], [0, img.shape[0]]], dtype="float32")

#         # Tính toán ma trận biến đổi perspective
#         M = cv2.getPerspectiveTransform(src_pts, dst_pts)

#         # Áp dụng biến đổi perspective lên hình ảnh gốc
#         img_output = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))

#         cv2.imshow("Warped Perspective", img_output)

#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     img = cv2.imread('assets/perspective_done.png')
#     process_image(img)
#---------------------------------------------------------------

# from transformers import pipeline

# corrector = pipeline("text2text-generation", model="bmd1905/vietnamese-correction")
# # Example
# MAX_LENGTH = 512

# # Define the text samples
# texts = [
#     "Bộ Lao động - Thương binh xã hội về việc hoán đồi ngày nghi, ngày đi làm bù các dịp",
#     " nghi lê, Tết ",
#     "Tư pháp thông báo việc nghi lễ như sau",
#     "thực hiện nghi lề Giỗ Tổ Hùng Vương, ",
#     "sau ngày nghí ",
#     "lề; kịp thời báo cáo những vấn đề bắt thường phát sinh trong thời gian nghi lễ"
# ]

# # Batch prediction
# predictions = corrector(texts, max_length=MAX_LENGTH)

# # Print predictions
# for text, pred in zip(texts, predictions):
#     print("- " + pred['generated_text'])

import requests
import json

def correct_text(text):
    url = "https://api.languagetool.org/v2/check"
    params = {
        'text': text,
        'language': 'vi'
    }
    try:
        response = requests.post(url, data=params)
        response.raise_for_status()  # Kiểm tra xem yêu cầu có thành công không

        # In nội dung phản hồi để kiểm tra
        print("Response content:", response.content)
        
        result = response.json()     # Chuyển đổi phản hồi thành JSON

        corrected_text = text
        offset_correction = 0  # Để điều chỉnh vị trí sau khi thay thế
        for match in result['matches']:
            replacements = match['replacements']
            if replacements:
                replacement = replacements[0]['value']
                offset = match['offset'] + offset_correction
                length = match['length']
                corrected_text = corrected_text[:offset] + replacement + corrected_text[offset+length:]
                offset_correction += len(replacement) - length

        return corrected_text
    except requests.exceptions.RequestException as e:
        print(f"Error in request: {e}")
    except json.decoder.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")

    return text

# Văn bản nhận dạng từ VietOCR
text_from_vietocr = " nghilê, Tết "

# Sửa lỗi chính tả với LanguageTool API
corrected_text = correct_text(text_from_vietocr)

print("Corrected text:", corrected_text)
