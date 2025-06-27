import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import easyocr

def vietnamese_ocr_and_overlay(image_path):
    # Kiểm tra file tồn tại
    if not os.path.exists(image_path):
        print(f"File không tồn tại: {image_path}")
        return

    # Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        print(f"Không thể đọc ảnh từ đường dẫn: {image_path}")
        return

    # Tiền xử lý ảnh cho chữ viết tay
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale_percent = 200  # Phóng to 200%
    width = int(gray.shape[1] * scale_percent / 100)
    height = int(gray.shape[0] * scale_percent / 100)
    gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LINEAR)
    gray = cv2.equalizeHist(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    thresh = cv2.medianBlur(thresh, 3)
    img_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

    # Lưu ảnh tiền xử lý để kiểm tra
    cv2.imwrite('preprocessed_image.jpg', img_rgb)
    print(f"Đã lưu ảnh tiền xử lý tại: {os.path.abspath('preprocessed_image.jpg')}")

    # Khởi tạo EasyOCR cho tiếng Việt
    reader = easyocr.Reader(['vi'], gpu=False)  # Set gpu=True nếu có GPU

    # Nhận diện văn bản
    results = reader.readtext(img_rgb)

    # In số vùng văn bản phát hiện
    print(f"Đã phát hiện {len(results)} vùng văn bản.")

    # Chuyển ảnh gốc sang PIL để vẽ chữ
    overlay_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(overlay_img)

    # Tải font tiếng Việt
    try:
        font_path = "C:/Windows/Fonts/times.ttf"
        font = ImageFont.truetype(font_path, 20)
        print(f"Đã tải font từ: {font_path}")
    except Exception as e:
        print(f"Không thể tải font: {e}. Dùng font mặc định.")
        font = ImageFont.load_default()

    for (bbox, text, prob) in results:
        # bbox là danh sách 4 điểm: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        top_left = bbox[0]
        bottom_right = bbox[2]
        # Điều chỉnh tọa độ về kích thước ảnh gốc
        x = int(top_left[0] * 100 / scale_percent)
        y = int(top_left[1] * 100 / scale_percent)
        w = int((bottom_right[0] - top_left[0]) * 100 / scale_percent)
        h = int((bottom_right[1] - top_left[1]) * 100 / scale_percent)
        conf = int(prob * 100)  # Chuyển đổi xác suất thành phần trăm
        print(f"Văn bản='{text}', Độ tin cậy={conf}%, Vị trí=({x}, {y}), Kích thước=({w}x{h})")

        if conf > 30 and text:  # Ngưỡng tin cậy 30%
            # Vẽ hình chữ nhật trắng để che chữ cũ
            draw.rectangle([x, y, x + w, y + h], fill=(255, 255, 255))

            # Tính toán vị trí để căn giữa văn bản
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_x = x + (w - text_width) // 2
            text_y = y + (h - text_height) // 2

            # Ghi đè văn bản mới (màu đen)
            draw.text((text_x, text_y), text, font=font, fill=(0, 0, 0))

    # Chuyển ảnh từ PIL về OpenCV
    overlay_img_cv = cv2.cvtColor(np.array(overlay_img), cv2.COLOR_RGB2BGR)

    # Lưu ảnh kết quả
    output_path = 'output_image.jpg'
    cv2.imwrite(output_path, overlay_img_cv)
    print(f"Đã lưu ảnh tại: {os.path.abspath(output_path)}")

    # Hiển thị ảnh
    cv2.imshow('Result', overlay_img_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Sử dụng hàm
image_path = r'img.png'
vietnamese_ocr_and_overlay(image_path)