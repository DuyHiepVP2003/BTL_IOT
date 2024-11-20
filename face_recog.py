import cv2
import numpy as np
import requests
import time

# Đường dẫn đến ảnh đầu vào
image_path = 'D:/code/iiot/image/captured_image2.jpg'

# Tải và chuyển ảnh đầu vào sang thang độ xám
input_image = cv2.imread(image_path)
input_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# Khởi tạo bộ nhận diện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Phát hiện khuôn mặt trong ảnh đầu vào
input_faces = face_cascade.detectMultiScale(input_gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

# Kiểm tra có khuôn mặt trong ảnh đầu vào không
if len(input_faces) == 0:
    print("Không tìm thấy khuôn mặt trong ảnh đầu vào.")
    exit()

# Cắt khuôn mặt đầu tiên được phát hiện trong ảnh đầu vào
(x, y, w, h) = input_faces[0]
input_face = input_gray[y:y+h, x:x+w]

# URL của ESP32-CAM
url = 'http://192.168.2.109/cam-lo.jpg'

# URL gửi yêu cầu mở cửa
handle_url = 'http://localhost:3000/handle'
while True:
    try:
        # Tải ảnh từ ESP32-CAM
        response = requests.get(url, timeout=5)  # Thêm timeout
        response.raise_for_status()  # Kiểm tra mã trạng thái
        frame = np.asarray(bytearray(response.content), dtype=np.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        # Chuyển ảnh sang thang độ xám
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Phát hiện khuôn mặt trong ảnh từ ESP32-CAM
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            # Cắt khuôn mặt từ ảnh ESP32-CAM
            face = gray_frame[y:y+h, x:x+w]

            # Chuyển đổi kích thước khuôn mặt ESP32-CAM để khớp với kích thước khuôn mặt từ ảnh đầu vào
            resized_face = cv2.resize(face, (input_face.shape[1], input_face.shape[0]))

            # So sánh hai khuôn mặt bằng cách tính toán sự khác biệt
            difference = cv2.absdiff(input_face, resized_face)
            result = np.sum(difference) / (input_face.shape[0] * input_face.shape[1])

            # Ngưỡng để xác định xem khuôn mặt có khớp không
            threshold = 1000  # Tùy chỉnh ngưỡng này dựa trên độ chính xác mong muốn

            if result < threshold:
                # Nếu đúng người, vẽ khung màu xanh lá cây và in ra thông báo
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Match", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                print("Phát hiện khuôn mặt: Khuôn mặt phù hợp!")

                # Gửi request mở cửa
                try:
                    response = requests.get("http://192.168.2.110/LED=ON", timeout=5)  # Thêm timeout để tránh treo chương trình
                    if response.status_code == 200:
                        print("Request thành công!")
                        print("Nội dung phản hồi:", response.text)
                    else:
                        print(f"Lỗi: {response.status_code}")
                except requests.exceptions.RequestException as e:
                        print(f"Xảy ra lỗi khi gửi request: {e}")

            else:
                # Nếu không đúng, vẽ khung màu đỏ và in ra thông báo
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Not Match", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                print("Phát hiện khuôn mặt: Khuôn mặt không phù hợp!")

        # Hiển thị khung hình
        cv2.imshow('Face Recognition', frame)
        # time.sleep(0.5)
        # Nhấn 'p' để chụp ảnh và lưu
        if cv2.waitKey(1) & 0xFF == ord('p'):
            cv2.imwrite('D:/code/iiot/image/captured_frame.jpg', frame)
            print("Ảnh đã được lưu vào D:/code/iiot/image/captured_frame.jpg")

    except (requests.exceptions.RequestException, ValueError) as e:
        print(f"Không thể tải ảnh từ ESP32-CAM: {e}")
        time.sleep(1)  # Chờ một chút trước khi thử lại

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng bộ nhớ và đóng cửa sổ
cv2.destroyAllWindows()
