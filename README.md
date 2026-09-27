# Công cụ xử lý ảnh chương 9

Ứng dụng Python có giao diện desktop, cho phép chọn ảnh cá nhân, chạy K-means, Spectral Clustering, Haar cascade, HOG và HOG + SVM, xem ảnh trước/sau và xuất kết quả PNG cùng tham số.

## Chạy trên máy này

Nhấp đúp `Mo_cong_cu.cmd`. Chọn **Mở ảnh** hoặc **Dùng ảnh mẫu**, chọn chức năng và tham số, bấm **Xử lý ảnh**, sau đó **Lưu kết quả PNG**.

## Cài đặt trên máy khác

Cần Python 3.9–3.12 có Tkinter. Mở terminal trong thư mục này:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Không sao chép thư mục `.venv` sang máy khác; tạo lại bằng các lệnh trên.

## Chức năng

| Chức năng | Cơ sở chương 9 | Kết quả |
|---|---|---|
| Giảm số màu | Mục 2.1.2: K-means trên vector RGB | Thay mỗi điểm ảnh bằng màu tâm cụm |
| Phân đoạn theo màu | Mục 2.1.3: phân cụm RGB | Tô màu minh họa cho từng nhãn cụm |
| Phân đoạn màu + vị trí | Mục 2.1.3: vector (R,G,B,αy/H,αx/W) | Các cụm chịu ảnh hưởng của cả màu và vị trí |
| So sánh K-means và Spectral | Đồ thị 12 láng giềng, ảnh 64×64 | Hai kết quả dùng cùng đặc trưng |
| Phát hiện mặt và mắt | Haar cascade đã huấn luyện sẵn trong OpenCV | Các khung vùng mặt và mắt |
| So sánh tham số phát hiện mặt | Bốn tổ hợp scaleFactor và minNeighbors | Bốn ảnh kết quả |
| Hiển thị HOG | 9 hướng, khối 2×2 ô, L2-Hys | Hình gradient và số chiều đặc trưng |
| HOG + SVM và NMS | Huấn luyện dữ liệu mẫu và hard negative mining | Hộp phát hiện trước/sau NMS trên ảnh riêng |

Mã K-means nằm trong `processing.py`; các thí nghiệm và xuất ảnh nằm trong `workflows.py`; mô hình HOG + SVM nằm trong `hog_detector.py`; giao diện nằm trong `app.py`.
Các thay đổi so với mã minh họa trong báo cáo: đọc ảnh người dùng thay cho `data.astronaut()`, xử lý hướng EXIF, giới hạn kích thước để chạy trên CPU, lấy mẫu ngẫu nhiên tối đa 15.000 điểm để huấn luyện và dự đoán nhãn cho mọi điểm ảnh theo từng lô. Dùng `KMeans`, `n_init=4`, `random_state=42` để kết quả ổn định giữa các lần chạy trong cùng môi trường.

## Gợi ý trình bày với thầy

1. Mở cùng một ảnh cá nhân, thử giảm màu với K=2, 4, 8, 16 và so sánh mức chi tiết.
2. Chọn Phân đoạn màu và vị trí. So sánh hai kết quả RGB và RGB kết hợp tọa độ.
3. Giữ K cố định, thử trọng số 0, 0.6, 1.5 để quan sát ảnh hưởng của vị trí.
4. Lưu các ảnh kết quả để đưa vào báo cáo thực nghiệm. Ghi rõ K, trọng số và kích thước xử lý.

## Dùng HOG + SVM và NMS

Chọn ảnh chân dung chính diện, đủ sáng. Chọn **Phát hiện HOG + SVM và NMS**, giữ mặt tối thiểu 60 px, ngưỡng SVM 0,6 và IoU 0,25 để thử trước. Lần đầu mỗi phiên app sẽ huấn luyện mô hình từ dữ liệu mẫu, mất nhiều thời gian hơn các lần sau. Ảnh của bạn không được dùng để huấn luyện.

Nếu mặt nhỏ, giảm kích thước tối thiểu về 25–40 px. Giảm ngưỡng SVM để nhận thêm ứng viên, nhưng có thể tăng báo nhầm. Giảm IoU để loại mạnh hơn các hộp chồng lấn. Điểm SVM không phải phần trăm xác suất; NMS không bảo đảm loại được các vùng báo nhầm.

## Xuất kết quả

**Lưu kết quả PNG** lưu ảnh ghép một hàng ngang. **Xuất kết quả & tham số** tạo thư mục riêng gồm `so_sanh.png`, ảnh thành phần, `metadata.json` và `Mo_ta_ket_qua.txt`. Nên lưu vào thư mục `outputs` để không đưa kết quả thử nghiệm lên Git.

## Phạm vi và giới hạn

- Công cụ nhận một ảnh người dùng để xử lý. HOG+SVM tự huấn luyện bằng dữ liệu mẫu khi dùng lần đầu mỗi phiên app. Chưa có PCA/Eigenfaces, phân loại kNN/Bayes/SVM nhiều lớp hoặc Random Forest.
- Phân đoạn không tự hiểu tên đối tượng, không bảo đảm mỗi cụm liền mạch và không tự tách nền chính xác trong mọi ảnh.
- Ảnh lớn được thu nhỏ theo cạnh tối đa đã chọn; ảnh lưu có kích thước xử lý, không nhất thiết bằng kích thước ảnh gốc.
- Màu trong ảnh phân đoạn là màu minh họa nhãn, không phải màu thật. Chức năng giảm màu giữ các màu đại diện học từ ảnh.
- Giảm số màu không bảo đảm dung lượng PNG giảm theo đúng tỷ lệ K/24-bit vì còn phụ thuộc cách mã hóa.
- Ảnh có độ trong suốt được ghép lên nền trắng. Ảnh động chỉ lấy khung hình đầu.
- Với ảnh có quá ít màu/đặc trưng khác nhau, số cụm thực tế có thể nhỏ hơn K yêu cầu.
- Mọi lần xử lý đều bắt đầu từ ảnh gốc đang mở. Xử lý cục bộ, không gửi ảnh lên Internet.
