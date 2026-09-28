# Công cụ xử lý ảnh — Nhóm 13

Ứng dụng Python chạy trên Windows bằng giao diện Tkinter. Chọn ảnh của bạn, điều chỉnh tham số, xem kết quả và xuất ảnh so sánh trên một hàng ngang. Ảnh được xử lý trên máy tính của bạn.

## Cài đặt lần đầu

Cài Python 3.12 cho Windows, có Tcl/Tk và pip. Khi cài, chọn **Add Python to PATH**. Clone hoặc tải ZIP repo, mở PowerShell trong thư mục có `app.py` rồi chạy:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Cần Internet để cài thư viện. Không sao chép `.venv` từ máy khác; mỗi máy tạo môi trường riêng.

## Mở ứng dụng

Nhấp đúp **start.cmd**, hoặc chạy:

```powershell
.\.venv\Scripts\python.exe app.py
```

1. Bấm **Mở ảnh** để chọn JPG, PNG, BMP, TIFF hoặc WebP. **Dùng ảnh mẫu** tạo ảnh phong cảnh minh họa để thử phân cụm.
2. Chọn chức năng và tham số đang được bật.
3. Bấm **Xử lý ảnh**, chờ kết quả.
4. **Lưu kết quả PNG** lưu ảnh ghép một hàng ngang. **Xuất kết quả & tham số** lưu thêm ảnh thành phần, metadata JSON và mô tả vào thư mục riêng.

## Chức năng

| Chức năng | Kết quả |
|---|---|
| Giảm số màu | Ảnh gốc và K-means với K=2, 4, 8, 16 |
| Phân đoạn màu và vị trí | RGB so với RGB kết hợp tọa độ |
| K-means và Spectral | Ảnh cắt vuông 64×64, đồ thị 12 láng giềng |
| Phát hiện mặt và mắt | Haar cascade của OpenCV |
| So sánh tham số phát hiện mặt | Bốn tổ hợp scaleFactor và minNeighbors |
| Hiển thị HOG | Hình gradient và số chiều đặc trưng |
| HOG + SVM và NMS | Hộp phát hiện trước/sau khi loại vùng chồng lấn |

Với phát hiện mặt, dùng ảnh chân dung chính diện, đủ sáng. HOG + SVM gợi ý mặt tối thiểu 60 px, ngưỡng SVM 0,6 và IoU 0,25. Giảm kích thước mặt tối thiểu giúp tìm mặt nhỏ nhưng tăng thời gian chạy. Điểm SVM không phải xác suất; mô hình có thể bỏ sót hoặc báo nhầm.

## Mã nguồn và tài nguyên

- `app.py`: giao diện; `start.cmd`: mở ứng dụng Windows.
- `processing.py`: đọc ảnh và K-means; `workflows.py`: thí nghiệm, ghép và xuất ảnh.
- `hog_detector.py`: HOG, SVM, quét cửa sổ và NMS.
- `models/hog_svm.npz`: trọng số đã huấn luyện, nạp với `allow_pickle=False`. Ảnh người dùng không dùng để huấn luyện.
- `scripts/prepare_model.py`: tạo lại trọng số bằng dữ liệu mẫu trong môi trường đã cài thư viện. Nếu thiếu trọng số, app sẽ thử huấn luyện lại khi dùng HOG + SVM.
- `assets/DejaVuSans.ttf`: font tiếng Việt cho ảnh xuất; giấy phép trong `assets/FONT_LICENSE.txt`.

Giữ `models` và `assets` khi sao chép ứng dụng. `.venv`, cache và kết quả trong `outputs` được bỏ qua khi push GitHub.

## Giới hạn

- Ảnh lớn được thu nhỏ theo cấu hình; Spectral xử lý 64×64, HOG + SVM tối đa 512 px.
- Phân cụm không tự hiểu tên đối tượng hoặc bảo đảm tách nền chính xác. Màu nhãn là màu minh họa.
- Số cụm thực tế có thể nhỏ hơn K nếu ảnh có ít màu. Giảm số màu không bảo đảm dung lượng PNG giảm tương ứng.
- Ảnh trong suốt ghép nền trắng; ảnh động chỉ lấy khung đầu.
- Chưa có PCA/Eigenfaces, kNN/Bayes hoặc Random Forest.
