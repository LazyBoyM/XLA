# Đưa Image Lab lên Vercel

Ứng dụng gồm giao diện HTML/CSS/JavaScript trong `public` và API Python trong `api/process.py`, với bảy chức năng xử lý ảnh.

## 1. Push các thay đổi lên GitHub

Đảm bảo repo có các thư mục `api`, `public`, `assets`, `models`, `scripts` và các file `vercel.json`, `.python-version`, `requirements.txt`, `workflows.py`, `processing.py`, `hog_detector.py`. Phải commit cả `models/hog_svm.npz` và font, không chỉ các file Python.

Không push `.venv`, cache, kết quả thử nghiệm hoặc `.env`. `.gitignore` đã loại các mục này.

## 2. Import trên Vercel

1. Chọn **Add New → Project → Import** repo GitHub.
2. **Root Directory:** chọn thư mục có `vercel.json`. Nếu repo chỉ chứa app thì để root mặc định; nếu repo chứa cả báo cáo và thư mục `image_tool`, chọn `image_tool`.
3. **Framework Preset:** chọn **Other**. Không chọn Flask hoặc Next.js; bản này dùng Python Function dạng file trong `/api`.
4. **Output Directory:** `public` (đã cấu hình trong `vercel.json`).
5. **Build Command:** `python3 scripts/check_assets.py` (đã cấu hình).
6. Để Install Command theo mặc định. Vercel tự cài các thư viện từ `requirements.txt` cho Python Function. Không đặt lệnh chạy `app.py` hoặc `web_local.py` làm build command.
7. Dùng Python **3.12**, đã khai báo trong `.python-version`. Không cần biến môi trường hoặc khóa API.
8. Deploy. Sau khi build thành công, mở URL, chọn **Thử với ảnh mẫu** và chạy một chức năng. Thử riêng HOG + SVM để xác nhận file mô hình đã được đóng gói.

Mã nguồn này đã có cấu hình để triển khai; việc kiểm tra trên máy cục bộ không thay thế một lần deploy thực tế vào tài khoản Vercel của bạn.

## 3. Các điều chỉnh cho môi trường web

- Ảnh được trình duyệt chuyển thành PNG tối đa 512 px trước khi gửi. Máy chủ cũng kiểm tra định dạng, kích thước và tham số. File gốc tối đa 25 MB trong giao diện.
- Spectral dùng ảnh vuông 64×64. HOG + SVM giới hạn mặt tối thiểu 40 px; mặc định 60 px.
- SVM dùng file trọng số được huấn luyện trước, không huấn luyện hoặc tải bộ dữ liệu trong request trên Vercel.
- API xử lý trong bộ nhớ, không ghi ảnh vào ổ đĩa và không có cơ sở dữ liệu. Ảnh được gửi đến máy chủ của deployment; đây không phải xử lý hoàn toàn trong trình duyệt.
- Ảnh kết quả ghép một hàng ngang. Tải PNG, metadata JSON, mô tả và từng ảnh thành phần. Nếu tổng phản hồi quá lớn, API chỉ trả ảnh ghép cùng thông số và thông báo rõ việc bỏ ảnh thành phần.
- API giới hạn request dưới 3,9 MB và response dưới 4 MB để chừa khoảng trống dưới giới hạn Vercel 4,5 MB. Thời gian tối đa cấu hình 60 giây; trình duyệt chờ tối đa 70 giây.
- Giới hạn một lượt xử lý đồng thời mỗi instance để giảm áp lực RAM/CPU. Yêu cầu trùng thời điểm có thể nhận thông báo thử lại; Vercel có thể tạo thêm instance theo gói sử dụng.

## 4. Nếu deploy gặp lỗi

- **Missing deployment asset:** kiểm tra đã push `models/hog_svm.npz`, font và các file `public`. Để tạo lại trọng số: cài requirements trên máy rồi chạy `python scripts/prepare_model.py`, commit file NPZ mới.
- **404 ở `/api/process`:** kiểm tra Root Directory, chọn Framework Other và giữ nguyên `api/process.py`. `/api/process` qua GET phải trả JSON có `ready: true`.
- **FUNCTION_INVOCATION_TIMEOUT / 504:** thử cạnh 256 px; với HOG + SVM tăng mặt tối thiểu. Kiểm tra maxDuration trong project không ghi đè cấu hình 60 giây.
- **FUNCTION_PAYLOAD_TOO_LARGE / 413:** giảm kích thước ảnh. Giao diện đã tự thu nhỏ; không gọi API bằng ảnh gốc quá lớn.
- **Bundle vượt giới hạn:** kiểm tra `.venv` hoặc kết quả ảnh không nằm trong repo/build. `.vercelignore` và `excludeFiles` đã loại dữ liệu local. Nếu thư viện của môi trường build vẫn vượt giới hạn chuẩn, kiểm tra lựa chọn Large Functions của Vercel; không bật dịch vụ trả phí khi chưa kiểm tra gói tài khoản.
- **500 khi xử lý:** xem Function Logs để biết loại lỗi; thử ảnh mẫu. Không cần đưa ảnh riêng hay khóa tài khoản vào báo cáo lỗi.

## Tài liệu nền tảng

- Python API directory: https://vercel.com/docs/functions/runtimes/python/api-directory
- Python runtime / phiên bản / bundle: https://vercel.com/docs/functions/runtimes/python
- Giới hạn thời gian và payload: https://vercel.com/docs/functions/limitations

Thông tin cấu hình được đối chiếu tài liệu Vercel ngày 27/09/2026; hạn mức thực tế còn phụ thuộc gói và thiết lập của project.
