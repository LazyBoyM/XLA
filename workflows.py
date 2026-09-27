"""Single-photo experiments and reproducible figure exports for chapter 9."""
import json
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sklearn.cluster import KMeans, SpectralClustering
from processing import MODES, process_image

FIGURES = {
    "2.1": "Giảm màu bằng K-means",
    "2.2": "Phân đoạn màu và vị trí",
    "2.3": "So sánh K-means và Spectral",
    "4.3": "Phát hiện khuôn mặt và mắt",
    "4.4": "So sánh tham số phát hiện mặt",
    "4.7": "Hiển thị đặc trưng HOG",
    "4.9": "Phát hiện HOG + SVM và NMS",
}
SLUGS = dict(zip(FIGURES, ('giam_mau', 'phan_doan', 'spectral', 'khuon_mat', 'tham_so_khuon_mat', 'hog', 'hog_svm_nms')))
DEFAULTS = dict(k=4, weight=0.6, max_side=800, scale=1.1, neighbors=5, min_size=30, cell=16, threshold=0.6, iou=0.25)


@dataclass
class Experiment:
    figure: str
    panels: list
    image: Image.Image
    caption: str
    metadata: dict
    notes: str


def font(size):
    for name in (str(Path(__file__).parent / 'assets' / 'DejaVuSans.ttf'), "C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def compose(panels):
    """A scientific comparison plate; every panel retains its aspect ratio."""
    columns = len(panels)
    rows = 1
    cw, ch, gap = 500, 450, 20
    sheet = Image.new("RGB", (columns * cw + (columns + 1) * gap, rows * ch + (rows + 1) * gap), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (title, image) in enumerate(panels):
        x, y = gap + (i % columns) * (cw + gap), gap + (i // columns) * (ch + gap)
        draw.text((x, y), title, fill="#152b38", font=font(23))
        preview = image.copy()
        preview.thumbnail((cw, ch - 48), Image.Resampling.NEAREST if max(image.size) <= 64 else Image.Resampling.LANCZOS)
        # Enlarge the 64x64 evidence for readability, without inventing detail.
        if max(image.size) <= 64:
            ratio = min(cw / image.width, (ch - 48) / image.height)
            preview = image.resize((int(image.width * ratio), int(image.height * ratio)), Image.Resampling.NEAREST)
        sheet.paste(preview, (x + (cw - preview.width) // 2, y + 45 + (ch - 48 - preview.height) // 2))
    return sheet


def prepared(image, max_side):
    source = image.convert("RGB").copy()
    source.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return source


def label_image(labels, size, k):
    import colorsys
    palette = np.array([tuple(round(v * 255) for v in colorsys.hsv_to_rgb((i * 0.618034) % 1, 0.65, 0.95)) for i in range(k)], dtype=np.uint8)
    return Image.fromarray(palette[labels].reshape(size[1], size[0], 3))


def detect(image, scale=1.1, neighbors=5, min_size=30, eyes=True):
    import cv2
    rgb = np.asarray(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    face_model = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_model = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    if face_model.empty() or eye_model.empty():
        raise RuntimeError("Không tìm thấy mô hình Haar cascade đi kèm OpenCV.")
    boxes = face_model.detectMultiScale(gray, scaleFactor=scale, minNeighbors=neighbors, minSize=(min_size, min_size))
    output = image.copy()
    draw = ImageDraw.Draw(output)
    records = []
    thickness = max(2, image.width // 250)
    for x, y, w, h in boxes:
        x, y, w, h = map(int, (x, y, w, h))
        draw.rectangle((x, y, x+w, y+h), outline="#00d6a1", width=thickness)
        eye_boxes = []
        if eyes:
            for ex, ey, ew, eh in eye_model.detectMultiScale(gray[y:y+h, x:x+w], scaleFactor=1.1, minNeighbors=6):
                ex, ey, ew, eh = map(int, (ex, ey, ew, eh))
                draw.rectangle((x+ex, y+ey, x+ex+ew, y+ey+eh), outline="#ffce50", width=thickness)
                eye_boxes.append([x+ex, y+ey, ew, eh])
        records.append(dict(face=[x, y, w, h], eyes=eye_boxes))
    return output, records


def run_experiment(image, figure, options=None):
    if figure not in FIGURES:
        raise ValueError("Hình không được hỗ trợ.")
    p = {**DEFAULTS, **(options or {})}
    if not 128 <= p['max_side'] <= 1600:
        raise ValueError("Cạnh tối đa phải từ 128 đến 1600.")
    if figure in ("2.2", "2.3") and (not 2 <= p['k'] <= 16 or not 0 <= p['weight'] <= 2):
        raise ValueError("K phải từ 2–16 và trọng số vị trí từ 0–2.")
    if figure in ("4.3", "4.4") and (not 1.01 <= p['scale'] <= 2 or not 0 <= p['neighbors'] <= 30 or not 10 <= p['min_size'] <= 500):
        raise ValueError("scaleFactor: 1,01–2; minNeighbors: 0–30; minSize: 10–500.")
    if figure == "4.7" and p['cell'] not in (4, 8, 16, 32):
        raise ValueError("Kích thước ô HOG phải là 4, 8, 16 hoặc 32.")
    source = prepared(image, p['max_side'])
    meta = dict(figure=figure, original_size=list(image.size), parameters=p, random_state=42)
    notes = "Kết quả minh họa trên một ảnh, không phải phép đo độ chính xác trên tập kiểm tra."
    panels = [("Ảnh đầu vào", source)]
    if figure == "2.1":
        meta['input_unique_rgb'] = int(len(np.unique(np.asarray(source).reshape(-1, 3), axis=0)))
        meta['actual_colors'] = {}
        for k in (2, 4, 8, 16):
            result = process_image(source, MODES[0], k=k, max_side=p['max_side'])
            panels.append((f"K = {k}", result.image))
            meta['actual_colors'][str(k)] = int(len(np.unique(np.asarray(result.image).reshape(-1, 3), axis=0)))
        meta.update(train_max_pixels=15000, n_init=4, max_iter=150)
        caption = "Hình 2.1. Lượng tử hóa màu trên ảnh do nhóm cung cấp bằng K-means với K = 2, 4, 8 và 16."
        notes += f" Ảnh đầu vào sau thu nhỏ có {meta['input_unique_rgb']:,} màu RGB khác nhau. Số màu không phải số màu mắt người phân biệt được; dung lượng PNG không giảm theo một tỷ lệ cố định."
    elif figure == "2.2":
        for mode, title in ((MODES[1], f"RGB, K = {p['k']}"), (MODES[2], f"RGB + vị trí, α = {p['weight']:g}")):
            result = process_image(source, mode, k=p['k'], spatial_weight=p['weight'], max_side=p['max_side'])
            panels.append((title, result.image))
        meta.update(train_max_pixels=15000, n_init=4, max_iter=150)
        caption = f"Hình 2.2. So sánh K-means dùng đặc trưng RGB và RGB kết hợp tọa độ chuẩn hóa, K = {p['k']}, trọng số vị trí α = {p['weight']:g}. Màu hiển thị biểu diễn nhãn cụm."
        notes += " Cùng màu minh họa giữa hai kết quả không nhất thiết chỉ cùng vùng. Việc bổ sung tọa độ không bảo đảm mọi vùng đều liền mạch."
    elif figure == "2.3":
        # Keep geometry: center crop a square, explicitly recorded in export.
        source = ImageOps.fit(image.convert("RGB"), (64, 64), method=Image.Resampling.LANCZOS)
        rgb = np.asarray(source, dtype=np.float64) / 255
        yy, xx = np.mgrid[:64, :64]
        features = np.column_stack((rgb.reshape(-1, 3), p['weight'] * yy.ravel()/64, p['weight'] * xx.ravel()/64))
        if len(np.unique(features, axis=0)) < p['k']:
            raise ValueError("Ảnh có quá ít đặc trưng khác nhau. Hãy dùng ảnh có chi tiết hoặc tăng trọng số vị trí.")
        km = KMeans(n_clusters=p['k'], n_init=10, random_state=42).fit_predict(features)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            sp = SpectralClustering(n_clusters=p['k'], affinity="nearest_neighbors", n_neighbors=12, assign_labels="kmeans", eigen_solver="arpack", random_state=42).fit_predict(features)
        meta.update(n_neighbors=12, affinity="nearest_neighbors", preprocessing="center square crop then resize 64x64", warnings=[str(w.message) for w in captured])
        panels = [("Ảnh đầu vào 64 × 64", source), (f"K-means, K = {p['k']}", label_image(km, source.size, p['k'])), (f"Spectral, K = {p['k']}", label_image(sp, source.size, p['k']))]
        caption = f"Hình 2.3. So sánh K-means và Spectral Clustering với K = {p['k']} trên cùng ảnh 64 × 64, sử dụng RGB và tọa độ chuẩn hóa với trọng số {p['weight']:g}; Spectral dùng đồ thị 12 láng giềng."
        notes += " Ảnh được cắt vuông ở giữa trước khi thu nhỏ. Các ảnh 64×64 được phóng lớn bằng nearest-neighbor khi ghép hình. Không mặc định Spectral tốt hơn K-means."
        if captured:
            notes += " Có cảnh báo từ thuật toán; xem metadata.json. Đồ thị không liên thông có thể ảnh hưởng cách diễn giải kết quả."
    elif figure == "4.3":
        output, boxes = detect(source, p['scale'], p['neighbors'], p['min_size'])
        panels.append((f"Phát hiện: {len(boxes)} vùng mặt", output))
        meta['detections'] = boxes
        caption = f"Hình 4.3. Phát hiện khuôn mặt (khung xanh) và mắt (khung vàng) bằng Haar cascade có sẵn trong OpenCV; scaleFactor = {p['scale']:g}, minNeighbors = {p['neighbors']}, minSize = {p['min_size']} × {p['min_size']}."
        notes += " Số vùng phát hiện không phải số khuôn mặt đúng. Mô hình đã huấn luyện sẵn; nhóm chỉ áp dụng lên ảnh mới. Có thể bỏ sót hoặc phát hiện nhầm."
    elif figure == "4.4":
        panels = []
        trials = [(1.05, 3), (1.05, 8), (1.2, 3), (1.2, 8)]
        meta['trials'] = []
        for scale, neighbors in trials:
            output, boxes = detect(source, scale, neighbors, p['min_size'], eyes=False)
            panels.append((f"scale={scale:g}, neighbors={neighbors}: {len(boxes)}", output))
            meta['trials'].append(dict(scaleFactor=scale, minNeighbors=neighbors, detections=boxes))
        caption = f"Hình 4.4. So sánh kết quả phát hiện khuôn mặt trên cùng ảnh với scaleFactor ∈ {{1,05; 1,20}} và minNeighbors ∈ {{3; 8}}, giữ minSize = {p['min_size']} × {p['min_size']}."
        notes += " Bốn cấu hình được cố định để dễ so sánh. Không phải ảnh nào cũng cho kết quả khác nhau; nếu giống nhau, ghi nhận đúng quan sát đó."
    elif figure == '4.9':
        from hog_detector import detect_hog
        source = prepared(image, min(p['max_side'], 512))
        boxes, scores, keep, detection_info = detect_hog(source, p['threshold'], p['iou'], p['min_size'])
        before, after = source.copy(), source.copy()
        draw_before, draw_after = ImageDraw.Draw(before), ImageDraw.Draw(after)
        for box in boxes:
            draw_before.rectangle(tuple(box), outline='#f26976', width=1)
        for index in keep:
            box = boxes[index]
            draw_after.rectangle(tuple(box), outline='#00d698', width=2)
            label = f'{scores[index]:.2f}'
            tx, ty = float(box[0]), max(0, float(box[1])-17)
            draw_after.rectangle(draw_after.textbbox((tx, ty), label, font=font(14)), fill='#182732')
            draw_after.text((tx, ty), label, fill='#00ffbc', font=font(14))
        panels = [(f'Trước NMS: {len(boxes)} hộp', before), (f'Sau NMS: {len(keep)} hộp', after)]
        meta.update(detection_info)
        caption = f'Hình 4.9. Phát hiện khuôn mặt bằng HOG + SVM trên ảnh đầu vào; trước NMS có {len(boxes)} hộp, sau NMS còn {len(keep)} hộp. Ngưỡng điểm SVM = {p["threshold"]:g}, ngưỡng IoU = {p["iou"]:g}, mặt tối thiểu {p["min_size"]} px trên ảnh xử lý.'
        notes += ' SVM huấn luyện bằng 100 ảnh mặt gốc trong lfw_subset (lật ngang thành 200 mẫu), ảnh âm mẫu và hai vòng khai thác mẫu âm khó. Ảnh người dùng chỉ để chạy thử, không dùng huấn luyện. Giới hạn cạnh xử lý 512 px để giảm thời gian quét. Điểm SVM không phải xác suất. NMS loại hộp chồng lấn, không bảo đảm loại được phát hiện sai. Không có hộp là kết quả hợp lệ; mô hình nhỏ có thể bỏ sót hoặc phát hiện nhầm.'
    else:
        from skimage.color import rgb2gray
        from skimage.feature import hog
        gray = rgb2gray(np.asarray(source))
        if min(gray.shape) < 2 * p['cell']:
            raise ValueError("Ảnh quá nhỏ so với ô HOG. Chọn ô nhỏ hơn hoặc ảnh lớn hơn.")
        vector, visual = hog(gray, orientations=9, pixels_per_cell=(p['cell'], p['cell']), cells_per_block=(2, 2), block_norm="L2-Hys", visualize=True, feature_vector=True)
        vmax = float(visual.max())
        display = (np.clip(visual / vmax, 0, 1) * 255).astype(np.uint8) if vmax > 0 else np.zeros_like(visual, dtype=np.uint8)
        panels = [("Ảnh xám đầu vào", Image.fromarray((gray * 255).astype(np.uint8)).convert("RGB")), (f"HOG: ô {p['cell']} × {p['cell']}, 9 hướng", Image.fromarray(display).convert("RGB"))]
        meta.update(feature_length=len(vector), orientations=9, cells_per_block=[2, 2], block_norm="L2-Hys", visualization="max normalized for display")
        caption = f"Hình 4.7. Trực quan hóa HOG trên ảnh của nhóm với 9 khoảng hướng, ô {p['cell']} × {p['cell']} điểm ảnh, khối 2 × 2 ô và chuẩn hóa L2-Hys."
        notes += " HOG biểu diễn gradient, chưa thực hiện phát hiện hay nhận dạng đối tượng. Độ sáng hình HOG được chuẩn hóa để dễ quan sát."
    meta['processed_size'] = list(source.size)
    caption = caption.split('. ', 1)[1]
    return Experiment(figure, panels, compose(panels), caption, meta, notes)


def export_experiment(result, directory, original_name):
    """Create a new run folder; preserve previous experiments and source files."""
    target = Path(directory) / (SLUGS[result.figure] + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    target.mkdir(parents=True, exist_ok=False)
    result.image.save(target / "so_sanh.png")
    for index, (title, image) in enumerate(result.panels):
        image.save(target / f"anh_{index+1:02d}.png")
    metadata = {**result.metadata, "source_file": Path(original_name).name, "panels": [title for title, _ in result.panels]}
    metadata.pop('figure', None)
    metadata['method'] = FIGURES[result.figure]
    import importlib.metadata
    metadata['versions'] = {name: importlib.metadata.version(name) for name in ('numpy', 'Pillow', 'scikit-learn', 'scikit-image', 'opencv-python-headless')}
    (target / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (target / "Mo_ta_ket_qua.txt").write_text(result.caption + "\n\n" + result.notes + "\n", encoding="utf-8-sig")
    return target
