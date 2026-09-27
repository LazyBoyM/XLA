"""Educational HOG + linear SVM face detector. No Haar proposals are used."""
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
import cv2
from skimage import data, color
from skimage.feature import hog
from sklearn.svm import LinearSVC

WINDOW = 25
HOG_OPTIONS = dict(orientations=9, pixels_per_cell=(5, 5), cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)


def feature(patch):
    return hog(patch, **HOG_OPTIONS)


def gray_float(image):
    array = np.asarray(image)
    return color.rgb2gray(array).astype(np.float32) if array.ndim == 3 else (array.astype(np.float32) / 255 if array.max() > 1 else array.astype(np.float32))


def scan(gray, coef, intercept, threshold=0.6, stride=3, min_size=25, collect_features=False):
    """Scan a fixed 25px window over an image pyramid; coordinates are exclusive."""
    height, width = gray.shape
    boxes, scores, vectors = [], [], []
    scale = min(1.0, WINDOW / min_size)
    evaluated = 0
    while min(round(height * scale), round(width * scale)) >= WINDOW:
        h, w = round(height * scale), round(width * scale)
        resized = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
        sy, sx = h / height, w / width
        batch, coords = [], []
        for y in range(0, h - WINDOW + 1, stride):
            for x in range(0, w - WINDOW + 1, stride):
                batch.append(feature(resized[y:y+WINDOW, x:x+WINDOW]))
                coords.append((x/sx, y/sy, (x+WINDOW)/sx, (y+WINDOW)/sy))
        if batch:
            batch = np.asarray(batch)
            predictions = batch @ coef + intercept
            selected = np.flatnonzero(predictions > threshold)
            boxes.extend(np.asarray(coords)[selected])
            scores.extend(predictions[selected])
            if collect_features:
                vectors.extend(batch[selected])
            evaluated += len(batch)
        scale *= 0.8
    return np.asarray(boxes, dtype=float).reshape(-1, 4), np.asarray(scores), np.asarray(vectors).reshape(-1, 576), evaluated


def nms(boxes, scores, iou_threshold=0.25):
    if len(boxes) == 0:
        return np.array([], dtype=int)
    boxes, scores = np.asarray(boxes, dtype=float), np.asarray(scores)
    if not 0 <= iou_threshold <= 1:
        raise ValueError('Ngưỡng IoU phải nằm trong [0, 1].')
    area = np.maximum(0, boxes[:, 2]-boxes[:, 0]) * np.maximum(0, boxes[:, 3]-boxes[:, 1])
    order = np.argsort(-scores, kind='stable')
    keep = []
    while len(order):
        index, others = order[0], order[1:]
        keep.append(index)
        left = np.maximum(boxes[index, :2], boxes[others, :2])
        right = np.minimum(boxes[index, 2:], boxes[others, 2:])
        intersection = np.prod(np.maximum(0, right-left), axis=1)
        union = area[index] + area[others] - intersection
        overlaps = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = others[overlaps <= iou_threshold]
    return np.asarray(keep, dtype=int)


@lru_cache(maxsize=1)
def trained_model():
    """Use portable numeric weights when bundled; desktop can train as fallback."""
    path = Path(__file__).parent / 'models' / 'hog_svm.npz'
    if path.is_file():
        with np.load(path, allow_pickle=False) as saved:
            coef = saved['coef'].copy()
            intercept = float(saved['intercept'])
            info = json.loads(str(saved['metadata']))
        if coef.shape != (576,) or not np.isfinite(coef).all() or not np.isfinite(intercept):
            raise RuntimeError('Mô hình HOG + SVM không hợp lệ.')
        return coef, intercept, info
    import os
    if os.environ.get('VERCEL'):
        raise RuntimeError('Thiếu models/hog_svm.npz. Hãy chạy scripts/prepare_model.py trước khi triển khai.')
    return train_model()


def train_model():
    """Explicit offline training. User images never enter the training set."""
    dataset = data.lfw_subset().astype(np.float32)
    positive = list(dataset[:100]) + [np.fliplr(im) for im in dataset[:100]]
    negative = list(dataset[100:])
    # These backgrounds contain no human faces; deliberately exclude camera().
    backgrounds = [gray_float(getattr(data, name)()) for name in ('coffee', 'chelsea', 'coins', 'text')]
    rng = np.random.default_rng(42)
    for background in backgrounds:
        h, w = background.shape
        for _ in range(160):
            size = int(rng.integers(25, min(h, w, 150) + 1))
            y, x = int(rng.integers(h-size+1)), int(rng.integers(w-size+1))
            negative.append(cv2.resize(background[y:y+size, x:x+size], (25, 25), interpolation=cv2.INTER_AREA))
    X = np.asarray([feature(patch) for patch in positive + negative])
    y = np.array([1]*len(positive) + [0]*len(negative))
    initial_negative_count = len(negative)
    hard_counts = []
    model = LinearSVC(C=1, max_iter=50000, dual=False, random_state=42).fit(X, y)
    for _ in range(2):
        mined = []
        for background in backgrounds:
            h, w = background.shape
            ratio = min(1, 160/max(h, w))
            small = cv2.resize(background, (round(w*ratio), round(h*ratio)), interpolation=cv2.INTER_AREA)
            _, scores, vectors, _ = scan(small, model.coef_[0], float(model.intercept_[0]), threshold=0, stride=4, collect_features=True)
            if len(scores):
                mined.extend(vectors[np.argsort(scores)[-80:]])
        hard_counts.append(len(mined))
        if mined:
            X = np.vstack((X, mined))
            y = np.concatenate((y, np.zeros(len(mined), dtype=int)))
            model = LinearSVC(C=1, max_iter=50000, dual=False, random_state=42).fit(X, y)
    info = dict(training_source='skimage.data.lfw_subset: 100 faces, 100 nonfaces', positive_original=100, positive_after_flip=200,
                negative_initial=initial_negative_count, background_sources=['coffee', 'chelsea', 'coins', 'text'],
                hard_negative_counts=hard_counts, hard_negative_rounds=2, C=1, random_state=42,
                window=[25,25], orientations=9, pixels_per_cell=[5,5], cells_per_block=[2,2], block_norm='L2-Hys',
                pyramid_factor=0.8, stride=3, implementation='skimage HOG + sklearn LinearSVC; no Haar proposals')
    return model.coef_[0].copy(), float(model.intercept_[0]), info


def detect_hog(image, threshold=0.6, iou=0.25, min_size=60):
    if not np.isfinite(threshold) or not -2 <= threshold <= 3:
        raise ValueError('Ngưỡng SVM phải từ -2 đến 3.')
    if not np.isfinite(iou) or not 0 <= iou <= 1:
        raise ValueError('Ngưỡng IoU phải từ 0 đến 1.')
    if min(image.size) < WINDOW:
        raise ValueError('Ảnh cần có mỗi chiều ít nhất 25 điểm ảnh.')
    if not 25 <= min_size <= 500:
        raise ValueError('Kích thước mặt tối thiểu cho HOG + SVM phải từ 25 đến 500 px.')
    coef, intercept, training = trained_model()
    boxes, scores, _, evaluated = scan(gray_float(image), coef, intercept, threshold, min_size=min_size)
    keep = nms(boxes, scores, iou)
    return boxes, scores, keep, dict(training=training, scanned_windows=evaluated, score_threshold=threshold, iou_threshold=iou, min_face_size=min_size,
                                    candidates=len(boxes), retained=len(keep), boxes=boxes.tolist(), scores=scores.tolist(), kept_indices=keep.tolist())
