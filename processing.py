"""K-means image operations adapted from chapter 9, sections 2.1.2–2.1.3."""
from dataclasses import dataclass
import numpy as np
from PIL import Image, ImageOps
from sklearn.cluster import KMeans


MODES = ("Giảm số màu", "Phân đoạn theo màu", "Phân đoạn màu + vị trí")


@dataclass
class Result:
    source: Image.Image
    image: Image.Image
    labels: np.ndarray
    colors: np.ndarray
    counts: np.ndarray


def load_image(path):
    with Image.open(path) as image:
        image.load()
        image = ImageOps.exif_transpose(image)
        if image.mode in ("RGBA", "LA") or "transparency" in image.info:
            rgba = image.convert("RGBA")
            canvas = Image.new("RGBA", rgba.size, "white")
            image = Image.alpha_composite(canvas, rgba)
        return image.convert("RGB")


def process_image(image, mode, k=8, spatial_weight=0.6, max_side=1000):
    if mode not in MODES:
        raise ValueError("Chế độ xử lý không hợp lệ.")
    if not 2 <= k <= 32 or not 128 <= max_side <= 1600:
        raise ValueError("Số cụm phải từ 2–32; cạnh tối đa từ 128–1600.")
    if not 0 <= spatial_weight <= 2:
        raise ValueError("Trọng số vị trí phải từ 0 đến 2.")
    source = image.convert("RGB").copy()
    source.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    rgb = np.asarray(source, dtype=np.float32) / 255.0
    h, w = rgb.shape[:2]
    pixels = rgb.reshape(-1, 3)
    if len(pixels) < k:
        raise ValueError("Ảnh có ít điểm ảnh hơn số cụm. Hãy giảm số cụm.")
    features = pixels
    if mode == MODES[2]:
        yy, xx = np.mgrid[:h, :w]
        coords = np.column_stack((yy.ravel() / h, xx.ravel() / w))
        features = np.column_stack((pixels, spatial_weight * coords)).astype(np.float32)

    # Fit a reproducible random subset; predict every pixel in bounded batches.
    rng = np.random.default_rng(42)
    ids = rng.choice(len(features), min(15000, len(features)), replace=False)
    sample = features[ids]
    actual_k = min(k, len(np.unique(sample, axis=0)))
    model = KMeans(n_clusters=actual_k, n_init=4, max_iter=150, random_state=42)
    model.fit(sample)
    labels = np.concatenate([
        model.predict(features[start:start + 65536])
        for start in range(0, len(features), 65536)
    ])
    counts = np.bincount(labels, minlength=actual_k)
    colors = np.clip(np.rint(model.cluster_centers_[:, :3] * 255), 0, 255).astype(np.uint8)
    if mode == MODES[0]:
        palette = colors
    else:
        # False colors visualize cluster labels; they are not semantic object labels.
        import colorsys
        palette = np.array([
            tuple(round(v * 255) for v in colorsys.hsv_to_rgb((i * 0.618034) % 1, 0.65, 0.95))
            for i in range(actual_k)
        ], dtype=np.uint8)
    output = Image.fromarray(palette[labels].reshape(h, w, 3))
    return Result(source, output, labels.reshape(h, w), colors, counts)
