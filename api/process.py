"""Vercel Python function. Images are processed in memory, never persisted."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'LOKY_MAX_CPU_COUNT'):
    os.environ.setdefault(name, '1')
import base64
import binascii
import io
import json
import math
import threading
import time
import warnings
from http.server import BaseHTTPRequestHandler
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image, ImageOps, UnidentifiedImageError
from workflows import run_experiment, FIGURES

METHODS = dict(zip(('quantize', 'segment', 'spectral', 'faces', 'face-parameters', 'hog', 'hog-svm'), FIGURES))
MAX_BODY = 3_900_000
MAX_RESPONSE = 4_000_000
LOCK = threading.BoundedSemaphore(1)
RANGES = {'k': (2,16,True), 'weight': (0,2,False), 'max_side': (128,512,True),
          'scale': (1.01,2,False), 'neighbors': (0,30,True), 'min_size': (10,500,True),
          'cell': (4,32,True), 'threshold': (0,3,False), 'iou': (0.05,1,False)}


class InputError(ValueError):
    pass


def image_uri(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


def execute(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get('method'), str) or payload['method'] not in METHODS:
        raise InputError('Chọn một chức năng hợp lệ.')
    raw_options = payload.get('options', {})
    if not isinstance(raw_options, dict) or set(raw_options) - set(RANGES):
        raise InputError('Tham số không hợp lệ.')
    options = {'max_side':512, 'min_size':60 if payload['method']=='hog-svm' else 30}
    for key, value in raw_options.items():
        lo, hi, integral = RANGES[key]
        if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value) or not lo <= value <= hi or (integral and int(value) != value):
            raise InputError(f'Tham số {key} phải nằm trong {lo}–{hi}.')
        options[key] = int(value) if integral else value
    if payload['method']=='hog-svm' and options['min_size'] < 40:
        raise InputError('Bản web yêu cầu mặt tối thiểu 40 px để giới hạn thời gian quét.')
    encoded = payload.get('image')
    if not isinstance(encoded, str) or len(encoded) > 3_800_000:
        raise InputError('Ảnh quá lớn. Hãy chọn ảnh nhỏ hơn.')
    try:
        binary = base64.b64decode(encoded, validate=True)
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(binary)) as uploaded:
                if uploaded.format not in ('PNG','JPEG','WEBP') or uploaded.width * uploaded.height > 4_194_304:
                    raise InputError('Chỉ nhận PNG/JPG/WebP tối đa 4 triệu điểm ảnh.')
                uploaded.load()
                oriented = ImageOps.exif_transpose(uploaded).convert('RGBA')
                white = Image.new('RGBA', oriented.size, 'white')
                source = Image.alpha_composite(white, oriented).convert('RGB')
    except (UnidentifiedImageError, OSError, binascii.Error, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise InputError('Không đọc được ảnh. Hãy chọn JPG, PNG hoặc WebP hợp lệ.') from error
    received_size = list(source.size)
    source.thumbnail((512,512), Image.Resampling.LANCZOS)
    started = time.perf_counter()
    result = run_experiment(source, METHODS[payload['method']], options)
    metadata = dict(result.metadata)
    metadata.pop('figure', None)
    metadata.update(method=FIGURES[result.figure], received_size=received_size, execution='web', web_max_side=512)
    # Avoid shipping long per-window lists; counts and retained detections suffice for the UI.
    if 'boxes' in metadata:
        kept = metadata.pop('kept_indices')
        boxes, scores = metadata.pop('boxes'), metadata.pop('scores')
        metadata['retained_detections'] = [{'box':boxes[i], 'score':scores[i]} for i in kept]
    response = dict(method=payload['method'], title=FIGURES[result.figure], image=image_uri(result.image),
                    panels=[{'title':title,'image':image_uri(im)} for title,im in result.panels],
                    description=result.caption, notes=result.notes, metadata=metadata,
                    seconds=round(time.perf_counter()-started,2))
    body = json.dumps(response, ensure_ascii=False).encode('utf-8')
    if len(body) > MAX_RESPONSE:
        response['panels'] = []
        response['notes'] += ' Các ảnh thành phần vượt giới hạn phản hồi; ảnh ghép và tham số vẫn có thể tải xuống.'
        body = json.dumps(response, ensure_ascii=False).encode('utf-8')
    if len(body) > MAX_RESPONSE:
        raise InputError('Kết quả quá lớn. Giảm cạnh xử lý xuống 256 rồi thử lại.')
    return body


class handler(BaseHTTPRequestHandler):
    def reply(self, status, body):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.reply(200, {'ready': True, 'methods': list(METHODS), 'max_side':512})

    def do_POST(self):
        if self.headers.get('Content-Type','').split(';')[0] != 'application/json':
            return self.reply(415, {'error':'Yêu cầu phải có định dạng JSON.'})
        try:
            length = int(self.headers.get('Content-Length','0'))
        except ValueError:
            return self.reply(400, {'error':'Yêu cầu không hợp lệ.'})
        if not 0 < length <= MAX_BODY:
            return self.reply(413, {'error':'Ảnh quá lớn. Hãy giảm kích thước ảnh.'})
        if not LOCK.acquire(blocking=False):
            return self.reply(429, {'error':'Máy chủ đang xử lý một ảnh khác. Vui lòng thử lại sau vài giây.'})
        try:
            payload = json.loads(self.rfile.read(length))
            self.reply(200, execute(payload))
        except (ValueError, UnicodeDecodeError) as error:
            self.reply(400, {'error':str(error) if isinstance(error, (InputError,)) else 'Ảnh hoặc tham số chưa phù hợp. Kiểm tra kích thước ảnh và các giá trị đã chọn.'})
        except Exception as error:
            print('Image processing failed:', type(error).__name__, file=sys.stderr)
            self.reply(500, {'error':'Không xử lý được ảnh. Thử ảnh khác hoặc giảm kích thước; nếu vẫn lỗi, kiểm tra Function Logs.'})
        finally:
            LOCK.release()
