"""Local browser app: python web_local.py, then http://127.0.0.1:8000."""
import argparse
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from api.process import handler


class LocalHandler(handler, SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(Path(__file__).parent / 'public'), **kwargs)

    def do_GET(self):
        if self.path.split('?')[0] == '/api/process':
            return handler.do_GET(self)
        return SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        if self.path.split('?')[0] != '/api/process':
            return self.reply(404, {'error':'Không tìm thấy đường dẫn.'})
        return handler.do_POST(self)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    print(f'Open http://127.0.0.1:{args.port}', flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port), LocalHandler).serve_forever()
