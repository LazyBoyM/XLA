"""Desktop demonstration for chapter 9. Run: python app.py"""
import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(min(4, os.cpu_count() or 1)))
os.environ.setdefault("OMP_NUM_THREADS", "4")
import queue
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
from processing import load_image
from workflows import FIGURES, DEFAULTS, run_experiment, export_experiment


class ImageTool(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Nhóm 13 • Công cụ xử lý ảnh — Chương 9")
        self.geometry("1180x780")
        self.minsize(1080, 720)
        self.configure(bg="#eef2f6")
        self.source = self.result = None
        self.filename = "anh_mau"
        self.busy = False
        self.messages = queue.Queue()
        self.photos = {}
        self.preview_job = None
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#eef2f6")
        style.configure("TLabel", background="#eef2f6", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10), padding=8)
        style.configure("Title.TLabel", font=("Segoe UI", 20, "bold"))
        style.configure("Accent.TButton", background="#126b70", foreground="white")
        wrapper = ttk.Frame(self, padding=20)
        wrapper.pack(fill="both", expand=True)
        ttk.Label(wrapper, text="Phòng thực hành xử lý ảnh", style="Title.TLabel").pack(anchor="w")
        ttk.Label(wrapper, text="K-means · Spectral Clustering · Haar Cascade · HOG").pack(anchor="w", pady=(4, 14))
        toolbar = ttk.Frame(wrapper)
        toolbar.pack(fill="x")
        self.open_button = ttk.Button(toolbar, text="1. Mở ảnh…", command=self.open_image)
        self.open_button.pack(side="left")
        self.demo_button = ttk.Button(toolbar, text="Dùng ảnh mẫu", command=self.demo)
        self.demo_button.pack(side="left", padx=8)
        self.save_button = ttk.Button(toolbar, text="3. Lưu kết quả PNG…", command=self.save, state="disabled")
        self.save_button.pack(side="right")
        self.export_button = ttk.Button(toolbar, text="Xuất kết quả & tham số…", command=self.export, state="disabled")
        self.export_button.pack(side="right", padx=8)
        ttk.Button(toolbar, text="Hướng dẫn", command=self.guide).pack(side="left")
        self.file_info = tk.StringVar(value="Chưa chọn ảnh. Hỗ trợ JPG, PNG, BMP, TIFF và WebP.")
        ttk.Label(wrapper, textvariable=self.file_info).pack(anchor="w", pady=10)
        settings = ttk.Frame(wrapper)
        settings.pack(fill="x", pady=(0, 8))
        ttk.Label(settings, text="Chức năng").grid(row=0, column=0, sticky="w")
        self.mode = tk.StringVar(value=FIGURES['2.1'])
        self.mode_box = ttk.Combobox(settings, values=list(FIGURES.values()), textvariable=self.mode, state="readonly", width=36)
        self.mode_box.grid(row=1, column=0, padx=(0, 15), sticky="w")
        self.mode_box.bind("<<ComboboxSelected>>", self.update_help)
        ttk.Label(settings, text="Số cụm K (2–16)").grid(row=0, column=1, sticky="w")
        self.k = tk.StringVar(value="4")
        self.k_box = ttk.Spinbox(settings, from_=2, to=16, textvariable=self.k, width=8)
        self.k_box.grid(row=1, column=1, padx=(0, 15), sticky="w")
        ttk.Label(settings, text="Trọng số vị trí (0–2)").grid(row=0, column=2, sticky="w")
        self.weight = tk.StringVar(value="0.6")
        self.weight_box = ttk.Spinbox(settings, from_=0, to=2, increment=0.1, textvariable=self.weight, width=8)
        self.weight_box.grid(row=1, column=2, padx=(0, 15), sticky="w")
        ttk.Label(settings, text="Cạnh tối đa khi xử lý").grid(row=0, column=3, sticky="w")
        self.max_side = tk.StringVar(value="800")
        self.size_box = ttk.Combobox(settings, values=(256, 512, 800, 1000, 1600), textvariable=self.max_side, width=8, state="readonly")
        self.size_box.grid(row=1, column=3, sticky="w")
        self.run_button = ttk.Button(settings, text="2. Xử lý ảnh", style="Accent.TButton", command=self.run, state="disabled")
        self.run_button.grid(row=1, column=4, padx=(18, 0))
        extra = ttk.Frame(wrapper)
        extra.pack(fill="x", pady=(2, 8))
        self.extra_vars, self.extra_boxes = {}, {}
        for index, (key, title, value, low, high, step) in enumerate((
            ('scale', 'scaleFactor', '1.1', 1.01, 2, 0.05),
            ('neighbors', 'minNeighbors', '5', 0, 30, 1),
            ('min_size', 'Mặt tối thiểu (px)', '30', 10, 500, 10),
            ('cell', 'Ô HOG (4/8/16/32)', '16', 4, 32, 4),
            ('threshold', 'Ngưỡng điểm SVM', '0.6', -2, 3, 0.1),
            ('iou', 'Ngưỡng IoU NMS', '0.25', 0, 1, 0.05),
        )):
            row, column = divmod(index, 4)
            ttk.Label(extra, text=title).grid(row=row, column=column*2, padx=(0, 8), pady=3)
            variable = tk.StringVar(value=value)
            box = ttk.Spinbox(extra, from_=low, to=high, increment=step, width=7, textvariable=variable)
            box.grid(row=row, column=column*2+1, padx=(0, 18), pady=3)
            self.extra_vars[key], self.extra_boxes[key] = variable, box
        self.help_text = tk.StringVar()
        ttk.Label(wrapper, textvariable=self.help_text, wraplength=1050).pack(anchor="w", pady=(2, 12))
        previews = ttk.Frame(wrapper)
        previews.pack(fill="both", expand=True)
        previews.columnconfigure((0, 1), weight=1, uniform="preview")
        previews.rowconfigure(1, weight=1)
        ttk.Label(previews, text="ẢNH GỐC").grid(row=0, column=0, sticky="w", pady=(0, 6))
        ttk.Label(previews, text="KẾT QUẢ").grid(row=0, column=1, sticky="w", padx=(12, 0), pady=(0, 6))
        self.left = tk.Canvas(previews, bg="#182732", highlightthickness=0)
        self.right = tk.Canvas(previews, bg="#182732", highlightthickness=0)
        self.left.grid(row=1, column=0, sticky="nsew")
        self.right.grid(row=1, column=1, sticky="nsew", padx=(12, 0))
        previews.bind("<Configure>", self.schedule_preview)
        self.progress = ttk.Progressbar(wrapper, mode="indeterminate")
        self.progress.pack(fill="x", pady=(12, 6))
        self.status = tk.StringVar(value="Mở ảnh để bắt đầu hoặc dùng ảnh mẫu có sẵn.")
        ttk.Label(wrapper, textvariable=self.status, wraplength=1050).pack(anchor="w")
        ttk.Label(wrapper, text="Xử lý trên máy tính • Xuất ảnh so sánh, ảnh thành phần và tham số của từng lần chạy.", wraplength=1050).pack(anchor="w", pady=(6, 0))
        self.update_help()
        self.after(100, self.poll)

    def update_help(self, event=None):
        figure = self.figure_id()
        spatial = figure in ('2.2', '2.3')
        self.weight_box.configure(state="normal" if spatial else "disabled")
        self.k_box.configure(state="normal" if spatial else "disabled")
        self.size_box.configure(state="disabled" if figure == '2.3' else "readonly")
        if event:
            self.k.set('3' if figure == '2.3' else '4')
            self.weight.set('0.5' if figure == '2.3' else '0.6')
            self.extra_vars['min_size'].set('60' if figure == '4.9' else '30')
        for key, box in self.extra_boxes.items():
            active = (figure == '4.3' and key in ('scale', 'neighbors', 'min_size')) or (figure == '4.4' and key == 'min_size') or (figure == '4.7' and key == 'cell') or (figure == '4.9' and key in ('threshold', 'iou', 'min_size'))
            box.configure(state="normal" if active else "disabled")
        explanations = {
            '2.1': "Tự tạo bộ ảnh gốc + K=2, 4, 8, 16 và đếm số màu ảnh đầu vào. Chọn ảnh phong cảnh hoặc hoa nhiều màu.",
            '2.2': "So sánh RGB và RGB + vị trí trên cùng ảnh. Gợi ý K=4, trọng số 0,6. Chọn vật thể nổi bật trên nền đơn giản.",
            '2.3': "Cắt vuông ở giữa rồi thu nhỏ 64×64 để so sánh K-means và Spectral (12 láng giềng). Gợi ý K=3, trọng số 0,5. Có thể chạy chậm hơn.",
            '4.3': "Dùng Haar cascade có sẵn. Mặt: khung xanh; mắt: khung vàng. Chọn chân dung chính diện, đủ sáng. Không phát hiện được cũng là một kết quả hợp lệ.",
            '4.4': "Tự chạy 4 cấu hình: scaleFactor 1,05/1,20 × minNeighbors 3/8. Giữ minSize cố định để so sánh. Kết quả có thể giống nhau.",
            '4.7': "HOG: 9 hướng, khối 2×2 ô, L2-Hys. Chọn ảnh người hoặc vật thể có đường nét rõ. Đây là hình đặc trưng, chưa phải nhận dạng đối tượng.",
            '4.9': "HOG + SVM: dùng mô hình đã huấn luyện sẵn từ ảnh mẫu và khai thác mẫu âm khó. Quét ảnh tối đa 512 px, so sánh trước/sau NMS. Có thể mất vài chục giây; vẫn có thể phát hiện nhầm.",
        }
        self.help_text.set(explanations[figure])

    def figure_id(self):
        return next(key for key, value in FIGURES.items() if value == self.mode.get())

    def set_source(self, image, filename):
        self.source, self.filename, self.result = image, filename, None
        self.file_info.set(f"{Path(filename).name}  ·  {image.width} × {image.height} px")
        self.status.set("Đã mở ảnh. Chọn chức năng, tham số rồi bấm Xử lý ảnh.")
        self.run_button.configure(state="normal")
        self.save_button.configure(state="disabled")
        self.export_button.configure(state="disabled")
        self.render_previews()

    def open_image(self):
        path = filedialog.askopenfilename(title="Chọn ảnh của bạn", filetypes=[("Ảnh", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"), ("Tất cả", "*.*")])
        if path:
            try:
                self.set_source(load_image(path), path)
            except Exception as error:
                messagebox.showerror("Không mở được ảnh", str(error))

    def demo(self):
        image = Image.new("RGB", (800, 520), "#b8dfe7")
        draw = ImageDraw.Draw(image)
        for y in range(520):
            draw.line((0, y, 800, y), fill=(100 + y // 5, 175 + y // 10, 215))
        draw.ellipse((550, 45, 690, 185), fill="#ffcc66")
        draw.polygon([(0, 430), (220, 150), (460, 520), (0, 520)], fill="#347b67")
        draw.polygon([(210, 520), (510, 220), (800, 520)], fill="#5b9c7b")
        draw.rectangle((525, 330, 655, 480), fill="#e39b6d")
        draw.polygon([(505, 335), (590, 260), (675, 335)], fill="#984f50")
        self.set_source(image, "anh_mau.png")

    def run(self):
        if self.source is None or self.busy:
            return
        try:
            figure = self.figure_id()
            options = dict(DEFAULTS)
            options['max_side'] = int(self.max_side.get())
            if figure in ('2.2', '2.3'):
                options.update(k=int(self.k.get()), weight=float(self.weight.get().replace(',', '.')))
            for key, box in self.extra_boxes.items():
                if str(box['state']) != 'disabled':
                    text = self.extra_vars[key].get().replace(',', '.')
                    options[key] = float(text) if key in ('scale', 'threshold', 'iou') else int(text)
        except ValueError:
            messagebox.showerror("Tham số không hợp lệ", "Kiểm tra các giá trị số; chỉ nhập số nguyên vào K, minNeighbors, minSize và ô HOG.")
            return
        self.busy = True
        for button in (self.open_button, self.demo_button, self.run_button, self.save_button, self.export_button):
            button.configure(state="disabled")
        self.status.set(f"Đang xử lý: {FIGURES[figure]}…")
        self.progress.start(12)
        source = self.source.copy()

        def work():
            started = time.perf_counter()
            try:
                result = run_experiment(source, figure, options)
                self.messages.put((result, options, time.perf_counter() - started, None))
            except Exception as error:
                self.messages.put((None, options, 0, str(error)))
        threading.Thread(target=work, daemon=True).start()

    def poll(self):
        try:
            result, options, seconds, error = self.messages.get_nowait()
        except queue.Empty:
            pass
        else:
            self.busy = False
            self.progress.stop()
            for button in (self.open_button, self.demo_button, self.run_button):
                button.configure(state="normal")
            if error:
                self.status.set("Xử lý chưa thành công. Kiểm tra tham số rồi thử lại.")
                messagebox.showerror("Không xử lý được ảnh", error)
            else:
                self.result = result
                self.status.set(f"{FIGURES[result.figure]} · {seconds:.1f} giây · Xử lý hoàn tất." + (" Có cảnh báo Spectral; xem ghi chú khi xuất." if result.metadata.get('warnings') else ""))
                self.render_previews()
            self.save_button.configure(state="normal" if self.result else "disabled")
            self.export_button.configure(state="normal" if self.result else "disabled")
        self.after(100, self.poll)

    def schedule_preview(self, event=None):
        if self.preview_job:
            self.after_cancel(self.preview_job)
        self.preview_job = self.after(100, self.render_previews)

    def render_previews(self):
        self.preview_job = None
        for name, canvas, image in (("left", self.left, self.source), ("right", self.right, self.result.image if self.result else None)):
            canvas.delete("all")
            width, height = max(canvas.winfo_width(), 100), max(canvas.winfo_height(), 100)
            if image is None:
                canvas.create_text(width / 2, height / 2, text="Chưa có ảnh" if name == "left" else "Kết quả sẽ xuất hiện ở đây", fill="#bbcbd6", font=("Segoe UI", 12))
            else:
                preview = image.copy()
                preview.thumbnail((max(width - 20, 1), max(height - 20, 1)), Image.Resampling.LANCZOS)
                self.photos[name] = ImageTk.PhotoImage(preview)
                canvas.create_image(width / 2, height / 2, image=self.photos[name])

    def save(self):
        if not self.result:
            return
        path = filedialog.asksaveasfilename(title="Lưu ảnh kết quả", initialfile="so_sanh.png", defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path:
            try:
                self.result.image.save(path, format="PNG")
                self.status.set(f"Đã lưu: {path}")
            except Exception as error:
                messagebox.showerror("Không lưu được ảnh", str(error))

    def export(self):
        if self.result is None or self.busy:
            return
        path = filedialog.askdirectory(title="Chọn thư mục lưu kết quả")
        if path:
            try:
                target = export_experiment(self.result, path, self.filename)
                self.status.set(f"Đã xuất kết quả: {target}")
                messagebox.showinfo("Đã xuất kết quả", f"Thư mục: {target}\n\nĐã lưu ảnh so sánh, ảnh thành phần, mô tả kết quả và tham số.")
            except Exception as error:
                messagebox.showerror("Không xuất được bộ hình", str(error))

    def guide(self):
        window = tk.Toplevel(self)
        window.title("Hướng dẫn sử dụng")
        window.geometry("850x650")
        text = tk.Text(window, wrap="word", font=("Segoe UI", 11), padx=18, pady=18)
        scrollbar = ttk.Scrollbar(window, command=text.yview)
        scrollbar.pack(side="right", fill="y")
        text.configure(yscrollcommand=scrollbar.set)
        text.pack(fill="both", expand=True)
        text.insert("1.0", (Path(__file__).parent / "README.md").read_text(encoding="utf-8"))
        text.configure(state="disabled")


if __name__ == "__main__":
    ImageTool().mainloop()
