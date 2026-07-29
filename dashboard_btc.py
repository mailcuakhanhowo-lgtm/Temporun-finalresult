import tkinter as tk
from tkinter import scrolledtext, font
import subprocess
import threading
import sys
import osimport json

# Đường dẫn file đồng bộ state giữa Terminal và UI
PATHS_FILE = "data/btc_paths.json"

# Đường dẫn mặc định (Phục vụ cho việc test nhanh của BTC)
DEFAULT_VIDEO_DIR = "data/videos"
DEFAULT_TASK_FILE = "data/private_round_tasks.jsonl"
DEFAULT_OUTPUT = "data/submission.json"

class TempoRunDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Tempo Run 2026 - Cỗ Xe Tăng Bất Tử (BTC Dashboard)")
        self.root.geometry("900x650")
        self.root.configure(bg="#f0f0f0")
        
        # Tiêu đề chính
        title_font = font.Font(family="Helvetica", size=16, weight="bold")
        tk.Label(root, text="🚀 HỆ THỐNG TRUY XUẤT VIDEO TỰ ĐỘNG - TEMPO RUN 2026", font=title_font, bg="#f0f0f0", fg="#333333").pack(pady=(15, 5))
        
        # Lưu ý đỏ cho BTC
        warn_text = (
            "LƯU Ý: Giao diện này mặc định chạy với dữ liệu mẫu (data/).\n"
            "Tuy nhiên, nếu ngài sửa đường dẫn ở dưới, hoặc đã chạy lệnh Terminal trước đó, các đường dẫn bí mật sẽ tự động được đồng bộ và lưu lại."
        )
        tk.Label(root, text=warn_text, fg="red", bg="#f0f0f0", font=("Arial", 10, "bold"), wraplength=850, justify="center").pack(pady=(0, 15))
        
        # 3 Ô nhập đường dẫn (Text Box)
        self.var_video = tk.StringVar(value=DEFAULT_VIDEO_DIR)
        self.var_task = tk.StringVar(value=DEFAULT_TASK_FILE)
        self.var_output = tk.StringVar(value=DEFAULT_OUTPUT)
        
        # Cố gắng nạp dữ liệu từ Terminal (nếu có)
        self.load_paths_from_json()
        
        frame_paths = tk.LabelFrame(root, text="Cấu hình Đường Dẫn (Tự động đồng bộ với Terminal)", bg="#f0f0f0", font=("Arial", 10, "bold"))
        frame_paths.pack(fill="x", padx=20, pady=5)
        
        tk.Label(frame_paths, text="--video_dir :", bg="#f0f0f0").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        tk.Entry(frame_paths, textvariable=self.var_video, width=80).grid(row=0, column=1, padx=5, pady=5)
        
        tk.Label(frame_paths, text="--task_file :", bg="#f0f0f0").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        tk.Entry(frame_paths, textvariable=self.var_task, width=80).grid(row=1, column=1, padx=5, pady=5)
        
        tk.Label(frame_paths, text="--output    :", bg="#f0f0f0").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        tk.Entry(frame_paths, textvariable=self.var_output, width=80).grid(row=2, column=1, padx=5, pady=5)
        
        # Nút Lưu Cấu Hình (Lưu đường dẫn mà không chạy)
        btn_save = tk.Button(frame_paths, text="💾 Lưu Cấu Hình\n(Không chạy)", bg="#5cb85c", fg="white", font=("Arial", 9, "bold"), command=self.save_paths_and_notify)
        btn_save.grid(row=0, column=2, rowspan=3, padx=15, pady=5, sticky="nsew")
        
        # Khung chứa các nút chạy lẻ
        frame_manual = tk.LabelFrame(root, text="Chạy từng bước (Manual Steps)", bg="#f0f0f0", font=("Arial", 10, "bold"))
        frame_manual.pack(fill="x", padx=20, pady=5)
        
        btn_font = font.Font(family="Arial", size=10)
        
        tk.Button(frame_manual, text="1. Cắt Ảnh (GĐ1)", font=btn_font, width=15, command=lambda: self.run_script("src/step1_extract_frames.py")).pack(side="left", padx=10, pady=10)
        tk.Button(frame_manual, text="2. Mã Hóa Ảnh (GĐ2)", font=btn_font, width=17, command=lambda: self.run_script("src/step2_encode_clip.py")).pack(side="left", padx=10, pady=10)
        tk.Button(frame_manual, text="3. Trạm 1 (LLM)", font=btn_font, width=15, command=lambda: self.run_script("src/station1_ollama.py")).pack(side="left", padx=10, pady=10)
        tk.Button(frame_manual, text="4. Trạm 2 (Nhúng CH)", font=btn_font, width=18, command=lambda: self.run_script("src/station2_clip.py")).pack(side="left", padx=10, pady=10)
        tk.Button(frame_manual, text="5. Trạm 3&4 (Truy Xuất)", font=btn_font, width=20, command=lambda: self.run_script("src/station34_pipeline.py")).pack(side="left", padx=10, pady=10)
        
        # Nút Chạy tự động (Pipeline hoàn chỉnh)
        frame_auto = tk.Frame(root, bg="#f0f0f0")
        frame_auto.pack(fill="x", padx=20, pady=10)
        
        btn_run_all = tk.Button(
            frame_auto, 
            text="🚀 CHẠY TOÀN BỘ QUY TRÌNH TỰ ĐỘNG (MAIN.PY)", 
            bg="#d9534f", fg="white", 
            font=("Arial", 12, "bold"), 
            height=2,
            command=self.run_full_pipeline
        )
        btn_run_all.pack(fill="x", pady=5)
        
        # Màn hình Log (Console đen chữ xanh)
        tk.Label(root, text="Màn hình Log (Console):", bg="#f0f0f0", font=("Arial", 10, "bold")).pack(anchor="w", padx=20)
        
        self.log_area = scrolledtext.ScrolledText(root, bg="#1e1e1e", fg="#4af626", font=("Consolas", 10))
        self.log_area.pack(fill="both", expand=True, padx=20, pady=(5, 20))
        
        self.process = None
        self.log("✅ Giao diện đã sẵn sàng! Đợi lệnh từ Ban Tổ chức...\n")

    def load_paths_from_json(self):
        try:
            if os.path.exists(PATHS_FILE):
                with open(PATHS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.var_video.set(data.get("video_dir", DEFAULT_VIDEO_DIR))
                    self.var_task.set(data.get("task_file", DEFAULT_TASK_FILE))
                    self.var_output.set(data.get("output", DEFAULT_OUTPUT))
        except Exception:
            pass
            
    def save_paths_to_json(self):
        try:
            os.makedirs(os.path.dirname(PATHS_FILE), exist_ok=True)
            with open(PATHS_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "video_dir": self.var_video.get(),
                    "task_file": self.var_task.get(),
                    "output": self.var_output.get()
                }, f, indent=4)
        except Exception:
            pass

    def save_paths_and_notify(self):
        self.save_paths_to_json()
        self.log(f"\n✅ ĐÃ LƯU CẤU HÌNH ĐƯỜNG DẪN THÀNH CÔNG!")
        self.log(f"  - Video: {self.var_video.get()}")
        self.log(f"  - Task: {self.var_task.get()}")
        self.log(f"  - Output: {self.var_output.get()}\n")

    def log(self, message):
        """Hàm ghi log an toàn vào Text Widget, tự động cuộn xuống dưới cùng."""
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END)

    def run_command_in_thread(self, cmd, env=None):
        """Khởi tạo một luồng (thread) chạy subprocess để không làm đơ giao diện Tkinter."""
        if self.process and self.process.poll() is None:
            self.log("⚠️ HỆ THỐNG ĐANG BẬN! Vui lòng chờ tiến trình hiện tại kết thúc.")
            return
            
        self.log(f"\n{'='*60}")
        self.log(f"▶ ĐANG THỰC THI LỆNH: {' '.join(cmd)}")
        self.log(f"{'='*60}\n")
        
        def target():
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                env=env
            )
            
            # Đọc từng dòng output từ stdout và đẩy lên GUI
            for line in iter(self.process.stdout.readline, ''):
                if line:
                    # Dùng root.after để cập nhật GUI một cách an toàn từ thread phụ
                    self.root.after(0, self.log, line.strip())
            
            self.process.stdout.close()
            return_code = self.process.wait()
            
            if return_code == 0:
                self.root.after(0, self.log, "\n✅ TIẾN TRÌNH HOÀN TẤT THÀNH CÔNG!")
            else:
                self.root.after(0, self.log, f"\n❌ LỖI NGHIÊM TRỌNG! Tiến trình thoát với mã lỗi: {return_code}")
                
        # Khởi chạy luồng ẩn (daemon=True để khi tắt app thì luồng cũng chết theo)
        threading.Thread(target=target, daemon=True).start()

    def run_script(self, script_path):
        self.save_paths_to_json() # Lưu cấu hình hiện tại trước khi chạy
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["VIDEO_DIR"] = os.path.abspath(self.var_video.get())
        env["TASK_FILE_PATH"] = os.path.abspath(self.var_task.get())
        env["SUBMISSION_FILE_PATH"] = os.path.abspath(self.var_output.get())
        self.run_command_in_thread([sys.executable, script_path], env=env)
        
    def run_full_pipeline(self):
        self.save_paths_to_json() # Lưu cấu hình hiện tại trước khi chạy
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        
        cmd = [
            sys.executable, "main.py",
            "--video_dir", self.var_video.get(),
            "--task_file", self.var_task.get(),
            "--output", self.var_output.get()
        ]
        self.run_command_in_thread(cmd, env=env)

if __name__ == "__main__":
    root = tk.Tk()
    app = TempoRunDashboard(root)
    root.mainloop()
