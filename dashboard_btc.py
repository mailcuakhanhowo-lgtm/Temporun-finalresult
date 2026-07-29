import tkinter as tk
from tkinter import scrolledtext, font
import subprocess
import threading
import sys
import os

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
            "LƯU Ý: Giao diện này sẽ chạy pipeline bằng các ĐƯỜNG DẪN MẶC ĐỊNH (data/videos, data/private_round_tasks.jsonl).\n"
            "Nếu BTC muốn chỉ định đường dẫn tùy chỉnh tuyệt đối trên máy, vui lòng sử dụng lệnh Terminal như trong README."
        )
        tk.Label(root, text=warn_text, fg="red", bg="#f0f0f0", font=("Arial", 10, "bold"), wraplength=850, justify="center").pack(pady=(0, 15))
        
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
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        self.run_command_in_thread([sys.executable, script_path], env=env)
        
    def run_full_pipeline(self):
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        # Bơm các đường dẫn mặc định vào biến môi trường để config.py nhận diện
        env["VIDEO_DIR"] = os.path.abspath(DEFAULT_VIDEO_DIR)
        env["TASK_FILE_PATH"] = os.path.abspath(DEFAULT_TASK_FILE)
        env["SUBMISSION_FILE_PATH"] = os.path.abspath(DEFAULT_OUTPUT)
        
        cmd = [
            sys.executable, "main.py",
            "--video_dir", DEFAULT_VIDEO_DIR,
            "--task_file", DEFAULT_TASK_FILE,
            "--output", DEFAULT_OUTPUT
        ]
        self.run_command_in_thread(cmd, env=env)

if __name__ == "__main__":
    root = tk.Tk()
    app = TempoRunDashboard(root)
    root.mainloop()
