import os
import sys
import argparse
import subprocess

def run_step(script_path, desc):
    print(f"\n{'='*60}")
    print(f"🚀 BẮT ĐẦU: {desc}")
    print(f"{'='*60}")
    
    # Kế thừa biến môi trường để truyền cấu hình
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    
    process = subprocess.Popen(
        [sys.executable, script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        env=env
    )
    
    for line in iter(process.stdout.readline, ''):
        line = line.strip()
        if line:
            print(f"  {line}")
            
    process.stdout.close()
    return_code = process.wait()
    
    if return_code != 0:
        print(f"\n❌ LỖI NGHIÊM TRỌNG tại: {desc}. Mã lỗi: {return_code}")
        sys.exit(return_code)
    else:
        print(f"✅ HOÀN TẤT: {desc}\n")

def main():
    parser = argparse.ArgumentParser(description="Tempo Run 2026 - Cỗ Xe Tăng Bất Tử (Local Pipeline)")
    parser.add_argument("--video_dir", type=str, required=True, help="Thư mục chứa video (.mp4) đầu vào")
    parser.add_argument("--task_file", type=str, required=True, help="Đường dẫn đến file jsonl chứa câu hỏi")
    parser.add_argument("--output", type=str, required=True, help="Đường dẫn file submission.json đầu ra")
    
    args = parser.parse_args()
    
    # Thiết lập biến môi trường để config.py đọc
    os.environ["VIDEO_DIR"] = os.path.abspath(args.video_dir)
    os.environ["TASK_FILE_PATH"] = os.path.abspath(args.task_file)
    os.environ["SUBMISSION_FILE_PATH"] = os.path.abspath(args.output)
    
    # Chạy lần lượt các trạm từ số 0
    run_step("src/step1_extract_frames.py", "Giai đoạn 1: Trích xuất Frame (OpenCV)")
    run_step("src/step2_encode_clip.py", "Giai đoạn 2: Mã hóa Ảnh thành Vector (CLIP ViT-bigG-14)")
    run_step("src/station1_ollama.py", "Trạm 1: Bóc tách ngữ nghĩa Query (Ollama Llama 3.2)")
    run_step("src/station2_clip.py", "Trạm 2: Đúc Vector Query (CLIP ViT-bigG-14)")
    run_step("src/station34_pipeline.py", "Trạm 3 & 4: Tìm kiếm Numpy (Soft Fusion) & Nhận diện chữ (EasyOCR)")

    print("🎉 TẤT CẢ CÁC GIAI ĐOẠN ĐÃ HOÀN TẤT THÀNH CÔNG!")
    print(f"📁 Kết quả đã được lưu tại: {args.output}")

if __name__ == "__main__":
    main()
