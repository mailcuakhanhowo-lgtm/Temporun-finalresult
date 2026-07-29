import os
import subprocess
import time
import json
import glob
import sys

def log(msg):
    print(msg)

def run_subprocess(cmd, desc):
    log(f"⏳ Bắt đầu: {desc}")
    log(f"> Lệnh: {' '.join(cmd)}")
    
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    
    process = subprocess.Popen(
        cmd, 
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
            log(f"  {line}")
            
    process.stdout.close()
    return_code = process.wait()
    
    if return_code != 0:
        log(f"❌ LỖI NGHIÊM TRỌNG tại: {desc}. Mã lỗi: {return_code}")
        raise Exception(f"Subprocess failed: {desc}")
    else:
        log(f"✅ Hoàn tất: {desc}")

def merge_results():
    log("🔄 Đang gộp kết quả từ các mẻ...")
    all_results = []
    chunk_files = sorted(glob.glob("data/submission_part_*.json"))
    
    for f in chunk_files:
        try:
            with open(f, "r", encoding="utf-8") as file:
                data = json.load(file)
                all_results.extend(data)
        except Exception as e:
            log(f"❌ Lỗi đọc file {f}: {e}")
            
    from src import config
    out_path = config.SUBMISSION_FILE_PATH
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"predictions": all_results}, f, indent=4)
        
    log(f"✅ Đã lưu kết quả chung cuộc tại: {out_path} (Tổng số câu: {len(all_results)})")

def run_pipeline():
    print("=" * 60)
    print("🚀 BẮT ĐẦU TRẠM 3 & 4 (HẬU KỲ KAGGLE BYPASS)")
    print("=" * 60)
    
    queries_path = "data/queries.json"
    from src import config
    jsonl_path = config.TASK_FILE_PATH
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            queries = [{"id": json.loads(line)["task_id"], "query": json.loads(line)["description"]} for line in f if line.strip()]
        with open(queries_path, "w", encoding="utf-8") as f:
            json.dump(queries, f, indent=4, ensure_ascii=False)
        log(f"✅ Đã tự động chuyển đổi {len(queries)} câu hỏi từ {os.path.basename(jsonl_path)}!")
    else:
        log(f"❌ Không tìm thấy {os.path.basename(jsonl_path)}. Vui lòng chuẩn bị file đề thi.")
        return
            
    total_queries = len(queries)
    log(f"📊 Tổng số câu hỏi: {total_queries}")
    
    # Check if query_vectors.npy exists
    if not os.path.exists("data/query_vectors.npy"):
        log("❌ LỖI NGHIÊM TRỌNG: Không tìm thấy data/query_vectors.npy!")
        log("👉 Bạn phải tải file này từ Kaggle về và đặt vào thư mục data/ trước khi chạy Trạm 3!")
        return

    chunk_size = 25
    chunks = [queries[i:i + chunk_size] for i in range(0, total_queries, chunk_size)]
    total_chunks = len(chunks)
    
    log(f"📦 Đã chia {total_queries} câu thành {total_chunks} mẻ (Mỗi mẻ {chunk_size} câu).")
    
    for i in range(total_chunks):
        chunk_id = str(i + 1).zfill(2)
        log(f"🔥 BẮT ĐẦU MẺ {chunk_id}/{total_chunks}")
        
        # Trạm 3: Lưới Thời Gian Numpy
        run_subprocess([sys.executable, "src/station3_numpy.py", "--chunk", chunk_id], f"Trạm 3 - Numpy Sliding Window (Mẻ {chunk_id})")
        
        # Trạm 4: OCR
        run_subprocess([sys.executable, "src/station4_ocr.py", "--chunk", chunk_id], f"Trạm 4 - OCR Cắt Tỉa (Mẻ {chunk_id})")
        
        if i < total_chunks - 1:
            cooldown_time = 30
            log(f"❄️ Đang tản nhiệt GPU ({cooldown_time} giây) để chống Thermal Throttling...")
            for s in range(cooldown_time, 0, -1):
                if s % 5 == 0:
                    log(f"   Còn {s} giây...")
                time.sleep(1)
                
    # Gộp kết quả
    merge_results()
    print("EOF")

if __name__ == "__main__":
    run_pipeline()
