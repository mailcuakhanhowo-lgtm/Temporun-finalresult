import os
import sys
import json
import argparse
import numpy as np
import cv2

# Giả lập import easyocr và thefuzz
try:
    import easyocr
    from thefuzz import fuzz
except ImportError:
    print("❌ LỖI: Chưa cài thư viện. Vui lòng chạy: pip install easyocr thefuzz python-Levenshtein")
    sys.exit(1)

def run_station4(chunk_id):
    print(f"🗡️ Bắt đầu Trạm 4: Sát Thủ Đột Kích OCR (Mẻ {chunk_id})...")
    
    top100_path = f"data/top500_candidates_chunk{chunk_id}.json"
    tar_index_path = "data/tar_index.json"
    
    if not os.path.exists(top100_path):
        print(f"❌ LỖI: Không tìm thấy {top100_path}")
        sys.exit(1)
        
    with open(top100_path, "r", encoding="utf-8") as f:
        top100_results = json.load(f)
        
    if not os.path.exists(tar_index_path):
        print(f"⚠️ CẢNH BÁO: Không tìm thấy {tar_index_path}. Bỏ qua nạp ảnh thực tế (Mock mode).")
        tar_index = {}
    else:
        with open(tar_index_path, "r", encoding="utf-8") as f:
            tar_index = json.load(f)
            
    print("  Đang nạp mô hình EasyOCR (English)...")
    # Tắt logging dài dòng của easyocr
    reader = easyocr.Reader(['en'], gpu=True, verbose=False)
    
    final_submission = []
    
    # --- TỐI ƯU I/O ---
    # Thay vì đọc file tar, đọc trực tiếp ảnh .webp từ thư mục frames (Local Pipeline)
    from src import config
    
    flat_candidates = []
    
    for q_id, candidates in top100_results.items():
        for cand in candidates:
            # frame_ms thay vì tên file.
            video_id = cand['video']
            frame_ms = cand['frame']
            
            # Khôi phục đường dẫn ảnh từ video_id và frame_ms
            # Format chuẩn của GĐ1: frame_00000266.webp (8 chữ số)
            img_path = os.path.join(config.FRAME_DIR, video_id, f"frame_{int(frame_ms):08d}.{config.IMAGE_FORMAT}")
            
            flat_candidates.append({
                "q_id": q_id,
                "cand": cand,
                "img_path": img_path,
                "ocr_texts": cand.get("ocr_texts", [])
            })
            
    print("  Đang nạp ảnh thần tốc từ ổ cứng (Local Disk)...")
    extracted_images = {} # Mapping từ ID nội bộ sang ảnh
    
    for i, item in enumerate(flat_candidates):
        img_np = None
        if os.path.exists(item["img_path"]):
            img_np = cv2.imread(item["img_path"], cv2.IMREAD_COLOR)
            
        extracted_images[i] = img_np
        item["extracted_idx"] = i
        
    # --- KÍCH HOẠT BATCH PROCESSING CHO OCR ---
    print("  Đang đưa ảnh vào máy chém EasyOCR (Batch Processing = 16)...")
    
    # Chạy OCR theo từng query để dễ map kết quả
    for q_id, candidates in top100_results.items():
        # Lấy lại danh sách ảnh cho query này theo trật tự gốc (Temporal Rank)
        my_cands = [c for c in flat_candidates if c["q_id"] == q_id]
        
        # Lọc ra những ảnh thực sự tồn tại
        valid_cands = [c for c in my_cands if extracted_images[c["extracted_idx"]] is not None]
        images_to_ocr = [extracted_images[c["extracted_idx"]] for c in valid_cands]
        
        target_ocr_words = my_cands[0]["ocr_texts"] if my_cands else []
        
        if not target_ocr_words or not images_to_ocr:
            # Fallback nếu không có chữ cần tìm hoặc không có ảnh
            results_list = []
            for i, c in enumerate(candidates[:10]):
                results_list.append({
                    "rank": i + 1,
                    "video_id": c["video"],
                    "frame_ms": int(c["frame"])
                })
            
            final_submission.append({
                "task_id": q_id,
                "results": results_list if results_list else [{"rank": 1, "video_id": "", "frame_ms": 0}]
            })
            continue
            
        # Đưa vào EasyOCR dạng Batched
        # readtext_batched trả về list các kết quả, mỗi kết quả là list các tuple (bbox, text, prob)
        ocr_results = reader.readtext_batched(images_to_ocr, batch_size=16)
        
        scored_cands = []
        for cand_idx, res in enumerate(ocr_results):
            cand_info = valid_cands[cand_idx]
            found_texts = [box[1].lower() for box in res]
            
            # Đếm số chữ khớp bằng Sliding Fuzzy Buff
            total_ocr_bonus = 0.0
            
            for target in target_ocr_words:
                best_ratio = 0.0
                target_lower = target.lower()
                for ft in found_texts:
                    ratio = fuzz.partial_ratio(target_lower, ft) / 100.0
                    if ratio > best_ratio:
                        best_ratio = ratio
                
                # Chỉ cộng điểm nếu tự tin > 90% (Khắt khe hơn, trước là 80%)
                if best_ratio >= 0.9:
                    total_ocr_bonus += 0.10 * best_ratio  # Giảm mức độ buff
                    
            # Chốt an toàn: Khống chế tối đa buff 2 từ (20% MAX) để chống lạm phát
            total_ocr_bonus = min(total_ocr_bonus, 0.20)
            
            # Tính Final Score
            temporal_score = cand_info["cand"]["temporal_score"]
            final_score = temporal_score * (1.0 + total_ocr_bonus)
            
            scored_cands.append({
                "cand": cand_info["cand"],
                "final_score": final_score
            })
            
        # Sắp xếp theo Final_Score giảm dần
        scored_cands.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Lưu Top 10 kết quả (hoặc ít hơn nếu không đủ)
        results_list = []
        for i, sc in enumerate(scored_cands[:10]):
            results_list.append({
                "rank": i + 1,
                "video_id": sc["cand"]["video"],
                "frame_ms": int(sc["cand"]["frame"])
            })
            
        final_submission.append({
            "task_id": q_id,
            "results": results_list
        })
        
    # Lưu file submission
    out_path = f"data/submission_part_{chunk_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_submission, f, indent=4)
        
    print(f"✅ Hoàn tất Trạm 4 (Mẻ {chunk_id})! Đã ghi File Submission Part.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=str, required=True)
    args = parser.parse_args()
    
    run_station4(args.chunk)
