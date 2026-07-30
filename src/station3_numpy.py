import os
import sys
import json
import numpy as np
import argparse

def run_station3(chunk_id):
    print(f"🕸️ Bắt đầu Trạm 3: Lưới Thời Gian Numpy (Mẻ {chunk_id})...")
    
    # 1. Xác định danh sách query cho chunk này
    queries_path = "data/queries.json"
    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)
        
    chunk_size = 25
    idx = int(chunk_id) - 1
    start_idx = idx * chunk_size
    end_idx = start_idx + chunk_size
    my_queries = queries[start_idx:end_idx]
    
    if not my_queries:
        print("⚠️ Không có câu hỏi nào trong mẻ này.")
        return
        
    print(f"  Xử lý {len(my_queries)} câu hỏi (Từ {start_idx} đến {end_idx - 1})...")
    
    # 2. Đọc Kho Thịt và Sổ Địa Chỉ
    # Dữ liệu thực tế được nén sẵn bởi Kaggle/Server
    possible_dirs = [
        "data/embeddings_1280",
        "data/embeddings",
        "data"
    ]
    
    kho_thit_path = None
    so_dia_chi_path = None
    
    for d in possible_dirs:
        k_path = f"{d}/all_embeddings.npy"
        s_path = f"{d}/metadata.json"
        if os.path.exists(k_path) and os.path.exists(s_path):
            kho_thit_path = k_path
            so_dia_chi_path = s_path
            break
            
    if not kho_thit_path:
        print(f"❌ LỖI: Không tìm thấy Kho thịt (all_embeddings.npy) hoặc Sổ địa chỉ (metadata.json) tại data/ hoặc các thư mục con.")
        print("💡 GỢI Ý: Chế độ giả lập (Mock mode) đang được bật vì thiếu dữ liệu thực tế.")
        mock_mode = True
    else:
        mock_mode = False
        print(f"  Đang nạp Kho thịt {kho_thit_path} (mmap_mode='r')...")
        kho_thit = np.load(kho_thit_path, mmap_mode='r')
        with open(so_dia_chi_path, "r", encoding="utf-8") as f:
            so_dia_chi = json.load(f)
            
    # 3. Đọc Vector (Lưu ý Tuyệt kỹ .item() từ Cố vấn)
    vec_path = "data/query_vectors.npy"
    if not os.path.exists(vec_path):
        print(f"❌ LỖI: Không tìm thấy {vec_path}.")
        sys.exit(1)
        
    # TUYỆT KỸ: Sử dụng .item() để trích xuất lại cấu trúc Dictionary
    vectors_dict = np.load(vec_path, allow_pickle=True).item()
    
    top100_results = {}
    
    for item in my_queries:
        if isinstance(item, dict):
            q_id = str(item.get("id"))
        else:
            q_id = str(queries.index(item)) # Cách cẩu thả nếu list string
            
        vec_data = vectors_dict.get(q_id)
        if not vec_data:
            continue
            
        main_vec = vec_data["main_shot_vec"]
        before_vecs = vec_data["context_before_vecs"]
        after_vecs = vec_data["context_after_vecs"]
        ocr_texts = vec_data["ocr_texts"]
        
        # --- THUẬT TOÁN TÌM KIẾM TRƯỢT (SLIDING WINDOW) ---
        if mock_mode:
            # Giả lập kết quả Top 100
            import random
            top100 = []
            for i in range(100):
                tar_name = random.choice(["dataset/part1.tar", "dataset/part2.tar"])
                top100.append({
                    "video": "vid_mock",
                    "frame": i * 10,
                    "temporal_score": random.uniform(0.5, 0.9),
                    "tar_name": tar_name,
                    "ocr_texts": ocr_texts,  # Pass sang Trạm 4
                    "q_id": q_id
                })
            top100_results[q_id] = top100
        else:
            # 1. Tính điểm Cảnh Chính (Điểm Gốc & Ngưỡng Cắt Tử Thần)
            TAU = 0.24  # Tăng từ 0.20 lên 0.24 để lọc khắt khe hơn
            score_main = np.dot(kho_thit, main_vec)
            score_safe = np.maximum(score_main - TAU, 0.0)
            
            # 2. Hàm Mũ Khuếch Đại (Power Scaling)
            score_core = np.square(score_safe)
            
            max_before = np.zeros_like(score_main)
            max_after = np.zeros_like(score_main)
            
            # 3. Tính điểm Cảnh Trước (Max Pooling với Wrap-around fix)
            if before_vecs:
                before_vec = before_vecs[0]
                score_before_raw = np.dot(kho_thit, before_vec)
                
                # Tuyệt kỹ Cuốn chiếu (In-place np.maximum)
                max_before = np.roll(score_before_raw, 1)
                max_before[:1] = 0.0
                
                for shift in range(2, 7):
                    rolled = np.roll(score_before_raw, shift)
                    rolled[:shift] = 0.0
                    np.maximum(max_before, rolled, out=max_before)
                
            # 4. Tính điểm Cảnh Sau (Max Pooling với Wrap-around fix)
            if after_vecs:
                after_vec = after_vecs[0]
                score_after_raw = np.dot(kho_thit, after_vec)
                
                # Tuyệt kỹ Cuốn chiếu (In-place np.maximum)
                max_after = np.roll(score_after_raw, -1)
                max_after[-1:] = 0.0
                
                for shift in range(2, 7):
                    rolled = np.roll(score_after_raw, -shift)
                    rolled[-shift:] = 0.0
                    np.maximum(max_after, rolled, out=max_after)
                
            # 5. Công thức Chốt (Soft Fusion V.2)
            max_before_safe = np.maximum(max_before - TAU, 0.0)
            max_after_safe = np.maximum(max_after - TAU, 0.0)
            
            # Giảm bonus từ 0.15 xuống 0.10 để khắt khe hơn với Cảnh Chính
            ALPHA = 0.10
            BETA = 0.10
            bonus_temp = 1.0 + (ALPHA * max_before_safe) + (BETA * max_after_safe)
            
            final_score = score_core * bonus_temp
            
            # 6. Sắp xếp và trích xuất Top 500
            top100_idx = np.argsort(final_score)[-500:][::-1]
            
            top100 = []
            for idx_val in top100_idx:
                # Đảm bảo so_dia_chi có dữ liệu tại idx_val
                if idx_val < len(so_dia_chi):
                    meta = so_dia_chi[idx_val]
                    if isinstance(meta, dict):
                        tar_name = meta.get("tar_name", "")
                        vid = meta.get("video_id", meta.get("video", "unknown"))
                        frm = meta.get("frame_ms", meta.get("frame", 0))
                    elif isinstance(meta, list) and len(meta) >= 2:
                        tar_name = ""
                        vid = str(meta[0])
                        frm_str = str(meta[1])
                        try:
                            frm = int(frm_str.replace("frame_", "").replace(".jpg", ""))
                        except ValueError:
                            frm = 0
                    else:
                        tar_name = ""
                        vid = "unknown"
                        frm = 0
                else:
                    tar_name = ""
                    vid = "unknown"
                    frm = 0
                
                top100.append({
                    "video": vid,
                    "frame": frm,
                    "temporal_score": float(final_score[idx_val]),
                    "tar_name": tar_name,
                    "ocr_texts": ocr_texts,
                    "q_id": q_id
                })
            
            top100_results[q_id] = top100
            
    # Ghi kết quả
    out_path = f"data/top500_candidates_chunk{chunk_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(top100_results, f, indent=4)
        
    print(f"✅ Hoàn tất Trạm 3 (Mẻ {chunk_id})! Đã lưu top 500 ứng viên tại {out_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=str, required=True, help="ID của mẻ (Ví dụ: 01)")
    args = parser.parse_args()
    
    run_station3(args.chunk)
