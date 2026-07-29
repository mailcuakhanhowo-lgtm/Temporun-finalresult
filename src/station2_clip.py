import os
import sys
import json
import torch
import numpy as np
from src import config

# Giả lập import open_clip (Trong thực tế cần cài đặt: pip install open-clip-torch)
try:
    import open_clip
except ImportError:
    print("❌ LỖI: Chưa cài open_clip. Vui lòng chạy: pip install open-clip-torch")
    sys.exit(1)

def run_station2():
    vision_eco = os.environ.get("FORCE_VISION_ECOSYSTEM", config.VISION_ECOSYSTEM)
    if vision_eco == "1280D":
        target_model = "ViT-bigG-14"
        target_pretrained = "laion2b_s39b_b160k"
    else:
        target_model = config.MODEL_NAME
        target_pretrained = config.MODEL_PRETRAINED
        
    print(f"🏭 Bắt đầu Trạm 2: Xưởng Đúc Đạn CLIP ({target_model})...")
    
    parsed_path = "data/parsed_queries.json"
    if not os.path.exists(parsed_path):
        print(f"❌ LỖI: Không tìm thấy {parsed_path}. Bạn đã chạy Trạm 1 chưa?")
        sys.exit(1)
        
    with open(parsed_path, "r", encoding="utf-8") as f:
        parsed_queries = json.load(f)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Trục xuất Ollama khỏi VRAM để nhường chỗ cho CLIP
    import time
    import requests
    print("  🧹 Đang dọn dẹp VRAM (Đuổi Ollama ra khỏi bộ nhớ)...")
    try:
        # Gửi request rỗng với keep_alive=0 để ép Ollama nhả VRAM
        for m in ["qwen2.5:3b", "llama3.2:3b", "gemma2:2b", config.OLLAMA_MODEL]:
            requests.post("http://localhost:11434/api/generate", json={"model": m, "keep_alive": 0}, timeout=2)
    except:
        pass
        
    print("  ⏳ Chờ 3 giây để Windows thu hồi RAM/VRAM từ Ollama...")
    time.sleep(3)
        
    print(f"  Đang nạp mô hình CLIP {target_model} vào RAM hệ thống (CPU)...")
    
    # --- CHIẾN THUẬT ÉP KIỂU NGUỘI VÀ TRÁNH 20GB RAM SPIKE ---
    import gc
    gc.collect()
    
    # BẢO KIẾM 1: MONKEY PATCHING (Mmap=True)
    # BẮT BUỘC phải có mmap=True để PyTorch không kéo 10GB file vào RAM vật lý
    original_torch_load = torch.load
    def mmap_torch_load(*args, **kwargs):
        kwargs['mmap'] = True
        return original_torch_load(*args, **kwargs)
    torch.load = mmap_torch_load
    
    try:
        # BƯỚC 1 & 2: Tạo model (5GB) và nạp weights bằng mmap + precision="fp16"
        model, _, _ = open_clip.create_model_and_transforms(
            target_model, 
            pretrained=target_pretrained, 
            device="cpu",
            precision="fp16"
        )
    finally:
        torch.load = original_torch_load
        
    tokenizer = open_clip.get_tokenizer(target_model)
    
    model.eval()
    
    # BƯỚC 2.5: Gọi Đội Dọn Rác dọn sạch xác FP32 (nếu có) để RAM thông thoáng nhất có thể
    gc.collect()
    
    if device == "cuda":
        print(f"  Đang bốc mô hình 5GB lọt thỏm vào VRAM GPU...")
        model = model.to(device)
        
    vectors_dict = {}
    
    print(f"  Đang đúc Vector cho {len(parsed_queries)} câu hỏi...")
    with torch.no_grad():
        for q_id, q_data in parsed_queries.items():
            # Xử lý Main Shot
            main_text = q_data.get("original_query", q_data.get("main_shot", ""))
            main_token = tokenizer([main_text]).to(device)
            main_vec = model.encode_text(main_token).cpu().float().numpy()[0]
            # Normalize vector
            main_vec = main_vec / np.linalg.norm(main_vec)
            
            # Xử lý Before
            before_vecs = []
            for text in q_data.get("context_before", []):
                t = tokenizer([text]).to(device)
                v = model.encode_text(t).cpu().float().numpy()[0]
                before_vecs.append(v / np.linalg.norm(v))
                
            # Xử lý After
            after_vecs = []
            for text in q_data.get("context_after", []):
                t = tokenizer([text]).to(device)
                v = model.encode_text(t).cpu().float().numpy()[0]
                after_vecs.append(v / np.linalg.norm(v))
                
            vectors_dict[q_id] = {
                "main_shot_vec": main_vec,
                "context_before_vecs": before_vecs,
                "context_after_vecs": after_vecs,
                "ocr_texts": q_data.get("ocr_texts", []) # Giữ lại để Trạm 4 dùng
            }
            
    out_path = "data/query_vectors.npy"
    # LƯU Ý TỪ CỐ VẤN: Sử dụng allow_pickle=True để lưu Dictionary
    np.save(out_path, vectors_dict, allow_pickle=True)
    
    print(f"✅ Hoàn tất Trạm 2! Đã lưu ma trận Vector 1280D tại {out_path}.")
    print("🧹 Đang giải phóng CLIP khỏi VRAM...")
    
    # Ép HĐH giải phóng bộ nhớ
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    try:
        run_station2()
    except Exception as e:
        print(f"❌ [LỖI NGHIÊM TRỌNG] Trạm 2 bốc khói: {str(e)}")
        sys.exit(1)
    finally:
        print("🧹 [Bảo vệ 2 Lớp] Kích hoạt xả rác toàn hệ thống (RAM/VRAM)...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        print("✅ [An Toàn] Tiến trình Trạm 2 đã tự hủy, trả lại 100% tài nguyên cho HĐH!")
