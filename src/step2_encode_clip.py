# src/step2_encode_clip.py
# ============================================================
# GIAI ĐOẠN 2 – MÃ HÓA CLIP → VECTOR
# ============================================================
# Nhiệm vụ: Quét toàn bộ ảnh .webp ở data/frames/,
#            dùng AI (CLIP) mã hóa thành vector số học,
#            xuất ra 2 file:
#              - all_embeddings.npy  (Kho Thịt – ma trận vector)
#              - metadata.json      (Sổ Địa Chỉ – ánh xạ hàng → video + timestamp)
# ============================================================

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import sys
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import open_clip          # Thư viện CHÍNH – gọn nhẹ, float16 siêu tốt
from PIL import Image
from tqdm import tqdm
from src import config


# ============================================================
# TẦNG 1 – FrameDataset: Kho hàng có Catalog
# ============================================================
# Đây là class kế thừa từ Dataset của PyTorch.
# Nó KHÔNG đọc ảnh ngay. Chỉ ghi nhớ danh sách đường dẫn.
# Khi DataLoader gọi __getitem__, 4 luồng CPU song song sẽ
# tự đọc ảnh từ ổ cứng → GPU không bao giờ ngồi chờ.
# ============================================================

class FrameDataset(Dataset):
    """
    Custom Dataset cho ảnh frame đã trích xuất từ video.
    DataLoader sẽ gọi __getitem__ từ nhiều luồng song song (num_workers=4).
    """

    def __init__(self, image_paths, preprocess_fn):
        """
        Khởi tạo dataset.

        Đầu vào:
            image_paths  : list[str] – danh sách đường dẫn tuyệt đối đến các file ảnh
            preprocess_fn: hàm transform của open_clip – biến ảnh PIL thành Tensor
        """
        self.image_paths = image_paths
        self.preprocess = preprocess_fn

    def __len__(self):
        """Trả về tổng số ảnh. DataLoader cần biết con số này để chia batch."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Đọc 1 ảnh tại vị trí idx, biến thành Tensor.

        BẮT BUỘC dùng .convert("RGB") vì:
        - Một số ảnh có thể là grayscale (1 kênh) hoặc RGBA (4 kênh).
        - CLIP yêu cầu đúng 3 kênh RGB. Thiếu convert → lỗi shape mismatch.
        """
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.preprocess(img)  # Trả về Tensor shape [3, H, W]


# ============================================================
# HÀM TẢI MODEL AI
# ============================================================

def load_model():
    """
    Tải mô hình AI (CLIP hoặc SigLIP) dựa trên config.MODEL_NAME.
    Áp dụng chiến lược Lazy Installation: chỉ import thư viện nặng
    khi thực sự cần, và báo lỗi rõ ràng nếu thiếu.

    Đầu ra:
        model         : Mô hình AI đã sẵn sàng chạy trên GPU (float16)
        preprocess_fn : Hàm biến ảnh PIL → Tensor chuẩn cho model
        device        : "cuda" hoặc "cpu"
    """
    # Bước 1 – Xác định thiết bị tính toán (GPU hay CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dang dung: {device.upper()}")

    if device == "cpu":
        print("CANH BAO: Khong phat hien GPU! Chay tren CPU se CUC KY CHAM.")

    # Bước 2 – Tải model
    model_name = config.MODEL_NAME
    model_pretrained = config.MODEL_PRETRAINED

    # NẾU Ở CHẾ ĐỘ 1280D: Ghi đè cứng (Override) model mà không làm hỏng config cũ của CUSTOM
    is_1280d = getattr(config, "VISION_ECOSYSTEM", "CUSTOM") == "1280D"
    if is_1280d:
        model_name = "ViT-bigG-14"
        model_pretrained = "laion2b_s39b_b160k"
        print("\n" + "="*60)
        print("🔥 CHẾ ĐỘ HỦY DIỆT (1280D) ĐANG KÍCH HOẠT 🔥")
        print("CẢNH BÁO: Hệ thống chuẩn bị load ViT-bigG-14 (12GB+ RAM).")
        print("          Hãy đảm bảo ngài đã Set Pagefile tối thiểu 30GB!")
        print("="*60 + "\n")

    print(f"Dang tai model: {model_name} (pretrained: {model_pretrained})...")

    if "SigLIP" in model_name or "siglip" in model_name.lower():
        # ─── TRƯỜNG HỢP B: SigLIP (Lazy Installation) ───
        # Chỉ import thư viện nặng khi thực sự cần.
        # Nếu chưa cài → báo lỗi rõ ràng cho Ông Chủ, không crash mù quáng.
        try:
            import timm            # noqa: F401
        except ImportError:
            print("SigLIP can them thu vien.")
            print("Hay chay: pip install timm sentencepiece")
            sys.exit(1)

    # [SỬA LỖI 0xC0000005] Lưu lại số luồng gốc của CPU
    original_threads = torch.get_num_threads()
    
    # CHỈ ép PyTorch dùng 1 luồng TẠM THỜI khi load ViT-bigG-14 (1280D)
    if is_1280d:
        torch.set_num_threads(1)
    
    print(f">> [DEBUG] Truoc khi tao model_and_transforms ({model_name})...", flush=True)
    import sys
    sys.stdout.flush()
    try:
        if is_1280d:
            # [CẤP CỨU RAM] Ép open_clip nạp thẳng ViT-bigG ở dạng thu gọn (fp16) 
            # để chặn đứng đỉnh sốc RAM ~10GB.
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=model_pretrained,
                precision="fp16"
            )
        else:
            # Hệ thống cũ: Giữ nguyên mặc định (fp32) để không ảnh hưởng độ chính xác
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=model_pretrained
            )
    except Exception as e:
        print(f"LỖI KHI LOAD MODEL: {e}")
        if is_1280d:
            torch.set_num_threads(original_threads)
        sys.exit(1)
        
    # Trả lại toàn bộ số luồng gốc cho hệ thống nếu đang ở chế độ 1280D
    if is_1280d:
        torch.set_num_threads(original_threads)
    
    print(">> [DEBUG] Da tao model tren RAM thanh cong!", flush=True)
    sys.stdout.flush()

    # ĐÃ XÓA TUYỆT KỸ CẤP CỨU RAM (model.visual = None) vì Giai đoạn 2 cần dùng Tháp Thị Giác để mã hóa ảnh.

    print(f">> [DEBUG] Truoc khi day model len {device.upper()}...", flush=True)
    model = model.to(device)
    print(f">> [DEBUG] Da day len {device.upper()} thanh cong!", flush=True)
    
    print(">> [DEBUG] Truoc khi convert sang .half()...", flush=True)
    # Thu convert float16 (nhieu GPU cu tren Windows bi crash ở đây)
    model = model.half()
    print(">> [DEBUG] Da convert .half() thanh cong!", flush=True)

    # .eval() = tắt chế độ training
    model.eval()

    print(f"Model da san sang tren {device.upper()} (float16)")
    return model, preprocess, device


# ============================================================
# HÀM MÃ HÓA TOÀN BỘ ẢNH → VECTOR
# ============================================================

def encode_frames(model, preprocess_fn, device):
    """
    Quét toàn bộ ảnh trong data/frames/, mã hóa thành vector bằng CLIP.
    CPU prefetch liên tục qua DataLoader → GPU không bao giờ ngồi chờ.

    Đầu ra:
        all_vectors   : numpy array 2D, dtype=float32, shape = [N, D]
                        N = tổng số ảnh, D = số chiều vector (ViT-L-14 → D=768)
        metadata_list : list[dict], mỗi phần tử chứa thông tin 1 frame:
                        {"index": 0, "video_id": "...", "timestamp_sec": 5, "frame_file": "..."}
    """
    # Danh sách gom kết quả từ tất cả batch
    all_batch_results = []  
    metadata_list = []      # Sổ Địa Chỉ – ánh xạ từng hàng vector → video + thời gian

    # Bước 1 – Quét danh sách thư mục video
    all_items = sorted(os.listdir(config.FRAME_DIR))
    video_ids = [item for item in all_items if os.path.isdir(os.path.join(config.FRAME_DIR, item))]

    if len(video_ids) == 0:
        print("Khong tim thay thu muc video nao trong data/frames/.")
        return np.array([]), []

    print(f"Tim thay {len(video_ids)} thu muc video.")

    # Bước 2 – KỊCH BẢN BÁN BUÔN (Flatten Dataset): Gom tất cả ảnh thành 1 list duy nhất
    global_index = 0
    all_image_paths = []
    
    for video_id in tqdm(video_ids, desc="Quét danh sách ảnh"):
        video_dir = os.path.join(config.FRAME_DIR, video_id)
        
        # BẮT BUỘC sorted() khi lấy danh sách ảnh
        image_files = sorted([f for f in os.listdir(video_dir) if f.endswith(f".{config.IMAGE_FORMAT}")])
        if not image_files:
            continue
            
        for frame_file in image_files:
            img_path = os.path.join(video_dir, frame_file)
            all_image_paths.append(img_path)
            
            # Ghi sổ địa chỉ (metadata) ngay lúc này để khớp 100% với danh sách đường dẫn
            frame_ms = int(frame_file.split("_")[1].split(".")[0])
            metadata_list.append({
                "index": global_index,
                "video_id": video_id,
                "frame_ms": frame_ms,
                "frame_file": frame_file,
            })
            global_index += 1
            
    if len(all_image_paths) == 0:
        print("CANH BAO: Khong co bat ky anh nao de ma hoa.")
        return np.array([]), []
        
    print(f"Tong cong co {len(all_image_paths)} anh can ma hoa. Khoi dong DataLoader...")

    # Bước 3 – Khởi tạo DataLoader MỘT LẦN DUY NHẤT
    dataset = FrameDataset(all_image_paths, preprocess_fn)
    loader = DataLoader(
        dataset,
        batch_size=config.CLIP_BATCH_SIZE,  # 64 ảnh / batch
        num_workers=4,    # Tuyển đúng 1 đội 4 công nhân dùng cho cả dự án!
        pin_memory=True,  
        shuffle=False,    # TUYỆT ĐỐI không xáo trộn!
    )

    # Bước 4 – Mã hóa toàn bộ ảnh qua GPU (Chạy 1 mạch không nghỉ)
    start_time_total = time.time()
    vectors_processed = 0

    for batch_tensors in tqdm(loader, desc="Mã hóa trên GPU"):
        batch_tensors = batch_tensors.to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                features = model.encode_image(batch_tensors)

        # Chuẩn hóa vector về độ dài = 1 (L2 Normalization)
        features = F.normalize(features, dim=-1)

        # Chuyển kết quả về CPU và sang numpy float32
        batch_np = features.cpu().numpy().astype(np.float32)
        all_batch_results.append(batch_np)
        
        vectors_processed += len(batch_np)
        elapsed = time.time() - start_time_total
        speed = vectors_processed / elapsed if elapsed > 0 else 0
        print(f"[SPEED] {speed:,.0f} vectors/giây", flush=True)

    # Bước 5 – Gộp toàn bộ vector thành 1 ma trận 2D
    all_vectors = np.concatenate(all_batch_results, axis=0)  # shape [N, D]
    return all_vectors, metadata_list


# ============================================================
# HÀM LƯU KHO THỊT + SỔ ĐỊA CHỈ RA Ổ CỨNG
# ============================================================

def save_database(all_vectors, metadata_list):
    """
    Lưu kết quả mã hóa ra ổ cứng.

    QUY TẮC CỨNG (không được vi phạm):
    ✅ Lưu tất cả vector vào MỘT file duy nhất: all_embeddings.npy
    ✅ Lưu metadata vào MỘT file duy nhất: metadata.json
    ❌ KHÔNG tạo file .npy riêng cho từng video (gây nghẽn Disk I/O ở GĐ5)
    """
    # Bước 1 – Tạo thư mục embeddings/ nếu chưa có
    os.makedirs(config.EMBEDDING_DIR, exist_ok=True)

    # Bước 2 – Lưu Kho Thịt (ma trận vector 2D)
    emb_path = os.path.join(config.EMBEDDING_DIR, "all_embeddings.npy")
    np.save(emb_path, all_vectors)

    # Bước 3 – Lưu Sổ Địa Chỉ (metadata)
    # BẮT BUỘC dùng "with open" + encoding="utf-8":
    #   - "with open": đảm bảo file LUÔN được đóng, dù chương trình lỗi giữa chừng
    #   - encoding="utf-8": hỗ trợ tên video tiếng Việt, không bị lỗi ký tự
    #   - ensure_ascii=False: giữ nguyên ký tự Unicode, không chuyển thành \uXXXX
    meta_path = os.path.join(config.EMBEDDING_DIR, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)

    # Bước 4 – In thống kê xác nhận
    print(f"Da luu {all_vectors.shape[0]} vector, shape: {all_vectors.shape}")
    print(f"   Kho Thịt  : {emb_path}  (~{all_vectors.nbytes / 1e9:.2f} GB)")
    print(f"   Sổ Địa Chỉ: {meta_path}")


# ============================================================
# HÀM ĐIỀU PHỐI CHÍNH (Tương đương int main() trong C++)
# ============================================================

def main():
    """
    Điểm khởi động của Giai đoạn 2.
    Gọi lần lượt: tải model → mã hóa ảnh → lưu kết quả.
    """
    # Bước 1 – Bắt đầu đếm thời gian
    start_time = time.time()
    print("=" * 60)
    print("BAT DAU GIAI DOAN 2: MA HOA CLIP")
    print("=" * 60)

    # Bước 2 – Tải model AI lên GPU
    model, preprocess_fn, device = load_model()

    # Bước 3 – Mã hóa toàn bộ ảnh thành vector
    all_vectors, metadata_list = encode_frames(model, preprocess_fn, device)

    # Bước 4 – Kiểm tra kết quả rỗng
    # Nếu data/frames/ không có ảnh nào → dừng lại, không lưu file rỗng
    if len(metadata_list) == 0:
        print("Khong co anh nao de ma hoa. Ket thuc.")
        return

    # Bước 5 – Lưu Kho Thịt + Sổ Địa Chỉ ra ổ cứng
    save_database(all_vectors, metadata_list)

    # Bước 6 – In tổng kết
    elapsed = time.time() - start_time
    print("-" * 60)
    print("HOAN THANH GIAI DOAN 2")
    print(f"   Tổng ảnh đã mã hóa : {len(metadata_list)}")
    print(f"   Kích thước vector   : {all_vectors.shape[1]} chiều")
    print(f"   Thời gian chạy      : {elapsed:.2f} giây")
    print("=" * 60)


if __name__ == "__main__":
    main()
