import os
import cv2
import time
import numpy as np
import concurrent.futures
from src import config

def calculate_histogram_difference(frame1, frame2):
    """
    Thuật toán lọc ảnh thế hệ mới: "HYBRID V TỐI THƯỢNG"
    
    Kiến trúc 5 lớp:
    1. Downscale 128x128 → Giảm tải CPU ~500 lần.
    2. Overlapping Grid 3x3 (stride=32, overlap=50%) → Chống mù biên.
    3. Hybrid V (Tắc Kè Hoa):
       - Ban Ngày (Mean V >= 40 VÀ Mean S >= 20): Histogram 2D [H, S] 32x32 bins → Miễn nhiễm đèn flash.
       - Ban Đêm (Mean V < 40 HOẶC Mean S < 20): Histogram 1D [V] 32 bins → Nhìn xuyên màn đêm.
    4. Bhattacharyya So sánh từng cặp ô cùng chế độ → Không Crash OpenCV.
    5. MAX Pooling → Tóm mọi sự kiện cục bộ dù nhỏ nhất.
    
    Đầu vào:
        frame1, frame2: Ảnh dạng numpy array của OpenCV (màu BGR mặc định).
    Đầu ra:
        float: Giá trị khác biệt từ 0.0 (giống hệt) đến 1.0 (khác hoàn toàn).
    """
    # ═══════════════════════════════════════════════════════════
    # BƯỚC 1: DOWNSCALE – Bóp cả 2 ảnh về 128x128 (Bản nháp siêu nhỏ)
    # CPU lúc này chỉ cần xử lý 4.096 điểm ảnh thay vì ~2 triệu. Nhẹ như lông!
    # ═══════════════════════════════════════════════════════════
    small1 = cv2.resize(frame1, (128, 128), interpolation=cv2.INTER_AREA)
    small2 = cv2.resize(frame2, (128, 128), interpolation=cv2.INTER_AREA)

    # ═══════════════════════════════════════════════════════════
    # BƯỚC 2: CHUYỂN HỆ MÀU HSV
    # ═══════════════════════════════════════════════════════════
    hsv1 = cv2.cvtColor(small1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(small2, cv2.COLOR_BGR2HSV)

    # ═══════════════════════════════════════════════════════════
    # BƯỚC 3 & 4: OVERLAPPING GRIDS + HYBRID V
    # Cắt ra 9 ô lưới 64x64 với bước nhảy (stride) 32px → Overlap 50% giữa các ô.
    # Các ô lợp ngói lên nhau, triệt tiêu hoàn toàn lỗi "giật mình biên giới".
    # ═══════════════════════════════════════════════════════════
    CELL_SIZE = 64   # Kích thước mỗi ô lưới
    STRIDE    = 32   # Bước nhảy (stride = CELL_SIZE / 2 = Overlap 50%)
    
    max_diff = 0.0   # Biến chứa độ lệch LỚN NHẤT trong số 9 ô (MAX Pooling)
    
    # Duyệt qua 3×3 = 9 ô lưới (tọa độ góc trên bên trái của từng ô)
    for row_start in range(0, 128 - CELL_SIZE + 1, STRIDE):  # 0, 32, 64
        for col_start in range(0, 128 - CELL_SIZE + 1, STRIDE):  # 0, 32, 64
            # Cắt ô lưới từ bản nháp
            row_end = row_start + CELL_SIZE
            col_end = col_start + CELL_SIZE
            
            cell1 = hsv1[row_start:row_end, col_start:col_end]
            cell2 = hsv2[row_start:row_end, col_start:col_end]

            # ─────────────────────────────────────────────────────────
            # HYBRID V: Kiểm tra điều kiện ánh sáng dựa trên ảnh MỚI (frame2)
            # Dùng cell2 (ảnh mới) để quyết định chế độ → CẢ 2 ô tính cùng 1 chế độ
            # → Luôn đảm bảo hist1 và hist2 có cùng hình dạng ma trận → Không Crash!
            # ─────────────────────────────────────────────────────────
            mean_v = np.mean(cell2[:, :, 2])  # Kênh V = kênh số 2 trong HSV
            mean_s = np.mean(cell2[:, :, 1])  # Kênh S = kênh số 1 trong HSV
            
            night_mode = (mean_v < config.HYBRID_V_THRESHOLD or
                          mean_s < config.HYBRID_S_THRESHOLD)

            if night_mode:
                # ─── CHẾ ĐỘ BAN ĐÊM / ĐEN TRẮNG ───────────────────
                # Đo Histogram 1D trên Kênh V (Độ sáng): 32 bins, dải [0, 256]
                # → Tên trộm áo đen trong bóng tối sẽ bị bắt ngay!
                hist1 = cv2.calcHist([cell1], [2], None, [32], [0, 256])
                hist2 = cv2.calcHist([cell2], [2], None, [32], [0, 256])
            else:
                # ─── CHẾ ĐỘ BAN NGÀY ───────────────────────────────
                # Đo Histogram 2D trên Kênh H và S: 32x32 bins
                # Cắt đứt Kênh V (Độ sáng) → Tự động miễn nhiễm đèn flash!
                # Lý do 32x32=1024 bins: ô 64x64=4096 pixels → 4 pixel/bin → đặc ruột!
                # (Nếu dùng 50x50=2500 bins → 1.6 pixel/bin → ma trận rỗng → Bhattacharyya ảo giác)
                hist1 = cv2.calcHist([cell1], [0, 1], None, [32, 32],
                                     [0, 180, 0, 256])
                hist2 = cv2.calcHist([cell2], [0, 1], None, [32, 32],
                                     [0, 180, 0, 256])

            # ─────────────────────────────────────────────────────────
            # BƯỚC 5: CHUẨN HÓA VÀ SO SÁNH BHATTACHARYYA
            # ─────────────────────────────────────────────────────────
            cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            
            diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
            
            # ─────────────────────────────────────────────────────────
            # MAX POOLING: Giữ lại con số sai lệch LỚN NHẤT trong 9 ô
            # Chỉ cần 1 ô có kẻ gian lọt vào → toàn bộ khung hình được giữ lại!
            # ─────────────────────────────────────────────────────────
            if diff > max_diff:
                max_diff = diff

    return max_diff

def process_single_video(video_path, output_dir):
    """
    Xử lý một file video: Cắt frame, lọc trùng lặp và lưu xuống ổ cứng.
    
    Đầu vào:
        video_path: Đường dẫn tuyệt đối đến file video gốc (.mp4).
        output_dir: Thư mục con sẽ chứa các frame của video này.
    Đầu ra:
        Trả về (số_ảnh_đã_lưu, số_ảnh_bị_bỏ_qua) để thống kê.
    """
    # Mở video để đọc
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"LỖI: Không thể mở video {video_path}")
        return 0, 0

    # Lấy thông số FPS (Số khung hình trên giây) của video
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print(f"LỖI: FPS của video {video_path} bằng 0")
        cap.release()
        return 0, 0

    # Tính toán số khung hình cần nhảy qua (frame hop)
    # Ví dụ: FPS=30, FRAME_INTERVAL_SEC=1.0 -> Nhảy 30 frames lấy 1 lần.
    frame_hop = int(round(fps * config.FRAME_INTERVAL_SEC))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Đang xử lý: {os.path.basename(video_path)} | FPS: {fps:.2f} | Tổng frames: {total_frames} | Nhảy: {frame_hop} frames/lần")

    saved_count = 0
    skipped_count = 0
    frame_truoc_do = None

    # Duyệt qua các khung hình, nhảy theo frame_hop
    for frame_idx in range(0, total_frames, frame_hop):
        # Đặt vị trí đọc của video nhảy thẳng tới frame_idx
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            break  # Đã hết video hoặc lỗi đọc frame

        # Tính mili-giây THỰC TẾ của frame này trong video.
        # Công thức: (số thứ tự frame / FPS) × 1000 = mili-giây
        # Ví dụ: frame_idx=8, fps=30 → (8/30)*1000 = 266 ms
        #
        # ✅ Lý do dùng cách này thay vì biến đếm cũ:
        #   - Hoạt động đúng với MỌI giá trị FRAME_INTERVAL_SEC (0.25, 1.0, 2.0...)
        #   - Biến đếm cũ: int(0.25) = 0 → cộng mãi mãi không tăng → tên file trùng nhau
        #   - Tên file bây giờ CHÍNH LÀ timestamp thật → GĐ2 đọc thẳng, không cần chuyển đổi
        current_ms = int((frame_idx / fps) * 1000)

        # Tên file: 8 chữ số để chứa video dài nhất (~180 giây = 180000 ms)
        # Ví dụ: frame_00000266.webp (ở mốc 266 mili-giây)
        filename = f"frame_{current_ms:08d}.{config.IMAGE_FORMAT}"
        output_path = os.path.join(output_dir, filename)

        # Logic lọc trùng lặp
        if frame_truoc_do is None:
            # Frame đầu tiên luôn được lưu lại
            cv2.imwrite(output_path, frame)
            frame_truoc_do = frame
            saved_count += 1
        else:
            # Từ frame thứ 2 trở đi, đem so sánh với frame trước đó
            diff = calculate_histogram_difference(frame_truoc_do, frame)
            
            # Nếu sự khác biệt lớn hơn ngưỡng cho phép -> Đây là cảnh mới!
            if diff > config.HISTOGRAM_THRESHOLD:
                cv2.imwrite(output_path, frame)
                frame_truoc_do = frame
                saved_count += 1
            else:
                # Cảnh tĩnh, trùng lặp -> Bỏ qua
                skipped_count += 1

    # Dọn dẹp tài nguyên
    cap.release()
    
    # BÍ KÍP LƯU ĐIỂM NHỚ: Cắm cờ xác nhận xong 100%
    done_flag_path = os.path.join(output_dir, "_DONE.txt")
    with open(done_flag_path, "w") as f:
        f.write("DONE")
        
    return saved_count, skipped_count

def process_video_task(args):
    """
    Ham bao boc (wrapper) de chay da luong cho process_single_video.
    Giai nen args (tuple) va tao thu muc con.
    """
    video_path, out_name = args
    output_dir = os.path.join(config.FRAME_DIR, out_name)
    os.makedirs(output_dir, exist_ok=True)
    return process_single_video(video_path, output_dir)

def get_v3c_video_list():
    """
    Quét trực tiếp vào cấu trúc V3C1/videos và V3C2/videos.
    Trả về:
      videos_to_process: Danh sách video CHƯA LÀM XONG.
      total_videos: Tổng số video có trong ổ cứng.
    """
    videos_to_process = []
    total_videos = 0
    
    # Các bộ dataset con (có thể mở rộng thêm V3C3, V3C4 sau này)
    datasets = ["V3C1", "V3C2"]
    v3c_root = config.VIDEO_DIR
    
    for ds in datasets:
        ds_dir = os.path.join(v3c_root, ds, "videos")
        if not os.path.exists(ds_dir):
            continue
            
        # Duyệt các thư mục con như 00001, 00002...
        for folder_name in sorted(os.listdir(ds_dir)):
            folder_path = os.path.join(ds_dir, folder_name)
            if os.path.isdir(folder_path):
                # File mp4 nằm trong thư mục con này, tên giống tên thư mục
                mp4_file = os.path.join(folder_path, f"{folder_name}.mp4")
                if os.path.exists(mp4_file):
                    total_videos += 1
                    out_name = f"{ds.lower()}_{folder_name}"
                    output_dir = os.path.join(config.FRAME_DIR, out_name)
                    
                    # KIỂM TRA ĐIỂM NHỚ: Nếu có cờ _DONE.txt thì bỏ qua!
                    if os.path.exists(os.path.join(output_dir, "_DONE.txt")):
                        continue
                        
                    videos_to_process.append((mp4_file, out_name))
                    
    return videos_to_process, total_videos

def main():
    """
    Hàm điều phối chính cho Giai đoạn 1.
    """
    start_time = time.time()
    print("=" * 60)
    print("BAT DAU GIAI DOAN 1: TRICH XUAT VA LOC FRAME (CHUAN V3C)")
    print("=" * 60)

    # Đảm bảo thư mục lưu frame tồn tại
    os.makedirs(config.FRAME_DIR, exist_ok=True)

    # Lấy danh sách video theo cấu trúc V3C
    videos_to_process, total_videos = get_v3c_video_list()
    
    if total_videos == 0:
        print(f"Khong tim thay file .mp4 nao trong thu muc V3C.")
        return
        
    already_done = total_videos - len(videos_to_process)
    print(f"Kiem tra Diem Nho: Da lam {already_done}/{total_videos} video.")
    
    if len(videos_to_process) == 0:
        print(f"✅ Tuyet voi! Toan bo {total_videos} video da duoc cat anh xong tu truoc!")
        return

    total_saved = 0
    total_skipped = 0

    try:
        from tqdm import tqdm
    except ImportError:
        print("Khong tim thay thu vien tqdm. Hay cai dat: pip install tqdm")
        return

    # Lặp qua từng video để xử lý (Dùng tqdm để Backend Web app.py bắt được %)
    # Chạy đa luồng bằng ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor(max_workers=config.MAX_WORKERS_G1) as executor:
        # Sử dụng initial = already_done để thanh tiến trình nối tiếp % chính xác!
        results = list(tqdm(
            executor.map(process_video_task, videos_to_process),
            total=total_videos,
            initial=already_done,
            desc="Tiến độ V3C",
            unit="video"
        ))

    # Tổng hợp kết quả từ các luồng
    for saved, skipped in results:
        total_saved += saved
        total_skipped += skipped

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("-" * 60)
    print("HOAN THANH GIAI DOAN 1")
    print(f"   - Tổng video đã xử lý: {len(videos_to_process)}")
    print(f"   - Tổng số frame đã LƯU: {total_saved}")
    print(f"   - Tổng số frame đã LỌC BỎ: {total_skipped}")
    print(f"   - Thời gian chạy: {elapsed_time:.2f} giây")
    print("=" * 60)

if __name__ == "__main__":
    main()
