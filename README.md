# Tempo Run 2026 - Cỗ Xe Tăng Bất Tử (Local Pipeline)

## 1. Giới thiệu Phương pháp
Phương pháp "Cỗ Xe Tăng Bất Tử" giải quyết bài toán Temporal Video Retrieval thông qua kiến trúc 4 Trạm Độc lập, chạy hoàn toàn offline trên Local:
- **Trạm 1 (Ollama Llama 3.2 3B)**: Bóc tách ngữ nghĩa câu truy vấn thành các trường dữ liệu (Cảnh chính, Bối cảnh trước/sau, Từ khóa OCR).
- **Trạm 2 (CLIP ViT-bigG-14 1280D)**: Mã hóa Frame ảnh (GĐ2) và câu truy vấn (Trạm 2) thành không gian vector chung.
- **Trạm 3 (Numpy - Soft Fusion)**: Tìm kiếm trượt thời gian (Sliding Window) kết hợp hàm mũ và hệ số nhân để tăng cường độ chính xác, truy xuất Top 500 ứng viên cực nhanh qua mmap.
- **Trạm 4 (EasyOCR)**: Quét nhận diện chữ (OCR) trên các ứng viên Top 500 và thưởng điểm tự tin nếu khớp với từ khóa trong câu hỏi.

## 2. Cấu trúc Repository
```
final/
├── data/                  # Thư mục trống để chứa dữ liệu đầu vào và trung gian
├── src/                   # Chứa mã nguồn chính
│   ├── config.py          # Bảng điều khiển trung tâm (Cấu hình toàn bộ siêu tham số)
│   ├── step1_extract_frames.py # Giai đoạn 1: Trích xuất Frame
│   ├── step2_encode_clip.py    # Giai đoạn 2: Mã hóa Frame -> Vector
│   ├── station1_ollama.py      # Trạm 1: Bóc tách ngữ nghĩa bằng LLM
│   ├── station2_clip.py        # Trạm 2: Đúc Vector Query
│   ├── station3_numpy.py       # Trạm 3: Tìm kiếm Numpy (Soft Fusion)
│   ├── station4_ocr.py         # Trạm 4: Nhận diện chữ
│   └── station34_pipeline.py   # Điều phối song song Trạm 3 & 4
├── environment.yml        # Tệp môi trường Conda
├── requirements.txt       # Tệp thư viện Pip
├── main.py                # Script chạy toàn bộ pipeline từ số 0
└── README.md              # File hướng dẫn này
```

## 3. Yêu cầu Phần cứng & Phần mềm
- **Hệ điều hành**: Ubuntu 22.04 / Windows 10/11
- **Card Đồ Họa (GPU)**: Đã kiểm thử trên NVIDIA RTX 4090 / RTX 5050. Yêu cầu VRAM tối thiểu 8GB.
- **CUDA Toolkit**: 12.1+ (Khuyến nghị dùng Conda để tự quản lý)
- **RAM hệ thống**: Khuyến nghị 32GB+
- **Ổ cứng**: SSD NVMe (cần ít nhất 50GB trống để lưu Frame và Vector Database).

## 4. Hướng dẫn cài đặt môi trường
Ban Tổ chức được khuyến nghị sử dụng **Conda** để tránh xung đột hệ thống.

**Cách 1: Sử dụng Conda (Khuyên dùng)**
```bash
conda env create -f environment.yml
conda activate temporun-2026
```

**Cách 2: Sử dụng Pip**
```bash
pip install -r requirements.txt
```

## 5. Hướng dẫn tải tài nguyên bổ sung
Mã nguồn yêu cầu **Ollama** cài đặt cục bộ để chạy mô hình Llama 3.2 3B ở Trạm 1.
- **Cài đặt Ollama**: Tải tại `https://ollama.com/download`
- **Tải mô hình**: Chạy lệnh sau trên terminal/cmd:
  ```bash
  ollama run llama3.2:3b
  ```
  *(Ollama phải được giữ trạng thái chạy ngầm khi thực thi mã nguồn).*

## 6. Mô tả Dữ liệu Đầu vào
- Mã nguồn nhận đường dẫn tự động thông qua dòng lệnh.
- Thư mục video gốc chứa file `.mp4` phải được truyền qua biến `--video_dir`.
- File câu hỏi (đề thi) `.jsonl` phải được truyền qua biến `--task_file`.

## 7. Mô tả Kết quả Đầu ra
Kết quả được xuất ra file JSON (ví dụ: `submission.json`) tại đường dẫn truyền qua biến `--output`. File kết quả đúng cấu trúc BTC yêu cầu: 10 predictions mỗi task_id, kèm theo frame_ms tính từ đầu clip.

## 8. Hướng dẫn chạy từng Script (Thủ công)
Nếu BTC muốn chạy từng bước (Thay vì dùng `main.py`), hãy cấu hình các biến môi trường `VIDEO_DIR`, `TASK_FILE_PATH`, `SUBMISSION_FILE_PATH` và chạy lần lượt:
```bash
python src/step1_extract_frames.py
python src/step2_encode_clip.py
python src/station1_ollama.py
python src/station2_clip.py
python src/station34_pipeline.py
```

## 9. Lệnh chạy toàn bộ Pipeline (Tự động từ con số 0)
Đây là lệnh chính thức để BTC đánh giá bài làm. Câu lệnh này sẽ tự động chạy toàn bộ quy trình từ cắt video -> ép vector -> trích xuất NLP -> tìm kiếm OCR -> ra kết quả.

```bash
python main.py --video_dir /du/ong/dan/toi/dataset/videos --task_file /du/ong/dan/toi/private_round_tasks.jsonl --output /du/ong/dan/toi/submission.json
```

*(Hãy thay đổi `/du/ong/dan/toi/...` thành đường dẫn thực tế trên máy đánh giá)*.

## 10. Các Tham số Mặc định
Mọi thông số được đặt trong `src/config.py`. Hệ thống đã được tuning tối ưu:
- `VISION_ECOSYSTEM = "1280D"`: Khóa cứng model ViT-bigG-14.
- `FRAME_INTERVAL_SEC = 1.0`: Lấy 1 khung hình mỗi giây.
- `TOP_K_SEARCH = 100`: Tối ưu hóa truy xuất bộ nhớ.
- `COMPUTE_DTYPE = "float16"`: Tránh Out of Memory cho GĐ2.

## 11. Các lỗi hoặc giới hạn đã biết
- Lỗi kết nối Ollama: Nếu Ollama không được bật hoặc cổng `11434` bị chiếm, Trạm 1 sẽ báo lỗi và Fallback về việc dùng nguyên văn câu hỏi (Tắt tính năng chia tách ngữ cảnh). BTC lưu ý kiểm tra dịch vụ Ollama trước khi chấm.
- Hệ thống cần nhiều CPU/RAM để cắt ảnh từ Video (OpenCV/FFmpeg) ở GĐ1, có thể gây nóng máy cục bộ trong 1-2 tiếng đầu tiên.
