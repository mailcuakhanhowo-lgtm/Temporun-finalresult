# Giải pháp truy xuất khoảnh khắc trên dữ liệu định dạng mp4 lớn kết hợp tăng cường độ chính xác bằng phương pháp mở rộng ngữ nghĩa với local LLM

## 1. Giới thiệu Phương pháp
Giải pháp giải quyết bài toán Temporal Video Retrieval thông qua kiến trúc 4 Module, chạy hoàn toàn offline trên thiết bị cục bộ:
- **Module 1 (Ollama Llama 3.2 3B)**: Bóc tách ngữ nghĩa câu truy vấn thành các trường dữ liệu (Cảnh chính, Bối cảnh trước/sau, Từ khóa OCR).
- **Module 2 (CLIP ViT-bigG-14 1280D)**: Mã hóa Frame ảnh và câu truy vấn thành không gian vector chung.
- **Module 3 (Numpy - Soft Fusion)**: Tìm kiếm trượt thời gian (Sliding Window) kết hợp hàm mũ và hệ số nhân để tăng cường độ chính xác, truy xuất Top 500 ứng viên.
- **Module 4 (EasyOCR)**: Quét nhận diện chữ (OCR) trên các ứng viên Top 500 và tích hợp điểm tự tin nếu khớp với từ khóa trong câu hỏi.

## 2. Cấu trúc Repository
```
final/
├── data/                  # Thư mục trống để chứa dữ liệu đầu vào và trung gian
├── src/                   # Chứa mã nguồn chính
│   ├── config.py          # Tập tin cấu hình các tham số hệ thống
│   ├── step1_extract_frames.py # Giai đoạn 1: Trích xuất Frame
│   ├── step2_encode_clip.py    # Giai đoạn 2: Mã hóa Frame -> Vector
│   ├── station1_ollama.py      # Module 1: Phân tích ngữ nghĩa bằng LLM
│   ├── station2_clip.py        # Module 2: Trích xuất đặc trưng câu truy vấn
│   ├── station3_numpy.py       # Module 3: Tính toán khoảng cách (Soft Fusion)
│   ├── station4_ocr.py         # Module 4: Nhận diện và đối chiếu văn bản
│   └── station34_pipeline.py   # Module điều phối song song Module 3 & 4
├── environment.yml        # Tệp môi trường Conda
├── requirements.txt       # Tệp thư viện Pip
├── main.py                # Script chạy toàn bộ quy trình
└── README.md              # File hướng dẫn này
```

## 3. Yêu cầu Phần cứng & Phần mềm
- **Hệ điều hành**: Ubuntu 22.04 / Windows 10/11
- **Card Đồ Họa (GPU)**: Đã kiểm thử trên NVIDIA RTX 4090 / RTX 5050. Yêu cầu VRAM tối thiểu 8GB.
- **CUDA Toolkit**: 12.1+ (Khuyến nghị dùng Conda để tự quản lý)
- **RAM hệ thống**: Khuyến nghị 32GB+
- **Ổ cứng**: SSD NVMe (cần ít nhất 50GB trống để lưu Frame và Vector Database).

## 4. Hướng dẫn cài đặt môi trường
Ban Tổ chức được khuyến nghị sử dụng **Conda** để quản lý môi trường.

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
Mã nguồn yêu cầu **Ollama** cài đặt cục bộ để chạy mô hình Llama 3.2 3B ở Module 1.
- **Cài đặt Ollama**: Tải tại `https://ollama.com/download`
- **Tải Script**:
  ```bash
  bash scripts/download_weights.sh
  ```
- **Tải mô hình thủ công (nếu không dùng script)**:
  ```bash
  ollama pull llama3.2:3b
  ```
  *(Ollama phải được giữ trạng thái chạy ngầm khi thực thi mã nguồn).*

## 6. Mô tả Dữ liệu Đầu vào
- Mã nguồn nhận đường dẫn tự động thông qua dòng lệnh.
- Thư mục video gốc chứa file `.mp4` được truyền qua tham số `--video_dir`.
- File câu hỏi (đề thi) `.jsonl` được truyền qua tham số `--task_file`.

## 7. Mô tả Kết quả Đầu ra
Kết quả được xuất ra file JSON (ví dụ: `submission.json`) tại đường dẫn truyền qua biến `--output`. File kết quả tuân thủ cấu trúc BTC yêu cầu: 10 predictions mỗi task_id, kèm theo frame_ms tính từ đầu clip.

## 8. Hướng dẫn chạy từng phần (Thủ công)
Nếu BTC muốn kiểm thử từng bước độc lập (thay vì dùng `main.py`), vui lòng cấu hình các biến môi trường `VIDEO_DIR`, `TASK_FILE_PATH`, `SUBMISSION_FILE_PATH` và chạy lần lượt:
```bash
python src/step1_extract_frames.py
python src/step2_encode_clip.py
python src/station1_ollama.py
python src/station2_clip.py
python src/station34_pipeline.py
```

## 9. Lệnh chạy toàn bộ Pipeline
Đây là lệnh tự động hóa toàn bộ quy trình:

```bash
python main.py --video_dir /du/ong/dan/toi/dataset/videos --task_file /du/ong/dan/toi/private_round_tasks.jsonl --output /du/ong/dan/toi/submission.json
```

*(Vui lòng thay đổi đường dẫn phù hợp với hệ thống đánh giá)*.

## 10. Các Tham số Mặc định
Mọi thông số được cấu hình trong `src/config.py`:
- Sử dụng mô hình ViT-bigG-14.
- `FRAME_INTERVAL_SEC = 1.0`: Trích xuất 1 khung hình mỗi giây.
- `COMPUTE_DTYPE = "float16"`: Tối ưu bộ nhớ VRAM cho quá trình mã hóa ảnh.

## 11. Các lưu ý khi vận hành
- Nếu dịch vụ Ollama không khả dụng (cổng 11434 không phản hồi), hệ thống sẽ tự động sử dụng nguyên văn câu truy vấn gốc. Xin lưu ý khởi động Ollama trước khi thực thi.
- Quá trình trích xuất khung hình ở Giai đoạn 1 yêu cầu tài nguyên CPU/RAM cao, mong BTC lưu ý.
