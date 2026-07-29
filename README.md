# Giải pháp truy xuất khoảnh khắc trên dữ liệu định dạng mp4 lớn kết hợp tăng cường độ chính xác bằng phương pháp mở rộng ngữ nghĩa với local LLM

> **Lời tựa từ Đội thi:** Toàn bộ quá trình nghiên cứu, từ việc học hỏi các kiến thức cơ bản về Video Retrieval cho đến từng dòng mã nguồn, đều được đội thi tự học hỏi từng bước và xây dựng hoàn toàn với sự trợ giúp đắc lực của các Trợ lý Trí tuệ Nhân tạo (AI).

## 1. Giới thiệu Phương pháp
Giải pháp giải quyết bài toán Temporal Video Retrieval thông qua đường ống xử lý tự động (End-to-End Pipeline). Quy trình được tối ưu hóa về dung lượng và bộ nhớ VRAM, chia thành 2 Giai đoạn Tiền xử lý và 4 Module Truy xuất cốt lõi nối tiếp nhau:

**Giai đoạn 1: Tiền xử lý Video (Frame Extraction)**
Quét toàn bộ dữ liệu gốc định dạng `.mp4`. Trích xuất 1 khung hình/giây (`FRAME_INTERVAL_SEC = 1.0`) và lưu dưới chuẩn `.webp` nhằm tối ưu dung lượng đĩa.
- **Xử lý đa luồng:** Sử dụng cơ chế phân luồng cấp phát song song (`MAX_WORKERS = 8`) để giải mã đồng loạt nhiều video cùng lúc, tối đa hóa hiệu suất CPU của thiết bị.
- **Thuật toán nội suy "Hybrid V" (Được AI đề xuất):** Tự động so sánh và loại bỏ khung hình tĩnh/trùng lặp qua 5 bước: Downscale 128x128 ➡️ Phân chia lưới 3x3 (Overlapping Grid 50%) ➡️ Chuyển đổi màu HSV ➡️ Tự động điều chỉnh ngày/đêm (ngưỡng `HYBRID_V_THRESHOLD = 40`, `HYBRID_S_THRESHOLD = 20`) ➡️ Tính khoảng cách Bhattacharyya trên Histogram. Giữ lại khung hình nếu sai lệch vượt `HISTOGRAM_THRESHOLD = 0.12`.

**Giai đoạn 2: Mã hóa Hình ảnh (Vision Encoding)**
Sử dụng mô hình thị giác `ViT-bigG-14` mã hóa ảnh `.webp` thành các ma trận Vector không gian 1280 chiều.
- **Chiến lược triển khai lai (Hybrid Deployment):** Cung cấp sẵn kịch bản mã hóa trên đám mây (Kaggle 2x T4 16GB) cho tập dữ liệu lớn. Đối với môi trường cục bộ (Local), hệ thống tự động ép định dạng số thực 16-bit (`COMPUTE_DTYPE = float16`) và giới hạn kích thước lô (`CLIP_BATCH_SIZE = 64`) để tối ưu hóa bộ nhớ.

**Module 1: Mở rộng Ngữ nghĩa (LLM Semantic Expansion)**
Sử dụng mô hình ngôn ngữ cục bộ `Ollama Llama 3.2 3B` (Greedy Decoding, `Temperature = 0.0`) phân tích file đề thi `.jsonl`. Câu truy vấn được bóc tách thành 3 luồng dữ liệu độc lập: Bối cảnh, Cảnh chính, và Từ khóa văn bản (OCR), tạo tiền đề cho quá trình truy xuất không gian - thời gian.

**Module 2: Mã hóa Truy vấn (Text Encoding)**
Dữ liệu văn bản từ Module 1 được đưa qua nhánh Text-Encoder của `ViT-bigG-14` để đồng bộ hóa thành Vector Truy vấn, khớp nối chuẩn xác với Không gian Vector Hình ảnh.

**Module 3: Truy xuất Không gian - Thời gian (Soft-Fusion Retrieval)**
Thực hiện phép nhân ma trận (Dot-Product) trượt trên trục thời gian (Temporal Sliding Window). 
- **Cách tính điểm:** Tính Cosine Similarity cho "Cảnh chính", cộng dồn trọng số từ các khung hình "Bối cảnh trước/sau" lân cận bằng hàm suy giảm mũ (Exponential Decay). Hệ thống truy vết chính xác chuỗi hành động và trích xuất số lượng ứng viên theo cấu hình (`TOP_K_SEARCH = 100`).

**Module 4: Nhận diện Văn bản & Kết xuất (OCR & Submission)**
Kích hoạt mô hình `EasyOCR` quét trực tiếp trên các ứng viên từ Module 3. 
- **Cách tính điểm:** Văn bản trích xuất được đối chiếu với "Từ khóa OCR" bằng thuật toán `Levenshtein`. Nếu mức tương đồng chuỗi vượt ngưỡng, hệ thống tự động cộng Điểm thưởng (Bonus Score) vào điểm Cosine gốc. 
- Cuối cùng, hệ thống lọc Top 10 khung hình hoàn hảo nhất mỗi truy vấn và kết xuất ra file `submission.json` theo định dạng của Ban Tổ chức.

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
- **Card Đồ Họa (GPU)**: Khuyến nghị GPU có tối thiểu 12GB VRAM (như RTX 3060, RTX 4070 trở lên) để chạy mượt mà mô hình ViT-bigG-14 ở Batch Size 64 (tiêu thụ khoảng 9.5GB VRAM). Do giới hạn phần cứng của thiết bị cá nhân (Laptop 8GB VRAM), đội ngũ phát triển đã linh hoạt triển khai Giai đoạn 2 (Vision Encoding) trên nền tảng đám mây Kaggle (2x T4 16GB VRAM) để hoàn thành quá trình mã hóa dữ liệu thực tế. Nếu BTC chạy trên máy chủ có VRAM >= 12GB, toàn bộ mã nguồn có thể chạy hoàn toàn cục bộ (Local).
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
