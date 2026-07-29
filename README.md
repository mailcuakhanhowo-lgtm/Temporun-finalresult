# Giải pháp truy xuất khoảnh khắc trên dữ liệu định dạng mp4 lớn kết hợp tăng cường độ chính xác bằng phương pháp mở rộng ngữ nghĩa với local LLM

> **Lời tựa từ Đội thi:** Toàn bộ quá trình nghiên cứu, từ việc học hỏi các kiến thức cơ bản về Video Retrieval cho đến từng dòng mã nguồn, đều được đội thi tự học hỏi từng bước và xây dựng hoàn toàn với sự trợ giúp đắc lực của các Trợ lý Trí tuệ Nhân tạo (AI).

## 1. Giới thiệu Phương pháp
Giải pháp giải quyết bài toán Temporal Video Retrieval thông qua đường ống xử lý tự động (End-to-End Pipeline) được chia thành 2 khối chính:

**Khối 1: Giai đoạn Tiền xử lý (Offline)**
- **Cắt ảnh (Frame Extraction)**: Lọc khung hình trùng lặp bằng thuật toán nội suy **Hybrid V** (tự động phát hiện ngày/đêm và tính khoảng cách Bhattacharyya), do AI đề xuất.
- **Mã hóa (Vision Encoding)**: Trích xuất đặc trưng không gian 1280 chiều từ các khung hình `.webp` bằng mô hình **ViT-bigG-14** (tối ưu hóa `float16`).

**Khối 2: Giai đoạn Truy vấn (Online - 4 Trạm xử lý)**
- **Trạm 1 (Mở rộng Ngữ nghĩa)**: Sử dụng LLM cục bộ **Ollama Llama 3.2 3B** phân rã truy vấn gốc thành Cảnh chính, Bối cảnh và Từ khóa.
- **Trạm 2 (Mã hóa Truy vấn)**: Đồng bộ hóa văn bản vào không gian vector qua nhánh Text-Encoder của **ViT-bigG-14**.
- **Trạm 3 (Truy xuất Không gian - Thời gian)**: Tìm kiếm ứng viên bằng thuật toán **Soft-Fusion Retrieval** (phép nhân ma trận kết hợp trượt thời gian Sliding Window và hàm mũ).
- **Trạm 4 (Nhận diện Văn bản)**: Kích hoạt **EasyOCR** kết hợp thuật toán so khớp chuỗi **Levenshtein** để cộng điểm thưởng (Confidence Bonus).

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
- **Hệ điều hành**: Windows 11
- **Card Đồ Họa (GPU)**: Khuyến nghị GPU có tối thiểu 12GB VRAM (như RTX 3060, RTX 4070 trở lên) để chạy mượt mà mô hình ViT-bigG-14 ở Batch Size 64 (thực tế tiêu thụ khoảng 9.5GB VRAM).
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
- Thư mục video gốc chứa file `.mp4` được truyền qua tham số `--video_dir` *(ví dụ: `"D:/Data_ChamThi/Videos"`)*.
- File câu hỏi (đề thi) `.jsonl` được truyền qua tham số `--task_file` *(ví dụ: `"D:/Data_ChamThi/private_round_tasks.jsonl"`)*.

## 7. Mô tả Kết quả Đầu ra
Kết quả được xuất ra file JSON tại đường dẫn truyền qua biến `--output` *(ví dụ: `"D:/KetQua/doi_co_xe_tang.json"`)*. File kết quả tuân thủ cấu trúc BTC yêu cầu: 10 predictions mỗi task_id, kèm theo frame_ms tính từ đầu clip.

## 8. Hướng dẫn chạy từng phần (Thủ công)
Nếu BTC muốn kiểm thử từng bước độc lập (thay vì dùng `main.py`), vui lòng cấu hình các biến môi trường `VIDEO_DIR`, `TASK_FILE_PATH`, `SUBMISSION_FILE_PATH` và chạy lần lượt:
```bash
python src/step1_extract_frames.py
python src/step2_encode_clip.py
python src/station1_ollama.py
python src/station2_clip.py
python src/station34_pipeline.py
```

## 9. Hướng dẫn chạy bằng Bảng Điều Khiển (Giao diện UI)
Để hỗ trợ Ban Tổ chức thao tác trực quan hơn, mã nguồn cung cấp một Bảng điều khiển (Dashboard) không yêu cầu cài đặt thêm thư viện web. Tại giao diện này, BTC có thể dễ dàng **chạy thử độc lập từng Giai đoạn/Trạm** hoặc **chạy toàn bộ quy trình từ A-Z** chỉ bằng một cú click chuột.

Vui lòng khởi động giao diện bằng lệnh:
```bash
python dashboard_btc.py
```
> **💡 Tính năng Đồng bộ Thông minh (Terminal ↔ UI):** Giao diện được trang bị sẵn 3 ô nhập đường dẫn. Theo mặc định, hệ thống trỏ về dữ liệu mẫu `data/`. Tuy nhiên, nếu BTC từng chạy lệnh ở Terminal (Mục 10) trước đó, Giao diện sẽ **tự động ghi nhớ và đồng bộ** toàn bộ các đường dẫn bí mật mà BTC đã gõ. BTC cũng có toàn quyền chỉnh sửa các đường dẫn này trực tiếp ngay trên Giao diện.
> 
> **Quy tắc ưu tiên (Ghi đè):** Nếu BTC chỉnh sửa đường dẫn trên Giao diện, hệ thống sẽ chạy và lưu lại cấu hình đó. Tuy nhiên, nếu sau đó BTC quyết định quay lại dùng lệnh Terminal (`python main.py ...`), **lệnh gõ tay trên Terminal luôn mang Quyền Ưu Tiên Cao Nhất**. Nó sẽ ghi đè mọi cấu hình của UI trước đó và ép UI phải đồng bộ theo lệnh mới này.

## 10. Lệnh chạy toàn bộ Pipeline
Đây là lệnh tự động hóa toàn bộ quy trình:

```bash
python main.py --video_dir "C:/thu_muc_cua_BTC/videos" --task_file "C:/thu_muc_cua_BTC/private_round_tasks.jsonl" --output "C:/thu_muc_cua_BTC/submission.json"
```

*(Vui lòng thay đổi đường dẫn `C:/thu_muc_cua_BTC/...` cho phù hợp với hệ thống đánh giá thực tế của Ban Tổ chức)*.

**Ví dụ thực tế:** Nếu máy chấm thi lưu dữ liệu ở ổ E, lệnh có thể trông như thế này:
```bash
python main.py --video_dir "E:/Test_Videos" --task_file "E:/de_thi.jsonl" --output "E:/ket_qua.json"
```

## 11. Các Tham số Mặc định
Mọi thông số được cấu hình trong `src/config.py`:
- Sử dụng mô hình ViT-bigG-14.
- `FRAME_INTERVAL_SEC = 1.0`: Trích xuất 1 khung hình mỗi giây.
- `COMPUTE_DTYPE = "float16"`: Tối ưu bộ nhớ VRAM cho quá trình mã hóa ảnh.

## 12. Các lỗi hoặc giới hạn đã biết
- **Giới hạn VRAM (ViT-bigG-14)**: Trong quá trình phát triển, mô hình `ViT-bigG-14` yêu cầu ~9.5GB VRAM, dẫn đến hiện tượng tràn bộ nhớ (Out of Memory) trên thiết bị cá nhân 8GB VRAM của đội thi. Để giải quyết, đội đã linh hoạt đưa quá trình mã hóa ảnh lên nền tảng đám mây Kaggle (sử dụng 2x T4 16GB VRAM) để hoàn thiện bài toán thực tế. Tuy nhiên, mã nguồn nộp cho Ban Tổ chức vẫn được thiết kế tối ưu (`float16`) để có thể chạy mượt mà hoàn toàn cục bộ trên máy chủ có cấu hình >=12GB VRAM.
- **Sự phụ thuộc Ollama**: Nếu dịch vụ Ollama cục bộ không khả dụng, hệ thống tự động Fallback sử dụng nguyên văn câu truy vấn gốc.
- Quá trình trích xuất khung hình ở Giai đoạn 1 yêu cầu tài nguyên CPU cao do chạy đa luồng đồng thời.
- **Giới hạn của Phương án Tách Trạm Tính Điểm (Temporal Soft Fusion V.2)**: Đội thi ứng dụng thuật toán Cửa sổ Trượt (Sliding Window) để phân tích các biến cố thời gian (Cảnh trước/Cảnh sau). Tuy nhiên, thực tế nghiệm thu cho thấy phương pháp này **không tạo ra sự đột phá về điểm số tổng thể** trên toàn tập dữ liệu. Dù vậy, đội thi vẫn quyết định giữ lại kiến trúc này làm mỏ neo cốt lõi nhằm đảm bảo hệ thống luôn có khả năng xử lý các câu hỏi phức tạp có chứa yếu tố thời gian.
