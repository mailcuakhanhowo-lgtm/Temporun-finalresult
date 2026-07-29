# src/config.py
# ============================================================
# BẢNG ĐIỀU KHIỂN TRUNG TÂM – Tempo Run 2026
# ============================================================
# Mọi thông số cấu hình của dự án được đặt TẠI ĐÂY.
# Các file xử lý (step1, step2...) chỉ được IMPORT từ file này.
# TUYỆT ĐỐI KHÔNG được viết cứng (hardcode) con số vào file khác.
# ============================================================

import os

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN (PATHS)
# ==========================================

# Lấy đường dẫn thư mục gốc của dự án.
# __file__  = đường dẫn đến file config.py này (trong thư mục src/)
# dirname() gọi 2 lần để leo lên 2 cấp: src/ → Temple run/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Thư mục chứa toàn bộ dữ liệu
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Thư mục lưu video gốc đầu vào (.mp4) (Cho phép BTC ghi đè qua biến môi trường)
VIDEO_DIR = os.environ.get("VIDEO_DIR", os.path.join(DATA_DIR, "videos"))

# Thư mục lưu ảnh cắt ra từ video
FRAME_DIR = os.environ.get("FRAME_DIR", os.path.join(DATA_DIR, "frames"))

# ==========================================
# 1B. CÔNG TẮC TỔNG – HỆ SINH THÁI THỊ GIÁC (DUAL ECOSYSTEM)
# ==========================================
# "CUSTOM" → Chế độ Đa Năng: Dùng model do ngài chọn (ViT-H, SigLIP...), 
#            số chiều D phụ thuộc vào model. Cho phép LLM mở rộng Query, Bầu cử.
# "1280D"  → Chế độ Hủy Diệt: Khóa cứng ViT-bigG-14 (1280D).
#            TẮT HOÀN TOÀN LLM. Query nguyên bản đâm thẳng vào model.
VISION_ECOSYSTEM = "1280D"

# Định tuyến động (Dynamic Routing) cho thư mục lưu Vector
# Khi gạt Công tắc Tổng, hệ thống tự động chĩa vào đúng kho thịt.
if VISION_ECOSYSTEM == "1280D":
    EMBEDDING_DIR = os.path.join(DATA_DIR, "embeddings_1280")
else:
    EMBEDDING_DIR = os.path.join(DATA_DIR, "embeddings")


# ==========================================
# 2. CẤU HÌNH GIAI ĐOẠN 1 – TRÍCH XUẤT FRAME
# ==========================================

# Tần suất cắt frame: cứ mỗi bao nhiêu GIÂY thì lấy 1 ảnh.
# Mặc định: 1 frame/giây (phù hợp với quy mô cuộc thi)
FRAME_INTERVAL_SEC = 1.0

# Định dạng ảnh lưu xuống ổ cứng.
# Dùng "webp" thay vì "jpg" → giảm ~50% dung lượng, cùng chất lượng.
IMAGE_FORMAT = "webp"

# Ngưỡng lọc trùng lặp (dùng MAX Pooling Bhattacharyya trên Overlapping Grid).
# Với thuật toán mới dùng hàm MAX (thay vì trung bình cộng), giá trị 0.12 là
# điểm ngọt nhất: loại bỏ ảnh rác cùng cảnh, nhưng không bỏ sót sự kiện nhỏ.
HISTOGRAM_THRESHOLD = 0.12

# Ngưỡng ánh sáng để kích hoạt chế độ Ban Đêm (Hybrid V).
# Kênh V (Value) trong HSV có giá trị từ 0 (đen) đến 255 (trắng).
# Nếu độ sáng trung bình của ô lưới < ngưỡng này → chế độ Ban Đêm bật.
HYBRID_V_THRESHOLD = 40

# Ngưỡng độ rực để phát hiện cảnh đen trắng (Hybrid V).
# Kênh S (Saturation) từ 0 (xám) đến 255 (rực rỡ).
# Nếu độ rực trung bình < ngưỡng này → cảnh đen trắng → chế độ Ban Đêm bật.
HYBRID_S_THRESHOLD = 20

# Số luồng song song (CPU Cores) dùng để giải mã video.
# Cấu trúc: 8 nhân cắt ảnh, chừa lại 2 nhân cho HĐH và UI.
MAX_WORKERS_G1 = 8


# ==========================================
# 3. CẤU HÌNH GIAI ĐOẠN 2 – MÃ HÓA CLIP (MODEL-AGNOSTIC)
# ==========================================
# KIẾN TRÚC MODEL-AGNOSTIC: Muốn đổi model, chỉ sửa 1 dòng ở đây.
# Các file step2, step3... sẽ đọc từ đây, không biết tên model cụ thể.

# Tên model CLIP sẽ dùng. Các lựa chọn đã được Ana đánh giá:
#   "ViT-L-14"      ← Phương án CHÍNH (Mặc định) – nhẹ, nhanh, ổn định
#   "ViT-SO400M-14-SigLIP" ← Phương án MẠNH NHẤT (Dành cho Kaggle)
MODEL_NAME = "ViT-SO400M-14-SigLIP"

# Bộ trọng số (pretrained weights) tương ứng với model trên.
# CLIP ViT-L-14 dùng bộ "openai", SigLIP dùng bộ "webli".
MODEL_PRETRAINED = "webli"

# Số ảnh xử lý cùng lúc trên GPU (batch size).
# 64 là con số tối ưu cho RTX 5050 8GB với float16.
# Tăng lên nếu còn VRAM dư, giảm xuống nếu bị lỗi Out Of Memory.
CLIP_BATCH_SIZE = 64

# Kiểu dữ liệu số dùng khi tính toán trên GPU.
# "float16" = dùng số 16-bit thay vì 32-bit mặc định
#   → Giảm 50% VRAM, GPU Blackwell (RTX 5050) xử lý nhanh hơn.
COMPUTE_DTYPE = "float16"


# ==========================================
# 4. CẤU HÌNH GIAI ĐOẠN 5 – TÌM KIẾM RUNTIME
# ==========================================
# ⚠️ BA QUY TẮC TOÁN HỌC BẤT BIẾN (KHÔNG ĐƯỢC VI PHẠM):
#
# Quy tắc 1 – PHÉP NHÂN MA TRẬN:
#   Dùng cú pháp: scores = Kho_Thit @ Query_Vector.T
#   → Lật (transpose) phía QUERY, KHÔNG lật Kho_Thịt khổng lồ.
#   → Giữ nguyên Kho_Thịt trong RAM để tận dụng Cache, tối ưu bộ nhớ đệm CPU/GPU.
#
# Quy tắc 2 – THAM SỐ HÓA TOP K:
#   Con số 100 bên dưới KHÔNG ĐƯỢC hardcode vào file step5 hay bất kỳ file nào khác.
#   Ông Chủ có thể tăng/giảm TOP_K_SEARCH tại đây để tuning tốc độ vs độ chính xác.
#
# Quy tắc 3 – THUẬT TOÁN TÌM TOP K:
#   TUYỆT ĐỐI KHÔNG dùng .sort() toàn cục → O(N log N), quá chậm với N = hàng triệu vector.
#   BẮT BUỘC dùng torch.topk(k=TOP_K_SEARCH) hoặc np.argpartition → O(N log K), siêu nhanh.
#   (Ví dụ: N=1,000,000 vector, K=100 → nhanh gấp ~10,000 lần so với sort() toàn cục)

# Số lượng kết quả tốt nhất sẽ được lọc ra từ Kho Thịt (trước khi gửi cho Gemini xác nhận).
# Ông Chủ tự tuning con số này khi đi thi tùy theo tốc độ và độ chính xác mong muốn.
TOP_K_SEARCH = 100


# ==========================================
# 5. CẤU HÌNH GIAI ĐOẠN 4 – MỞ RỘNG QUERY (QUERY EXPANSION)
# ==========================================
# Kiến trúc: Cloud LLM (Groq/Gemini) → Fallback Local LLM (Qwen 0.5B) → CLIP Text Encoder

# === CÔNG TẮC CHÍNH ===
# Chọn "groq"   → Siêu tốc (~0.1s/query, dùng LPU chuyên dụng của Groq)
# Chọn "gemini" → Dự phòng ổn định, hiểu Tiếng Việt tốt hơn
LLM_PROVIDER = "gemini"

# === CHẾ ĐỘ MỞ RỘNG QUERY (LLM_MODE) ===
# "off"   → Không dùng AI, tốc độ ánh sáng, an toàn tuyệt đối.
# "cloud" → Dùng Gemini/Groq (Không tốn RAM, cần mạng).
# "local" → Dùng Ollama (Chạy ngầm ở localhost:11434, không sợ rớt mạng).
LLM_MODE = "off"

# Cấu hình Ollama Model
OLLAMA_MODEL = "llama3.2:3b"

# === API KEYS (đọc từ biến môi trường, KHÔNG hardcode vào code) ===
# Cách set key trên Windows:
#   Mở PowerShell, gõ: $env:GROQ_API_KEY = "your_key_here"
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")
# Lấy Gemini API Key tại: aistudio.google.com
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# === CẤU HÌNH GROQ MODEL ===
# Llama 3.1 8B: tốc độ tối ưu (~800-1000 tokens/giây trên LPU Groq)
# Đây là model tốt nhất để mở rộng query ngắn gọn và nhanh.
GROQ_MODEL_NAME = "llama3-8b-8192"

# === THAM SỐ QUERY ===
# Số biến thể sinh ra từ 1 câu query gốc.
# ⚡ Tăng → Chính xác hơn nhưng chậm hơn | Giảm → Nhanh hơn nhưng ít đa dạng
# Điểm ngọt nhất: 3 (cân bằng tốc độ/độ chính xác cho cuộc thi)
NUM_QUERY_VARIANTS = 3

# === TIMEOUT ===
# Thời gian tối đa (giây) chờ Cloud API phản hồi.
# Nếu vượt ngưỡng này → tự động kích hoạt Qwen 0.5B dự phòng cục bộ.
API_TIMEOUT_SEC = 30

# ==========================================
# 6. CẤU HÌNH GIAI ĐOẠN 6 & 7 – XUẤT KẾT QUẢ
# ==========================================

# Hằng số cản (K) trong công thức RRF: Điểm = 1 / (RRF_CONSTANT + Hạng)
RRF_CONSTANT = 60

# Trọng số Mỏ Neo (Trọng số dành cho câu gốc do BTC cung cấp)
# Giúp câu gốc có tiếng nói gấp N lần so với các biến thể do LLM băm ra.
ANCHOR_WEIGHT = 2.0

# ==========================================
# 6B. CẤU HÌNH PHƯƠNG ÁN MỚI – GỘP VECTOR SỚM (EARLY WEIGHTED FUSION)
# ==========================================
# Phương án bổ sung chạy SONG SONG với RRF. Không ảnh hưởng phương án gốc.
# "rrf"            → Bầu cử cuối chu kỳ (Phương án gốc, an toàn, ổn định)
# "early_weighted" → Gộp Vector tuyến tính (Phương án mới, siêu tốc, chính xác cao)
FUSION_MODE = "none"

# Bộ trọng số dùng riêng cho chế độ "early_weighted".
# Key = tên JSON định danh mà LLM phải trả về. Value = trọng số nhân vào vector.
# Câu gốc (original_rephrase) có trọng số áp đảo = 10.0 để làm Mỏ Neo.
# Các chi tiết nguyên tử (background_context, extra_details) chỉ "nhích" vector.
FUSION_WEIGHTS = {
    "original_rephrase":  10.0,  # Câu đảo ngữ toàn cảnh (trọng số bá chủ)
    "main_subject":        4.0,  # Chủ thể chính (người, vật, xe...)
    "action":              3.0,  # Hành động (lái xe, chạy, cầm...)
    "background_context":  2.0,  # Bối cảnh phụ (trời mưa, ban đêm...)
    "extra_details":       1.0,  # Chi tiết lẻ (áo đỏ, mũ xanh...)
}

# Số lượng kết quả CUỐI CÙNG trả về sau khi Bầu cử RRF. 
# Giới hạn của BTC là 10 kết quả mỗi task.
FINAL_TOP_K = 10

# CÔNG TẮC LAI (HYBRID) CHO GIAI ĐOẠN 6
# False (Mặc định khi đi thi hàng loạt): Tắt Gemini soi ảnh, xuất thẳng JSON nộp bài siêu tốc.
# True: Bật Gemini soi ảnh (Dùng khi thi Interactive thong thả).
USE_GEMINI_RERANK = False

# ==========================================
# 7. CẤU HÌNH GIAI ĐOẠN 7 – TẠO FILE NỘP BÀI
# ==========================================

# Đường dẫn file đề thi (chứa các task cần giải quyết).
_private_task_path = os.path.join(DATA_DIR, "private_round_tasks.jsonl")
_public_task_path = os.path.join(DATA_DIR, "public_round_tasks.jsonl")
_default_task = _private_task_path if os.path.exists(_private_task_path) else _public_task_path
TASK_FILE_PATH = os.environ.get("TASK_FILE_PATH", _default_task)

# Đường dẫn file kết quả sẽ nộp cho BTC.
SUBMISSION_FILE_PATH = os.environ.get("SUBMISSION_FILE_PATH", os.path.join(DATA_DIR, "submission.json"))

# Số luồng đa nhiệm (Multi-threading) dùng trong GĐ7.
# Lý do dùng đa luồng: gọi 1.000 Cloud API tuần tự tốn >8 phút (quá lâu để nộp bài).
# Đa luồng giúp gọi song song, giảm xuống ~1-2 phút.
#
# ⚠️ TẠI SAO CHỐT Ở 5 LUỒNG?
#   - Groq Free Tier giới hạn: ~30 requests/phút = ~0.5 request/giây
#   - 5 luồng × 1 request/luồng = 5 req/s ở peak → có thể chạm rate limit nếu tăng thêm
#   - 5 là điểm ngọt nhất: đủ nhanh mà không bị lỗi 429 Too Many Requests
#   - Nếu có API key trả phí với rate limit cao hơn, có thể tăng lên 10-20.
MAX_WORKERS_G7 = 5

# ==========================================
# 8. CẤU HÌNH ĐÁM MÂY KAGGLE
# ==========================================
# Số lượng ảnh tối đa trong mỗi khối nén ZIP tải lên Kaggle.
# Tăng -> Ít file ZIP hơn, nhanh hơn nhưng nếu rớt mạng tải lại sẽ tốn thời gian.
# Giảm -> Nhiều file ZIP hơn, chia nhỏ rủi ro, an toàn cho mạng yếu.
# Số khuyến nghị: 250000 (khoảng 1.5 - 2GB một cục ZIP).
KAGGLE_CHUNK_SIZE = 250000
