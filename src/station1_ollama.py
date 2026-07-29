import json
import requests
import os
import sys
from pathlib import Path

# Thêm đường dẫn dự án vào sys.path để import an toàn
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from src import config

# Few-shot prompt siêu việt ép khuôn JSON
SYSTEM_PROMPT = """You are an advanced Video Search AI. 
Your job is to break down a user's natural language video search query into 4 distinct components to assist a Temporal & OCR matching engine.
Respond STRICTLY in JSON format with exactly 4 keys:
1. "context_before": A list of strings describing what happens BEFORE the main action. (Empty list [] if none).
2. "main_shot": A single string describing the most prominent, visual, and distinctive action/object.
3. "context_after": A list of strings describing what happens AFTER the main action. (Empty list [] if none).
4. "ocr_texts": A list of strings.

CRITICAL RULES FOR OCR_TEXTS:
- ONLY extract words that appear exactly inside quotes (" " or “”) in the query.
- NEVER extract physical objects (e.g., red cloth, table) into ocr_texts. 
- If there are no quotes, ocr_texts MUST BE empty [].

EXAMPLES:
Query: A boy wipes his face with a red cloth, then smiles.
{
  "context_before": [],
  "main_shot": "A boy wipes his face with a red cloth",
  "context_after": ["then smiles."],
  "ocr_texts": []
}

Query: A man is walking. The subtitle changes to "I was told I was slim".
{
  "context_before": ["A man is walking."],
  "main_shot": "The subtitle changes to \"I was told I was slim\".",
  "context_after": [],
  "ocr_texts": ["I was told I was slim"]
}
"""

def parse_query_ollama(query_text, model=None):
    if model is None:
        model = config.OLLAMA_MODEL
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": f"{SYSTEM_PROMPT}\n\nQuery: '{query_text}'\nJSON:",
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.0
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        result_text = response.json().get("response", "")
        # Parse the JSON string from Ollama
        parsed = json.loads(result_text)
        
        # Đảm bảo format chuẩn
        return {
            "original_query": query_text,
            "context_before": parsed.get("context_before", []),
            "main_shot": parsed.get("main_shot", query_text),
            "context_after": parsed.get("context_after", []),
            "ocr_texts": parsed.get("ocr_texts", [])
        }
    except Exception as e:
        print(f"[CẢNH BÁO] Lỗi Ollama cho query '{query_text}': {e}")
        # Fallback an toàn (Hạ cấp êm ái)
        return {
            "original_query": query_text,
            "context_before": [],
            "main_shot": query_text,
            "context_after": [],
            "ocr_texts": []
        }

def run_station1():
    import time
    print("🧠 Bắt đầu Trạm 1: Khối Não Bộ Ollama (Bóc tách Ngữ nghĩa)...")
    
    queries_path = "data/queries.json"
    queries_path = "data/queries.json"
    from src import config
    jsonl_path = config.TASK_FILE_PATH
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            raw_queries = [{"id": json.loads(line)["task_id"], "query": json.loads(line)["description"]} for line in f if line.strip()]
        with open(queries_path, "w", encoding="utf-8") as f:
            json.dump(raw_queries, f, indent=4, ensure_ascii=False)
        print(f"✅ Đã nạp {len(raw_queries)} câu hỏi từ {os.path.basename(jsonl_path)}!")
    else:
        print(f"❌ LỖI: Không tìm thấy file {os.path.basename(jsonl_path)}")
        sys.exit(1)
        
    out_path = "data/parsed_queries.json"
    tmp_path = "data/parsed_queries.tmp.json"
    
    parsed_results = {}
    if os.path.exists(out_path):
        print("📥 Phát hiện Checkpoint cũ, nạp dữ liệu để Resume...")
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                old_results = json.load(f)
            
            # Chỉ giữ lại các ID có trong file đề thi hiện tại để tránh kẹt data cũ
            valid_ids = set()
            for idx, item in enumerate(raw_queries):
                if isinstance(item, dict):
                    valid_ids.add(str(item.get("id", idx)))
                else:
                    valid_ids.add(str(idx))
                    
            parsed_results = {k: v for k, v in old_results.items() if k in valid_ids}
            print(f"  Đã nạp {len(parsed_results)} câu đã giải hợp lệ (bỏ qua {len(old_results) - len(parsed_results)} câu rác).")
        except Exception as e:
            print(f"  Lỗi đọc Checkpoint: {e}")
            parsed_results = {}
    
    for idx, item in enumerate(raw_queries):
        if isinstance(item, dict):
            q_id = str(item.get("id", idx))
            q_text = item.get("query", "")
        else:
            q_id = str(idx)
            q_text = str(item)
            
        if q_id in parsed_results:
            continue # Đã giải rồi thì bỏ qua
            
        print(f"  Đang bóc tách [{q_id}]: {q_text} (Model: {config.OLLAMA_MODEL})")
        start_time = time.time()
        parsed_results[q_id] = parse_query_ollama(q_text)
        elapsed = time.time() - start_time
        print(f"  ✅ Xong [{q_id}] trong {elapsed:.2f}s")
        
        # Atomic Write (Ghi Nguyên Tử)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(parsed_results, f, indent=4, ensure_ascii=False)
        os.replace(tmp_path, out_path)
        
        # Check Tín hiệu Dừng
        if os.path.exists("data/stop.signal"):
            print("🛑 [STOP] Đã nhận lệnh dừng, bảo toàn dữ liệu thành công!")
            sys.exit(0)
        
    print(f"✅ Hoàn tất Trạm 1! Đã bóc tách {len(parsed_results)} câu hỏi ra {out_path}.")

if __name__ == "__main__":
    run_station1()
