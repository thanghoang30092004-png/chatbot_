# Recruitment AI Chatbot

Chatbot hỗ trợ tuyển dụng bằng tiếng Việt. Ứng dụng kết hợp:

- **RAG** để tìm kiếm thông tin trong JD, CV và chính sách công ty.
- **SQL agent** để truy vấn dữ liệu có cấu trúc trong SQLite.
- **LangChain tool-calling agent** để tự chọn công cụ phù hợp với câu hỏi.
- **FastAPI** làm backend API.
- **Streamlit** làm giao diện chat.
- **Chroma** làm vector store và `all-MiniLM-L6-v2` làm embedding model.

## Kiến trúc

```text
Streamlit frontend (:8501)
        |
        | POST /chat
        v
FastAPI backend (:6000)
        |
        +-- Agent LangChain
        |     +-- search_jd     -> Chroma JD
        |     +-- search_cv     -> Chroma CV
        |     +-- search_policy -> Chroma policy
        |     +-- sql_query     -> SQLite recruitment.db
        |
        +-- LLM OpenAI-compatible endpoint
```

## Cấu trúc thư mục

```text
build_chatbot/
├── backend/
│   ├── agents/              # Agent và các tool
│   ├── api/                 # FastAPI app và Pydantic models
│   ├── core/                # LLM, embedding loader
│   ├── indexing/            # Tạo vectorstore từ tài liệu PDF
│   ├── retrieval_stategies/ # Các chiến lược retrieval
│   ├── retrievers/          # Nạp retriever từ Chroma
│   └── tools/               # RAG tool và SQL tool
├── data/recruitment_data/
│   ├── cv/                  # CV ứng viên dạng PDF
│   ├── jd/                  # Job Description dạng PDF
│   ├── policy/              # Chính sách dạng PDF
│   └── sql/recruitment.db   # Dữ liệu tuyển dụng SQLite
├── frontend/app.py          # Giao diện Streamlit
└── vectorstores/            # Các Chroma persistent stores
```

## Yêu cầu

- Python 3.10 trở lên.
- Có thể tải model embedding Hugging Face `sentence-transformers/all-MiniLM-L6-v2`.
- Có quyền truy cập LLM endpoint được cấu hình trong `backend/core/load_llm.py`.
- Các vectorstore đã có sẵn trong thư mục `vectorstores/`, hoặc tạo lại theo hướng dẫn bên dưới.

## Cài đặt

Từ thư mục dự án:

```bash
cd /mnt/data/build_chatbot
python -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn streamlit requests slowapi pydantic \
  langchain-core langchain-classic langchain-openai \
  langchain-community langchain-chroma langchain-huggingface \
  langchain-text-splitters pymupdf chromadb sentence-transformers
```

Nếu môi trường đã có Conda, có thể dùng environment hiện tại thay cho `.venv`.

## Cấu hình LLM

LLM hiện được cấu hình trực tiếp trong `backend/core/load_llm.py`:

```python
LLM_API_URL_CORE = "http://10.0.99.116:8070/v1"
MODEL_NAME_CORE = "Qwen/Qwen3.5-35B-A3B"
```

Endpoint cần cung cấp API tương thích OpenAI Chat Completions. Nếu dùng endpoint hoặc model khác, cập nhật hai hằng số trên trước khi chạy backend.

## Chạy ứng dụng

### 1. Khởi động backend

Mở terminal thứ nhất:

```bash
cd /mnt/data/build_chatbot
source .venv/bin/activate
PYTHONPATH=. uvicorn backend.api.main:app --host 0.0.0.0 --port 6000
```

Kiểm tra backend:

```bash
curl http://localhost:6000/health
```

Kết quả mong đợi:

```json
{"status":"ok"}
```

### 2. Khởi động frontend

Mở terminal thứ hai:

```bash
cd /mnt/data/build_chatbot
source .venv/bin/activate
API_URL=http://localhost:6000 streamlit run frontend/app.py --server.fileWatcherType none
```

Sau đó mở URL Streamlit được in trong terminal, thường là `http://localhost:8501`.

Nếu backend chạy ở địa chỉ khác, thay đổi giá trị `API_URL` khi khởi động frontend.

## Tạo lại vectorstore

Vectorstore hiện có thể dùng ngay. Khi thay đổi hoặc thêm tài liệu PDF, chạy:

```bash
cd /mnt/data/build_chatbot
source .venv/bin/activate
PYTHONPATH=. python backend/indexing/build_indexes.py
```

Script đọc metadata từ `data/recruitment_data/metadata.json`, chia tài liệu thành các đoạn nhỏ và tạo ba collection:

- `jd_docs` trong `vectorstores/jd_chroma`
- `cv_docs` trong `vectorstores/cv_chroma`
- `policy_docs` trong `vectorstores/policy_chroma`

## API

### `GET /health`

Kiểm tra trạng thái backend.

### `POST /chat`

Request:

```json
{
  "message": "Nam có phù hợp AI Engineer không?",
  "session_id": null
}
```

`session_id` có thể bỏ qua ở request đầu tiên. Backend sẽ tạo session mới và trả lại ID để dùng cho các lượt chat tiếp theo.

Response rút gọn:

```json
{
  "session_id": "...",
  "answer": "...",
  "tools_used": [],
  "latency_ms": 1234.5,
  "success": true
}
```

### `GET /sessions`

Liệt kê các session chưa hết hạn.

### `DELETE /sessions/{session_id}`

Xóa một session hội thoại.

Backend giới hạn tối đa **20 request/phút cho mỗi địa chỉ client**, lưu tối đa 10 message gần nhất trong session và tự dọn session không hoạt động sau 1 giờ.

## Ví dụ câu hỏi

- Nam có phù hợp AI Engineer không? Vì sao?
- So sánh CV Nam với JD AI Engineer.
- Ứng viên nào có `match_score` cao nhất?
- Quy trình phỏng vấn có mấy vòng?
- Chính sách phúc lợi có hybrid không?
- Liệt kê các application đang ở `technical_interview` hoặc `manager_interview`.

## Luồng chọn công cụ

- Câu hỏi về mô tả công việc, kỹ năng, lương hoặc yêu cầu tuyển dụng: `search_jd`.
- Câu hỏi về ứng viên, học vấn, kinh nghiệm hoặc dự án: `search_cv`.
- Câu hỏi về quy trình, phúc lợi hoặc quy định: `search_policy`.
- Câu hỏi về điểm số, trạng thái, lịch phỏng vấn hoặc thống kê: `sql_query`.
- Câu hỏi kết hợp văn bản và số liệu có thể sử dụng nhiều tool.

## Dữ liệu mẫu

Chi tiết dataset, metadata và các câu hỏi mẫu nằm trong [`data/recruitment_data/README.md`](data/recruitment_data/README.md).

## Lưu ý triển khai

- Các module hiện sử dụng đường dẫn tuyệt đối `/mnt/data/build_chatbot`; nếu chuyển dự án sang máy khác, cần cập nhật `BASE_DIR` và `DB_PATH` trong backend.
- Không nên commit thông tin truy cập hoặc endpoint nội bộ khi triển khai ra môi trường công khai.
- LLM được giới hạn một lượt inference đồng thời để phù hợp với việc dùng chung một GPU.
- Backend cần được khởi động thành công trước frontend; frontend sẽ báo lỗi kết nối nếu không gọi được `/health` hoặc `/chat`.
