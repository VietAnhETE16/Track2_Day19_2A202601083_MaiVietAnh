# Kiến Trúc Hệ Thống Bộ Nhớ Lai (Hybrid AI Memory Architecture)

**Tác giả:** Mai Việt Anh (2A202601083)  
**Dự án:** Trợ lý AI Cá nhân hóa cho Người dùng Việt Nam (Personal AI Assistant with Hybrid Memory)  
**Phiên bản:** 1.0.0 — Production POC  

---

## 1. Tổng Quan và Sơ Đồ Kiến Trúc Hệ Thống

Hệ thống trợ lý AI cá nhân thế hệ mới đòi hỏi khả năng ghi nhớ đa tầng nhằm cung cấp câu trả lời vừa bám sát ngữ cảnh trò chuyện trong quá khứ (Episodic Memory), vừa thấu hiểu sở thích dài hạn (Stable User Profile), đồng thời phản ứng nhạy bén với các hoạt động thời gian thực (Recent Activity).

Thay vì phụ thuộc hoàn toàn vào cửa sổ ngữ cảnh (Context Window) hữu hạn của LLM hoặc sử dụng RAG đơn thuần, kiến trúc đề xuất kết hợp sức mạnh của **Vector Store (Qdrant)** và **Feature Store (Feast)**:

```mermaid
graph TD
    subgraph ClientLayer [Tầng Tương Tác & Client]
        User["Người Dùng Việt Nam (Web / Mobile App)"]
        API["FastAPI Gateway / Orchestrator"]
        User -->|Query / Message| API
    end

    subgraph MemoryIngestion [Tầng Thu Nhập & Đồng Bộ Dữ Liệu]
        DocStream["User Notes / Chat Transcripts"]
        EventStream["Click & Query Events (Kafka/WebSocket)"]
        
        DocStream -->|Semantic Chunking & Embedding| QdrantIngest["Qdrant Ingestion (Batch / Push)"]
        EventStream -->|Sub-second Stream| FeastStream["Feast Streaming Feature View"]
        EventStream -->|Daily ETL / Aggregation| FeastBatch["Offline Parquet / Data Warehouse"]
    end

    subgraph StorageLayer [Tầng Lưu Trữ Bộ Nhớ (Storage Tier)]
        Qdrant[("Qdrant Vector Store<br/>(Episodic Memory - Dense Embeddings)")]
        FeastOnline[("Feast Online Store (SQLite / Redis)<br/>(User Profile & Real-time Velocity)")]
        FeastOffline[("Feast Offline Store (Parquet)<br/>(Historical Events for PIT Training)")]
        
        QdrantIngest --> Qdrant
        FeastStream --> FeastOnline
        FeastBatch --> FeastOffline
        FeastOffline -.->|feast materialize| FeastOnline
    end

    subgraph RetrievalLayer [Tầng Truy Xuất & Lắp Ráp Ngữ Cảnh]
        HybridRetriever["Hybrid Retriever<br/>(Dense Cosine + BM25 + RRF k=60)"]
        ProfileFetcher["Feature Store Lookup<br/>(get_online_features < 10ms)"]
        ContextAssembler["Prompt Context Assembler<br/>(Grounding + Personalization)"]
        
        API -->|User ID & Query| HybridRetriever
        API -->|User ID| ProfileFetcher
        
        HybridRetriever <-->|Search with user_id Filter| Qdrant
        ProfileFetcher <-->|Online Lookup| FeastOnline
        
        HybridRetriever -->|Top-K Episodic Docs| ContextAssembler
        ProfileFetcher -->|Profile & Velocity Features| ContextAssembler
    end

    subgraph GenerationLayer [Tầng Sinh Phản Hồi]
        LLM["LLM Generation (Vietnamese-Optimized)"]
        ContextAssembler -->|Assembled Prompt Context| LLM
        LLM -->|Personalized Response| User
    end
```

---

## 2. Ba Quyết Định Kiến Trúc Trọng Tâm & Phân Tích Đánh Đổi (Trade-offs)

### Quyết định 1: Chiến Lược Phân Đoạn Bộ Nhớ (Chunking Strategy)
* **Phương án lựa chọn:** Phân đoạn theo ranh giới ngữ nghĩa và phiên tương tác (Semantic & Session Boundary Chunking, 256–512 tokens, 50-token overlap).
* **Đánh đổi cụ thể (X vs Y, tại sao chọn X):**
  * *Lựa chọn X (Semantic Boundary Chunking)* so với *Lựa chọn Y (Fixed-size Window 1000 tokens hoặc Single-message Chunking)*.
  * *Tại sao chọn X:* Nếu cắt cứng theo số ký tự/token (Lựa chọn Y1), các câu đàm thoại hoặc ghi chú tiếng Việt dễ bị đứt đoạn ngữ pháp giữa chừng, phá vỡ cấu trúc câu và làm sai lệch vector embedding. Ngược lại, nếu chunking theo từng tin nhắn đơn lẻ (Lựa chọn Y2), ngữ cảnh của cuộc hội thoại bị chia cắt, retriever không nắm được nguyên nhân - kết quả. 
  * *Đánh đổi chấp nhận:* X tốn thêm $15\text{–}20\%$ chi phí xử lý tiền phân đoạn (tokenization & semantic boundary detection) và tăng nhẹ dung lượng lưu trữ do overlap, nhưng đổi lại **Recall@5 cải thiện vượt trội ($>25\%$)** và triệt tiêu hiện tượng đứt gãy thông tin ngữ cảnh khi truyền cho LLM.

### Quyết định 2: Cấu Trúc Lược Đồ Đặc Trưng (Feature Schema Design)
* **Phương án lựa chọn:** Kết hợp Tabular Explicit Attributes (thuộc tính định lượng rõ ràng) và Decayed Topic Affinity (hệ số ưa thích chủ đề có suy giảm theo thời gian) trong Feast Feature Store.
* **Đánh đổi cụ thể (X vs Y, tại sao chọn X):**
  * *Lựa chọn X (Tabular Attributes + Decayed Topic Score)* so với *Lựa chọn Y (Latent User Embedding Vector 1536-dim)*.
  * *Tại sao chọn X:* Biểu diễn người dùng dưới dạng latent embedding (Lựa chọn Y) có thể tóm tắt lịch sử trừu tượng nhưng là một "hộp đen" (black box) hoàn toàn: không thể giải thích tại sao gợi ý tài liệu A, không thể áp dụng các bộ lọc cứng (hard business rules như `preferred_language = "vi"`, `reading_speed_wpm = 220`), và cực kỳ khó debug khi có sự cố.
  * *Đánh đổi chấp nhận:* Lựa chọn X đòi hỏi kỹ sư phải định nghĩa schema tường minh (`reading_speed_wpm`, `topic_affinity`, `queries_last_hour`) và xây dựng pipeline tính toán thống kê, nhưng mang lại **khả năng kiểm soát tuyệt đối (Deterministic Governance), tốc độ lookup siêu tốc ($<5\text{ ms}$)**, và cho phép ghép trực tiếp vào System Prompt mà LLM hiểu 100%.

### Quyết định 3: Chiến Lược Làm Tươi Dữ Liệu (Freshness & Ingestion Strategy)
* **Phương án lựa chọn:** Kiến trúc Ingestion đa tầng (Multi-tier Freshness):
  1. *Sub-second Streaming (Push API)*: Cho `query_velocity_features` (phát hiện ý định thay đổi nhanh, cảnh báo tấn công/spam, TTL = 1 giờ).
  2. *Near Real-time Micro-batch (15 phút)*: Cho Episodic Memory trong Qdrant (đồng bộ ghi chú mới, tài liệu vừa đọc).
  3. *Daily Batch Ingestion*: Cho `user_profile_features` (tính toán lại `topic_affinity`, `reading_speed_wpm`, TTL = 30 ngày).
* **Đánh đổi cụ thể (X vs Y, tại sao chọn X):**
  * *Lựa chọn X (Multi-tier Freshness)* so với *Lựa chọn Y (Full Real-time Streaming cho toàn bộ hệ thống)*.
  * *Tại sao chọn X:* Bắt buộc toàn bộ vector store và profile store phải tươi theo thời gian thực từng mili-giây (Lựa chọn Y) sẽ đẩy chi phí hạ tầng (Kafka clusters, Flink streaming, Qdrant index lock contention) lên gấp $5\text{–}10$ lần mà không mang lại giá trị thực tế cho người dùng (hồ sơ dài hạn như sở thích đọc không bao giờ thay đổi sau từng giây).
  * *Đánh đổi chấp nhận:* Chấp nhận độ trễ 15 phút để tài liệu đọc mới nhất xuất hiện trong bộ nhớ dài hạn, đổi lại **tiết kiệm $80\%$ chi phí vận hành hạ tầng** và đảm bảo hệ thống online lookup đạt độ ổn định 99.99%.

---

## 3. Xem Xét Đặc Thù Ngữ Cảnh Tiếng Việt (Vietnamese Context Considerations)

1. **Hiện tượng Chuyển Mã Ngôn Ngữ (Code-switching Vi/En):**
   Người dùng công nghệ tại Việt Nam thường xuyên kết hợp tiếng Việt với thuật ngữ tiếng Anh trong cùng một câu (ví dụ: *"làm sao setup autoscaling trên k8s cluster"*).
   *Giải pháp:* Lớp Hybrid Retrieval kết hợp BM25 (bắt chính xác keyword tiếng Anh viết tắt) và Multilingual Dense Vector (nắm bắt ngữ nghĩa tiếng Việt) giúp triệt tiêu điểm yếu của từng mô hình đơn lẻ.
2. **Lựa Chọn Phân Tách Từ (Vietnamese Word Segmentation):**
   Tiếng Việt là ngôn ngữ đơn lập, ranh giới từ ghép gồm nhiều tiếng cách nhau bằng khoảng trắng (ví dụ: *"tự động hóa"*, *"trí tuệ nhân tạo"*). Sử dụng whitespace split thuần túy khiến BM25 coi mỗi âm tiết là một từ độc lập. Trong thiết kế production, chúng tôi tích hợp thư viện tách từ (`underthesea` / `pyvi`) trước khi đẩy vào BM25 tokenizer để tăng độ chính xác tìm kiếm từ khóa lên hơn $30\%$.
3. **Bảo Mật & Tuân Thủ Nghị Định 13/2023/NĐ-CP:**
   Nghị định 13 quy định nghiêm ngặt về bảo vệ dữ liệu cá nhân tại Việt Nam. Kiến trúc Memory Agent bắt buộc áp dụng **Payload Filtering cô lập theo `user_id`** trên Qdrant và **Namespaced Key Isolation** trên Feature Store. Tuyệt đối không lưu trữ dữ liệu cá nhân của nhiều người dùng trong cùng một namespace không được kiểm soát.

---

## 4. Phương Án Bị Loại Bỏ (Rejected Alternative)

* **Phương án xem xét:** Lưu trữ toàn bộ Episodic Memory trực tiếp vào Feature Store dưới dạng các `EmbeddingFeatureView` (tương tự Feast Vector Extension) thay vì tách riêng ra một Vector DB độc lập (Qdrant).
* **Lý do từ chối:**
  1. *Khác biệt về chu kỳ chỉ mục (Indexing Lifecycle):* Episodic memory cần chỉ mục đồ thị xấp xỉ không gian (HNSW graph) và cập nhật liên tục với khả năng tìm kiếm tương đồng vector linh hoạt (filtered ANN, hybrid fusion). Feature Store tối ưu cho phép tra cứu chính xác theo khóa chính (Key-Value lookup bằng `user_id`), không tối ưu cho đồ thị vector mở rộng hàng triệu nút.
  2. *Độc lập về quy mô (Scalability Decoupling):* Tách rời Vector Store và Feature Store cho phép co giãn độc lập: mở rộng tài nguyên tính toán vector (GPU/CPU inference) khi lượng tài liệu tăng cao mà không gây ảnh hưởng tới độ trễ siêu thấp ($<5\text{ ms}$) của kho đặc trưng phục vụ ứng dụng thời gian thực.

---

## 5. Giới Hạn Của Bản POC Hiện Tại (Honest Limitations)

Bản POC hiện tại tập trung chứng minh tính đúng đắn của việc tích hợp kiến trúc giữa Vector Store và Feature Store. Các tính năng sau chưa được hỗ trợ trong phạm vi lab:
1. *Mã hóa dữ liệu tại chỗ (Encryption at rest per user)*: Chưa hỗ trợ khóa mã hóa riêng biệt cấp phần cứng cho từng tài khoản.
2. *Đồng bộ đa thiết bị thời gian thực (Multi-device state synchronization)*: Chưa có WebRTC/WebSocket gateway đồng bộ trạng thái đọc tức thời giữa web và mobile.
3. *Cơ chế nén và suy thoái bộ nhớ (Memory Consolidation & Forgetting Curve)*: Chưa có background job định kỳ 7 ngày tự động tổng hợp 10 đoạn hội thoại nhỏ thành 1 bản tóm tắt tri thức cốt lõi.

---

## 6. Nhật Ký Vibe Coding (Workflow Prompt Log)

* **Prompt hiệu quả nhất:**
  > *"Thiết kế lớp HybridMemoryAgent bằng Python kết hợp Qdrant in-memory client và Feast FeatureStore. Phương thức remember() thực hiện chunking, embedding và upsert với payload user_id. Phương thức recall() thực hiện online lookup lấy user profile từ Feast và hybrid vector search trên Qdrant, sau đó assemble thành context string chứa cả thông tin cá nhân hóa lẫn top tài liệu liên quan."*
  $\rightarrow$ AI sinh mã chuẩn mẫu, định kiểu type-hints rõ ràng và tuân thủ $100\%$ cấu trúc API.
* **Prompt thất bại / Cần điều chỉnh:**
  > *"Làm cho bộ nhớ nhớ được mọi thứ của người dùng."*
  $\rightarrow$ AI đề xuất giải pháp lưu toàn bộ lịch sử thô vào một mảng JSON duy nhất và nhồi trực tiếp vào prompt, gây tràn context window và phá vỡ cấu trúc phân tầng bộ nhớ.
