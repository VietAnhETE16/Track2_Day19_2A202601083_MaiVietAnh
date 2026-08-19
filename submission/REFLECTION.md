# Reflection — Lab 19

**Tên:** Mai Việt Anh  
**Mã học viên:** 2A202601083  
**Cohort:** K3  
**Path đã chạy:** Lite  

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` /
> `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid
> (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

Trên tập 50 queries thực nghiệm:
- **`exact` (BM25: 96.7%, Hybrid: 96.7%):** BM25 bắt chính xác thuật ngữ kỹ thuật nguyên văn; vector embedding bị loãng do không phân biệt được từ khóa chuyên biệt.
- **`paraphrase` (Hybrid: 32.0%, BM25: 33.3%, Vector: 24.0%):** Câu hỏi diễn đạt lại bằng từ đồng nghĩa; Hybrid tổng hòa tốt nhất các tín hiệu ngữ cảnh.
- **`mixed` (Hybrid: 100.0%, Vector: 98.5%, BM25: 97.0%):** Hybrid thắng tuyệt đối nhờ thuật toán RRF ($k=60$) kết hợp đồng thời độ khớp từ khóa và tương đồng ngữ nghĩa.

**Khi nào KHÔNG dùng Hybrid Search?**
1. **Dùng thuần BM25:** Khi tìm kiếm mã định danh, số hợp đồng, SKU sản phẩm hoặc thuật ngữ pháp lý/y tế đòi hỏi đối soát nguyên văn 100% (exact match) mà không cần suy diễn ngữ nghĩa.
2. **Dùng thuần Vector:** Khi truy vấn đa phương thức (hình ảnh, âm thanh), tìm kiếm đa ngôn ngữ (cross-lingual), hoặc câu hỏi mang tính khái niệm trừu tượng không có từ khóa cố định.
3. **Giới hạn tài nguyên/độ trễ siêu thấp (<5ms):** Hybrid bắt buộc chạy cả 2 lượt truy xuất + fusion, tăng gấp đôi tải tính toán.

---

## Điều ngạc nhiên nhất khi làm lab này

Hiện tượng rò rỉ dữ liệu Target Encoding trên khóa có cardinality cao (`session_id`) tạo ra train AUC tới 0.992 nhưng test AUC chỉ 0.518, và tính cấp thiết của Point-in-Time join trong Feast để loại bỏ lift ảo.

---

## Bonus challenge

- [x] Đã làm bonus (xem `bonus/`)
- [ ] Pair work với: _Làm độc lập_
