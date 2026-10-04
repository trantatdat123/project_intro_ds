# Tạo lại slide

`build_slides.mjs` đọc metrics và bảng của pipeline rồi tạo slide PowerPoint có bảng/biểu đồ chỉnh sửa được. Chạy `python main.py --stage all` trước; script dừng nếu chưa có kết quả semantic clustering. Slide là snapshot của lượt chạy tạo deck, cần tạo lại khi dữ liệu/kết quả đổi.

Builder dùng `@oai/artifact-tool` và bộ công cụ finalization của skill Presentations trong runtime Codex. Đây là công cụ tạo tài liệu, không thuộc dependencies của pipeline Python. Có thể chỉnh sửa trực tiếp deck đã xuất bằng PowerPoint.

Trong môi trường Codex hiện tại, cần:

1. Tạo `.build/slides/node_modules` dạng junction tới thư mục packages của runtime Node.
2. Copy `scripts/build_slides.mjs` vào `.build/slides/build.mjs`.
3. Đặt `JOBLENS_ROOT`, `JOBLENS_PRESENTATION_SKILL`, `JOBLENS_RUNTIME_PYTHON` thành đường dẫn tuyệt đối; đặt `RUNTIME_NODE_MODULES` tới thư mục packages của runtime để finalizer đọc lại deck.
4. Chạy `mark_artifact_operation_started.mjs --operation-kind create --expected-output-count 1 --output-format pptx` từ thư mục skill trước lượt tạo.
5. Chạy builder bằng Node của runtime. Kết quả: `reports/slides/joblens_vietnam.pptx`.

Không ghi đè deck đã có: đặt `JOBLENS_SLIDES_OUTPUT` thành tên file mới cho lượt tiếp theo. Preview PNG và validation receipt được lưu riêng trong `.build/slides/`.
