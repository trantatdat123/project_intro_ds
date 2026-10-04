"""EDA, biểu đồ và báo cáo từ kết quả thực; không điền số liệu giả khi thiếu stage."""

from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import ROOT, write_json


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "figure.dpi": 140,
                     "axes.spines.top": False, "axes.spines.right": False})
COLORS = ["#176B87", "#46A58B", "#E4A14A", "#845EC2", "#C75B72", "#64748B"]


def _save(figure, name, report_dir):
    folder = Path(report_dir) / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    figure.savefig(folder / name, bbox_inches="tight")
    plt.close(figure)


def generate_eda(frame, report_dir=ROOT / "reports"):
    report_dir = Path(report_dir)
    tables = report_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    roles = frame.role_standard.value_counts().rename_axis("role").reset_index(name="jobs")
    years = frame.groupby("year").agg(jobs=("id", "size"), median_salary_vnd=("salary_mean", "median"),
                                      salary_rows=("salary_mean", "count")).reset_index()
    skills = Counter(skill for row in frame.extracted_skills for skill in set(row))
    skill_table = pd.DataFrame(skills.most_common(), columns=["skill", "jobs"])
    skill_table["share"] = skill_table.jobs / len(frame)
    salaries = frame.groupby("role_standard").salary_mean.agg(["count", "median", "mean"]).reset_index()
    for name, table in (("role_distribution", roles), ("year_distribution", years),
                        ("skill_frequency", skill_table), ("salary_by_role", salaries)):
        table.to_csv(tables / f"{name}.csv", index=False, encoding="utf-8-sig")
    figure, axes = plt.subplots(figsize=(9, 4))
    axes.barh(roles.role.iloc[::-1], roles.jobs.iloc[::-1], color=COLORS[::-1])
    axes.set(xlabel="Số tin trong dữ liệu", title="Nhóm nghề từ quy tắc title")
    _save(figure, "roles.png", report_dir)
    figure, axes = plt.subplots(figsize=(8, 4))
    axes.bar(years.year.astype(str), years.jobs, color=COLORS[0])
    axes.set(xlabel="Năm trong nguồn", ylabel="Số tin", title="Phân bố năm của bộ dữ liệu đã lọc")
    _save(figure, "years.png", report_dir)
    figure, axes = plt.subplots(figsize=(9, 6))
    top = skill_table.head(20).iloc[::-1]
    axes.barh(top.skill, top.jobs, color=COLORS[1])
    axes.set(xlabel="Số JD đề cập", title="20 kỹ năng xuất hiện nhiều nhất")
    _save(figure, "skills.png", report_dir)
    valid = frame.loc[frame.salary_eligible]
    groups = [(role, group.salary_mean.to_numpy() / 1e6) for role, group in valid.groupby("role_standard") if len(group) >= 5]
    if groups:
        figure, axes = plt.subplots(figsize=(10, 4))
        axes.boxplot([values for _, values in groups], tick_labels=[role for role, _ in groups], showfliers=False)
        axes.tick_params(axis="x", rotation=20)
        axes.set(ylabel="Triệu VND/tháng", title="Midpoint lương niêm yết theo nhóm nghề")
        _save(figure, "salary_roles.png", report_dir)
    figure, axes = plt.subplots(figsize=(8, 4))
    axes.hist(frame.experience_min.dropna(), bins=np.arange(0, 22), color=COLORS[2])
    axes.set(xlabel="Cận dưới kinh nghiệm (năm)", ylabel="Số tin", title="Kinh nghiệm tối thiểu; loại giá trị thiếu")
    _save(figure, "experience.png", report_dir)
    return {"roles": roles, "years": years, "skills": skill_table, "salaries": salaries}


def classification_figures(metrics, report_dir=ROOT / "reports"):
    labels = metrics["labels"]
    matrix = np.array(metrics["confusion_matrix"])
    figure, axes = plt.subplots(figsize=(8, 6))
    image = axes.imshow(matrix, cmap="Blues")
    axes.set_xticks(range(len(labels)), labels=labels, rotation=30, ha="right")
    axes.set_yticks(range(len(labels)), labels=labels)
    for row, column in np.ndindex(matrix.shape):
        axes.text(column, row, str(matrix[row, column]), ha="center", va="center",
                  color="white" if matrix[row, column] > matrix.max() / 2 else "black")
    axes.set(xlabel="Dự đoán", ylabel="Nhãn từ title", title="Confusion matrix trên tập test khác công ty")
    figure.colorbar(image, ax=axes, fraction=0.04)
    _save(figure, "classification_confusion.png", report_dir)


def salary_figures(actual, predicted, shap_summary, report_dir=ROOT / "reports"):
    actual, predicted = np.asarray(actual) / 1e6, np.asarray(predicted) / 1e6
    figure, axes = plt.subplots(figsize=(6, 5))
    axes.scatter(actual, predicted, s=10, alpha=0.35, color=COLORS[0])
    maximum = max(actual.max(), predicted.max())
    axes.plot([0, maximum], [0, maximum], color=COLORS[2], linewidth=1.5)
    axes.set(xlabel="Midpoint niêm yết (triệu VND/tháng)", ylabel="Dự đoán (triệu VND/tháng)", title="Dự đoán lương trên tập test")
    _save(figure, "salary_predictions.png", report_dir)
    figure, axes = plt.subplots(figsize=(9, 5))
    top = shap_summary.head(15).iloc[::-1]
    axes.barh(top.feature.str.replace(r"^[^_]+__", "", regex=True), top.mean_abs_shap_log, color=COLORS[1])
    axes.set(xlabel="Mean |SHAP| trên log1p(lương)", title="Mức ảnh hưởng trong mô hình lương")
    _save(figure, "salary_shap.png", report_dir)


def graph_figure(graph, report_dir=ROOT / "reports"):
    import networkx as nx
    strongest = sorted(graph.nodes, key=lambda node: graph.nodes[node]["job_count"], reverse=True)[:25]
    subgraph = graph.subgraph(strongest).copy()
    # Chỉ vẽ 40 cạnh mạnh nhất; GraphML vẫn giữ toàn bộ cạnh đã lọc.
    keep = sorted(subgraph.edges, key=lambda edge: subgraph.edges[edge]["weight"], reverse=True)[:40]
    subgraph.remove_edges_from([edge for edge in list(subgraph.edges) if edge not in keep])
    figure, axes = plt.subplots(figsize=(11, 8))
    position = nx.spring_layout(subgraph, seed=42, k=0.7, iterations=80)
    nx.draw_networkx(subgraph, pos=position, ax=axes, node_color=COLORS[1], font_size=9,
                     node_size=[250 + subgraph.nodes[node]["job_count"] / 8 for node in subgraph],
                     edge_color="#B5C4CF", width=0.7)
    axes.axis("off")
    axes.set_title("Đồng xuất hiện kỹ năng trong JD")
    _save(figure, "skill_network.png", report_dir)


def cluster_figure(reduced, labels, report_dir=ROOT / "reports"):
    figure, axes = plt.subplots(figsize=(8, 6))
    scatter = axes.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab10", s=4, alpha=0.5)
    axes.set(xlabel="Tọa độ UMAP 1", ylabel="Tọa độ UMAP 2", title="KMeans trên UMAP (15 chiều)")
    figure.colorbar(scatter, ax=axes, label="Cụm")
    _save(figure, "clusters.png", report_dir)


def markdown_table(frame, columns=None):
    if columns is not None:
        frame = frame[columns]
    values = frame.copy().fillna("—")
    def cell(value):
        if isinstance(value, float):
            return f"{value:,.3f}"
        return str(value).replace("|", "/").replace("\n", " ")
    rows = [[cell(value) for value in row] for row in values.to_numpy()]
    return "| " + " | ".join(values.columns) + " |\n| " + " | ".join(["---"] * len(values.columns)) + " |\n" + "\n".join("| " + " | ".join(row) + " |" for row in rows)


def generate_report(frame, config, quality, classification=None, salary=None, cluster=None, report_dir=ROOT / "reports"):
    report_dir = Path(report_dir)
    summary = generate_eda(frame, report_dir)
    salary_table = summary['salaries'].rename(columns={'role_standard': 'Nghề', 'count': 'Số tin',
                           'median': 'Trung vị (triệu VND/tháng)', 'mean': 'Trung bình (triệu VND/tháng)'})
    salary_table.iloc[:, 2:] = salary_table.iloc[:, 2:] / 1e6
    source_link = "https://huggingface.co/datasets/tinixai/vietnamese-job-descriptions"
    sections = ["# JobLens Vietnam: Báo cáo IntroDS", "## 1. Mục tiêu và dữ liệu",
        f"Phân tích title, kỹ năng, nhóm nghề và lương niêm yết của **{len(frame):,}** tin đã lọc từ [TinixAI]({source_link}). Bộ lọc IT/AI/Data trong notebook dựa trên title; dữ liệu gốc, nội dung raw và năm được giữ riêng với các cột đã xử lý.",
        "Dữ liệu không phải mẫu ngẫu nhiên của toàn thị trường. Mức phủ giữa các năm, việc đăng lại tin và thay đổi nguồn có thể ảnh hưởng số dòng; năm 2026 có thể chưa đủ cả năm. `year` là năm do nguồn cung cấp, không có ngày/tháng để xác minh thời điểm riêng từng tin.",
        "License của nguồn: CC BY-NC 4.0. Ghi công TiniX AI; dataset card dẫn công trình CareerPathKG (Le và cộng sự, 2026).",
        "## 2. Tiền xử lý", f"Từ điển có {len(__import__('json').loads((ROOT / 'configs/skill_dictionary.json').read_text(encoding='utf-8'))):,} tên kỹ năng chuẩn và các alias. FlashText khớp theo ranh giới từ. HTML, entity và khoảng trắng được làm sạch; `C++`, `C#`, `.NET` được giữ khi trích kỹ năng.",
        f"Kinh nghiệm thiếu: **{quality['experience_missing_rows']:,}** tin. `Không` đơn lẻ được để thiếu; `không yêu cầu kinh nghiệm` được lấy là 0. Khoảng kinh nghiệm được biểu diễn bằng cận dưới, không phải số năm thực tế của ứng viên.",
        f"Lương đọc được để tạo midpoint: **{quality['salary_eligible_rows']:,}** tin. Khoảng đóng dùng trung bình hai đầu; lương thỏa thuận, giá trị không hợp lệ và cận mở không được điền một midpoint giả. USD quy đổi bằng giả định cố định **{config['usd_to_vnd_rate']:,} VND/USD**, không phải tỷ giá lịch sử. Lương chưa ghi đơn vị thời gian được giả định là tháng; gross/net và lạm phát chưa được hiệu chỉnh.",
        f"Có **{quality['duplicate_content_rows']:,}** dòng nằm trong nhóm nội dung trùng theo công ty, title, nội dung và năm. Dữ liệu EDA giữ các dòng này để phản ánh nguồn; mô hình chia theo công ty để giảm rò rỉ do đăng lại cùng JD.",
        "## 3. EDA", "### Nhóm nghề", markdown_table(summary['roles']),
        "![Nhóm nghề](figures/roles.png)", "### Phân bố năm", markdown_table(summary['years']),
        "![Phân bố năm](figures/years.png)", "### Kỹ năng phổ biến", markdown_table(summary['skills'].head(15)),
        "![Kỹ năng](figures/skills.png)", "### Lương theo nhóm", markdown_table(salary_table),
        "![Lương theo nhóm](figures/salary_roles.png)", "## 4. Thiết kế đánh giá",
        "Train/validation/test được chia theo `company_group` với seed cố định, khoảng 60/20/20 theo số công ty. Tỷ lệ theo số tin có thể khác. Các công ty đã nhận diện không xuất hiện ở hai tập; tên công ty viết khác nhau vẫn có thể thuộc cùng thực thể. Tin thiếu công ty được nhóm theo fingerprint nội dung.",
        "Nhãn nghề tạo từ title bằng quy tắc, không phải nhãn chuyên gia. Tin không thuộc rõ một trong năm nhóm hoặc có nhiều nhóm được ghi `Other`, giữ cho EDA và loại khỏi bài toán phân loại năm nhóm. Mô hình phân loại chỉ dùng mô tả/yêu cầu; không đưa title, matched_groups hoặc role_label vào đầu vào. F1 đo mức khớp nhãn sơ bộ, không chứng minh độ chính xác với nhãn đã kiểm định.",
        "TF-IDF, imputer, categorical encoder và vocabulary kỹ năng của mô hình chỉ fit trên train. Chọn mô hình theo validation; test dùng để báo cáo. Nhãn title yếu và các dấu hiệu chức danh trong JD vẫn có thể giúp mô hình khớp quy tắc.",
    ]
    if classification:
        comparison = pd.DataFrame([{ "Mô hình": name, "F1 validation": item['validation']['macro_f1'],
                                    "F1 test": item['test']['macro_f1'], "Precision macro test": item['test']['precision_macro'],
                                    "Recall macro test": item['test']['recall_macro']} for name, item in classification['models'].items()])
        sections += ["## 5. Phân loại nghề", markdown_table(comparison),
                     f"Chọn **{classification['selected_model']}** theo macro-F1 validation. Baseline dự đoán lớp phổ biến trên train có macro-F1 test **{classification['baseline_macro_f1']:.3f}**.",
                     "![Confusion matrix](figures/classification_confusion.png)"]
        selected_test = classification['models'][classification['selected_model']]['test']
        per_class = pd.DataFrame([{ "Nghề": name, **scores} for name, scores in selected_test['classification_report'].items()
                                  if name in selected_test['labels']])
        sections += ["### Precision/recall từng nghề trên test", markdown_table(per_class),
                     f"Số tin phân loại: train={classification['split_rows']['train']:,}, validation={classification['split_rows']['validation']:,}, test={classification['split_rows']['test']:,}. Các nhóm ít mẫu cần được đọc cùng support; F1 của chúng kém ổn định hơn."]
    else:
        sections += ["## 5. Phân loại nghề", "Chưa có kết quả thực; chạy stage classify."]
    if salary:
        comparison = pd.DataFrame([{ "Mô hình": name, "MAE validation (triệu VND)": item['validation']['mae_vnd'] / 1e6,
                                    "MAE test (triệu VND)": item['test']['mae_vnd'] / 1e6,
                                    "RMSE test (triệu VND)": item['test']['rmse_vnd'] / 1e6,
                                    "R² test": item['test']['r2']} for name, item in salary['models'].items()])
        sections += ["## 6. Dự đoán lương", markdown_table(comparison),
            f"Chọn **{salary['selected_model']}** theo MAE validation. Baseline trung vị train có MAE test **{salary['baseline_mae_vnd'] / 1e6:.2f} triệu VND**. Bài toán dùng tin có một nhóm nghề xác định và midpoint hợp lệ.",
            f"Ngưỡng cắt đuôi lương được học từ train: {salary['training_bounds_vnd'][0]:,.0f}–{salary['training_bounds_vnd'][1]:,.0f} VND. Validation/test giữ các midpoint hợp lệ, kể cả nằm ngoài ngưỡng này.",
            "Mô hình học `log1p(midpoint)`, đánh giá MAE/RMSE sau biến đổi về VND. Dự đoán là midpoint lương niêm yết; không phải lương thực nhận hay lời hứa trả lương.",
            "![Dự đoán lương](figures/salary_predictions.png)", "### Giải thích bằng SHAP",
            f"Tree SHAP được tính cho **{salary['shap_model']}** trên thang log1p(lương), với {salary.get('shap_rows', config['shap_sample_size'])} tin test. Mean |SHAP| đo mức ảnh hưởng trong mô hình; dấu dương/âm không phải phần bù lương nhân quả. Kỹ năng tương quan có thể chia sẻ đóng góp. Không diễn giải kết quả này thành mức tăng lương khi học thêm một kỹ năng.",
            "![SHAP](figures/salary_shap.png)"]
    else:
        sections += ["## 6. Dự đoán lương", "Chưa có kết quả thực; chạy stage salary."]
    if cluster:
        sections += ["## 7. Mạng kỹ năng và cụm JD", f"Đồ thị có **{cluster['graph']['nodes']}** kỹ năng và **{cluster['graph']['edges']}** cạnh đồng xuất hiện sau ngưỡng lọc. Cạnh có count, Jaccard và lift; không suy ra quan hệ phụ thuộc kỹ thuật từ việc cùng được nhắc trong JD.",
                     "![Mạng kỹ năng](figures/skill_network.png)"]
        centrality_path = report_dir / "skill_centrality.csv"
        if centrality_path.exists():
            sections += ["### Các nút có tổng số lần đồng xuất hiện lớn nhất",
                         markdown_table(pd.read_csv(centrality_path).head(10)),
                         "Weighted degree là tổng count cạnh; betweenness được tính trên đồ thị không trọng số, vì count lớn không biểu thị khoảng cách lớn. Centrality phụ thuộc vào ngưỡng giữ nút/cạnh."]
        if cluster.get('semantic'):
            sections += ["Sentence embeddings chuẩn hóa được giảm chiều bằng UMAP. KMeans và HDBSCAN mô tả cấu trúc nội dung; không được xem là nhóm nghề đúng đã xác minh. Text bị cắt ở max_seq_length của embedding model; JD dài có thể mất phần cuối.",
                         markdown_table(pd.DataFrame(cluster['semantic'].values())), "![Các cụm](figures/clusters.png)"]
        else:
            sections += ["Chưa chạy sentence embeddings/semantic clustering trong lần chạy này."]
    sections += ["## 8. Kết luận và hướng mở rộng",
        "Bộ dữ liệu cho phép mô tả kỹ năng được đề cập, lương niêm yết và các nhóm nghề trong phạm vi title đã lọc. Kết quả phân loại cần được đánh giá thêm bằng mẫu gán nhãn thủ công; lương cần tách gross/net, thời gian và đơn vị chính xác hơn trước khi so sánh xu hướng năm.",
        "Bước tiếp theo: rà soát title giữ/bỏ, gán nhãn nghề thủ công bằng hai người độc lập, chuẩn hóa công ty để giảm alias, đánh giá holdout theo năm, bootstrap khoảng tin cậy và nghiên cứu sai lệch do các tin không công khai lương.",
        "## 9. Chạy lại và artifacts", "`python main.py --stage all` chạy ingestion, preprocessing, features, EDA, phân loại, hồi quy, graph/clustering và tạo báo cáo. `python main.py --stage all --no-embeddings` chạy các phần không cần tải sentence model; kết quả sẽ ghi rõ phần semantic chưa chạy. Cấu hình ở `configs/pipeline.json`.",
        "Mô hình lưu trong `artifacts/models/`, metrics và bảng nằm trong `reports/`, dữ liệu sạch trong `data/processed/jobs_cleaned.csv`/Parquet. Notebook EDA, modeling và clustering dùng lại các module này."]
    findings = []
    if classification:
        chosen = classification['models'][classification['selected_model']]['test']
        findings.append(f"**{classification['selected_model']}** đạt macro-F1 test **{chosen['macro_f1']:.3f}** trên {chosen['rows']:,} tin có nhãn nghề đơn từ title. Đây là mức khớp nhãn sơ bộ; cần một tập nhãn độc lập do người kiểm định để đánh giá độ đúng về nghề.")
    if salary:
        chosen = salary['models'][salary['selected_model']]['test']
        improvement = 100 * (1 - chosen['mae_vnd'] / salary['baseline_mae_vnd'])
        findings.append(f"**{salary['selected_model']}** có MAE test **{chosen['mae_vnd']/1e6:.2f} triệu VND/tháng**, R² **{chosen['r2']:.3f}**, giảm MAE {improvement:.1f}% so với baseline trung vị train trên cùng tập test. Sai số vẫn lớn so với nhiều midpoint lương trong dữ liệu; dự đoán cần được đọc như ước lượng có sai số.")
    insertion = sections.index("## 8. Kết luận và hướng mở rộng") + 1
    sections[insertion:insertion] = findings
    sections += ["## 10. Tài liệu phương pháp",
                 "[GroupShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html), [LightGBM](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html), [TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html), [SentenceTransformer](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html).",
                 "Phiên bản thư viện, Python, revision nguồn và fingerprint dữ liệu được ghi trong `run_manifest.json`. Các bảng và slide là snapshot của lượt chạy; cần tạo lại khi thay đổi dữ liệu/cấu hình."]
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "project_report.md"
    path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    return path
