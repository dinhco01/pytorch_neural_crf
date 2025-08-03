import streamlit as st
import pandas as pd
import random
import nltk
from typing import Dict, List
import plotly.express as px
from collections import Counter

try:
    from transformers_predictor import TransformersNERPredictor
    from ner_predictor import NERPredictor
except ImportError:
    st.error("Không thể import các module.")
    st.stop()

st.set_page_config(
    page_title="Named Entity Recognition cho văn bản Lịch Sử Việt Nam",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }

    .entity-box {
        padding: 4px 10px;
        border-radius: 15px;
        margin: 3px;
        display: inline-block;
        font-weight: bold;
        color: white;
        font-size: 0.9rem;
        box-shadow: 0 3px 6px rgba(0,0,0,0.25);
        transition: all 0.3s ease;
        border: 1px solid rgba(255,255,255,0.3);
        text-shadow: 0 1px 2px rgba(0,0,0,0.3);
    }

    .entity-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.4);
        filter: brightness(1.1);
    }

    .entity-PER {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 50%, #a93226 100%);
        border-color: #d63031;
    }
    .entity-LOC {
        background: linear-gradient(135deg, #27ae60 0%, #229954 50%, #1e8449 100%);
        border-color: #00b894;
    }
    .entity-TME {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 50%, #21618c 100%);
        border-color: #74b9ff;
    }
    .entity-TITLE {
        background: linear-gradient(135deg, #f39c12 0%, #e67e22 50%, #d35400 100%);
        border-color: #fdcb6e;
    }
    .entity-NUM {
        background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 50%, #7d3c98 100%);
        border-color: #a29bfe;
    }

    .prediction-box {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        line-height: 2;
        font-size: 1.1rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #ffffff, #f8f9fa);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid #e9ecef;
        height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: transform 0.3s ease;
        margin-bottom: 1rem;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }

    .sample-text {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .legend-container {
        background: linear-gradient(135deg, #f8f9fa, #ffffff);
        padding: 1.2rem;
        border-radius: 15px;
        border: 1px solid #e9ecef;
        box-shadow: 0 3px 8px rgba(0,0,0,0.1);
    }

    .legend-item {
        display: flex;
        align-items: center;
        margin: 8px 0;
        font-size: 0.95rem;
        font-weight: 500;
    }

    .legend-label {
        margin-left: 10px;
        color: #2d3436;
    }

    .stMetric {
        background: none !important;
    }

    .stMetric > div {
        background: none !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Khởi tạo session state
if "predictor" not in st.session_state:
    st.session_state.predictor = None
if "model_type" not in st.session_state:
    st.session_state.model_type = None
if "sample_data" not in st.session_state:
    st.session_state.sample_data = None
if "user_input_text" not in st.session_state:
    st.session_state.user_input_text = ""


def clean_punctuation(tokens) -> str:
    if not isinstance(tokens, list):
        return tokens

    punctuation_marks = [".", ",", "?", "!", ":", ";"]
    new_tokens = list(tokens)

    i = len(new_tokens) - 1
    while i > 0:
        if new_tokens[i] in punctuation_marks:
            new_tokens[i - 1] += new_tokens[i]
            del new_tokens[i]
        i -= 1

    return " ".join(new_tokens)


# Load dữ liệu mẫu từ file CSV
@st.cache_data
def load_sample_data():
    import ast

    try:
        df = pd.read_csv("./output_data/vietnam-history-data-ner/ner_sentences.csv")
        df["tokens_eval"] = df["tokens"].apply(ast.literal_eval)
        df["sentence_length"] = df["tokens_eval"].apply(len)
        df["text"] = df["tokens_eval"].apply(clean_punctuation)
        df_sorted = df.sort_values(by="sentence_length", ascending=False).head(3000)
        sampled_1000 = df_sorted.sample(n=1000, random_state=42)
        sample_texts = sampled_1000["text"].to_list()
        return df, sample_texts
    except FileNotFoundError:
        raise FileNotFoundError(
            "Không tìm thấy file dữ liệu mẫu. Vui lòng kiểm tra đường dẫn!"
        )


# Load mô hình
def load_model(model_type, model_path, device="cpu"):
    try:
        if model_type == "PhoBERT-CRF":
            return TransformersNERPredictor(model_path, cuda_device=device)
        elif model_type == "BiLSTM-CRF":
            return NERPredictor(model_path, cuda_device=device)
    except Exception as e:
        st.error(f"Lỗi khi load mô hình: {e}")
        return None


# Highlight các entities
def highlight_entities(tokens: List[str], labels: List[str]) -> str:
    """Tạo HTML với highlight cho các entities"""
    html_content = []
    i = 0

    while i < len(tokens):
        token = tokens[i]
        label = labels[i]

        if label.startswith("B-"):
            # Bắt đầu entity mới
            entity_type = label[2:]
            entity_tokens = [token]
            j = i + 1

            # Thu thập tất cả tokens của entity
            while j < len(tokens) and (
                labels[j] == f"I-{entity_type}" or labels[j] == f"E-{entity_type}"
            ):
                entity_tokens.append(tokens[j])
                if labels[j] == f"E-{entity_type}":
                    j += 1
                    break
                j += 1

            entity_text = " ".join(entity_tokens)
            html_content.append(
                f'<span class="entity-box entity-{entity_type}" title="{entity_type}">{entity_text}</span>'
            )
            i = j
        else:
            # Token thường
            html_content.append(token)
            i += 1

        # Thêm khoảng trắng nếu không phải token cuối
        if i < len(tokens):
            html_content.append(" ")

    return "".join(html_content)


# Chuyển đổi prediction thành JSON
def predictions_to_json(tokens: List[str], labels: List[str]) -> Dict:
    entities = []
    entity_count = 0
    i = 0

    while i < len(tokens):
        label = labels[i]

        if label.startswith("B-"):
            entity_type = label[2:]
            entity_tokens = [tokens[i]]
            start_idx = i
            j = i + 1

            while j < len(tokens) and (
                labels[j] == f"I-{entity_type}" or labels[j] == f"E-{entity_type}"
            ):
                entity_tokens.append(tokens[j])
                if labels[j] == f"E-{entity_type}":
                    j += 1
                    break
                j += 1

            end_idx = j - 1
            entity_text = " ".join(entity_tokens)

            entities.append(
                {
                    "type": entity_type,
                    "text": entity_text,
                    "start": start_idx,
                    "end": end_idx,
                }
            )
            entity_count += 1
            i = j
        else:
            i += 1

    return {
        "text": " ".join(tokens),
        "total_entities": entity_count,
        "entities": entities,
        "entity_types": list(set([e["type"] for e in entities])),
    }


# So sánh kết quả dự đoán với ground truth
def compare_predictions(pred_entities: List, true_entities: List) -> Dict:
    pred_set = set((e["type"], e["text"], e["start"], e["end"]) for e in pred_entities)
    true_set = set((e["type"], e["text"], e["start"], e["end"]) for e in true_entities)

    tp = len(pred_set & true_set)  # True Positive
    fp = len(pred_set - true_set)  # False Positive
    fn = len(true_set - pred_set)  # False Negative

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


# Header
st.markdown(
    '<h1 class="main-header">🏛️ Named Entity Recognition</h1>', unsafe_allow_html=True
)
st.markdown(
    '<p style="text-align: center; font-size: 1.2rem; color: #666;">Chương trình nhận dạng thực thể cho văn bản Lịch Sử Việt Nam</p>',
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.header("⚙️ Cấu hình")

    # Chọn mô hình
    model_type = st.selectbox(
        "Chọn mô hình",
        ["PhoBERT-CRF", "BiLSTM-CRF"],
        help="PhoBERT-CRF: Mô hình transformer, BiLSTM-CRF: Mô hình RNN truyền thống",
    )

    # Cấu hình đường dẫn mô hình
    if model_type == "PhoBERT-CRF":
        model_path = "model_files/phobert_base"
    else:
        model_path = "model_files/bilstm/bilstm.tar.gz"

    device = st.selectbox("Thiết bị", ["cpu", "cuda:0"], help="Chọn CPU hoặc GPU")

    # Load mô hình
    if st.button("🔄 Load mô hình"):
        with st.spinner("Đang load mô hình..."):
            st.session_state.predictor = load_model(model_type, model_path, device)
            st.session_state.model_type = model_type
            if st.session_state.predictor:
                st.success("Mô hình đã được load thành công!")
            else:
                st.error("Không thể load mô hình!")

    # Thông tin mô hình
    if st.session_state.predictor:
        st.info(f"Mô hình hiện tại: {st.session_state.model_type}")

    # Legend màu sắc
    st.subheader("Thông tin thực thể")
    legend_html = """
    <div class="legend-container">
        <div class="legend-item">
            <span class="entity-box entity-PER">PER</span>
            <span class="legend-label">Người (Person)</span>
        </div>
        <div class="legend-item">
            <span class="entity-box entity-LOC">LOC</span>
            <span class="legend-label">Địa điểm (Location)</span>
        </div>
        <div class="legend-item">
            <span class="entity-box entity-TME">TME</span>
            <span class="legend-label">Thời gian (Time)</span>
        </div>
        <div class="legend-item">
            <span class="entity-box entity-TITLE">TITLE</span>
            <span class="legend-label">Chức danh (Title)</span>
        </div>
        <div class="legend-item">
            <span class="entity-box entity-NUM">NUM</span>
            <span class="legend-label">Số (Number)</span>
        </div>
    </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)

# Tabs chính
tab1, tab2, tab3 = st.tabs(
    ["Dự đoán văn bản", "Kiểm tra dữ liệu mẫu", "Thống kê và phân tích"]
)

with tab1:
    st.subheader("Nhập văn bản để dự đoán")

    if not st.session_state.predictor:
        st.warning("⚠️ Vui lòng load mô hình trước khi sử dụng!")
    else:
        # Load sample data if not loaded
        if st.session_state.sample_data is None:
            st.session_state.sample_data = load_sample_data()[0]

        # Load sample texts for selectbox
        sample_texts = ["---Nhập văn bản bên dưới---"] + load_sample_data()[1]

        # Sample text selection
        selected_sample = st.selectbox("Chọn mẫu để dự đoán", sample_texts)

        # Handle sample selection change
        if selected_sample != "---Nhập văn bản bên dưới---":
            st.session_state.user_input_text = selected_sample
        else:
            st.session_state.user_input_text = ""

        # Text input area
        user_input = st.text_area(
            "Nhập hoặc chỉnh sửa văn bản",
            value=st.session_state.user_input_text,
            height=120,
            placeholder="Nhập câu văn lịch sử Việt Nam để nhận diện thực thể...",
            key="text_input",
        )

        # Update session state when text changes
        if user_input != st.session_state.user_input_text:
            st.session_state.user_input_text = user_input

        # Button layout - only show buttons when there's text
        if user_input.strip():
            col1, col2 = st.columns([1, 9])
            with col1:
                predict_btn = st.button("Dự đoán", type="primary")
        else:
            predict_btn = False

        # Handle predict button
        if predict_btn and user_input.strip():
            with st.spinner("Đang xử lý..."):
                try:
                    # Tokenize
                    nltk.download("punkt_tab", quiet=True)

                    if st.session_state.model_type == "PhoBERT-CRF":
                        tokens = nltk.word_tokenize(user_input)
                        predictions = st.session_state.predictor.predict([tokens])
                        labels = predictions[0] if predictions else []
                    else:
                        predictions = st.session_state.predictor.predict(user_input)
                        tokens = nltk.word_tokenize(user_input)
                        labels = predictions if isinstance(predictions, list) else []

                    if tokens and labels:
                        # Hiển thị kết quả với highlight
                        st.subheader("Kết quả dự đoán")
                        highlighted_text = highlight_entities(tokens, labels)
                        st.markdown(
                            f'<div class="prediction-box">{highlighted_text}</div>',
                            unsafe_allow_html=True,
                        )

                        # JSON output
                        json_result = predictions_to_json(tokens, labels)

                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("Thông tin tổng quan")
                            st.metric("Tổng số từ", len(tokens))
                            st.metric("Số entities", json_result["total_entities"])
                            if json_result["entity_types"]:
                                st.write(
                                    "**Loại entities**",
                                    ", ".join(json_result["entity_types"]),
                                )

                        with col2:
                            st.subheader("JSON Output")
                            st.json(json_result)

                        # Bảng chi tiết entities
                        if json_result["entities"]:
                            st.subheader("Chi tiết các entities")
                            df_entities = pd.DataFrame(json_result["entities"])
                            st.dataframe(df_entities, use_container_width=True)

                except Exception as e:
                    st.error(f"Lỗi trong quá trình dự đoán: {e}")

with tab2:
    st.subheader("Kiểm tra với dữ liệu mẫu")

    if not st.session_state.predictor:
        st.warning("⚠️ Vui lòng load mô hình trước khi sử dụng!")
    else:
        # Load dữ liệu mẫu
        if st.session_state.sample_data is None:
            st.session_state.sample_data = load_sample_data()[0]

        df = st.session_state.sample_data

        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"Tổng số mẫu: {len(df)}")
        with col2:
            if st.button("🎲 Chọn ngẫu nhiên"):
                selected_idx = random.randint(0, len(df) - 1)
            else:
                selected_idx = 0

        # Tạo danh sách options cho selectbox với format "index - text"
        sample_options = []
        for idx in range(len(df)):
            sample_text = df.iloc[idx]["text"]
            # Giới hạn độ dài text hiển thị để tránh selectbox quá dài
            display_text = (
                sample_text[:100] + "..." if len(sample_text) > 100 else sample_text
            )
            sample_options.append(f"{idx} - {display_text}")

        # Chọn mẫu với format mới
        selected_option = st.selectbox(
            "Chọn mẫu:",
            sample_options,
            index=selected_idx,
            help="Chọn mẫu để kiểm tra dự đoán của mô hình",
        )

        # Lấy chỉ số từ option được chọn
        sample_idx = int(selected_option.split(" - ")[0])
        sample = df.iloc[sample_idx]

        # Parse dữ liệu
        try:
            true_tokens = eval(sample["tokens"])
            true_labels = eval(sample["bio_tags"])
            true_entities = eval(sample["entities"])

            st.markdown(
                f'<div class="sample-text"><strong>ID:</strong> {sample["stc_id"]}</div>',
                unsafe_allow_html=True,
            )

            # Hiển thị ground truth
            st.subheader("Ground Truth")
            true_highlighted = highlight_entities(true_tokens, true_labels)
            st.markdown(
                f'<div class="prediction-box">{true_highlighted}</div>',
                unsafe_allow_html=True,
            )

            # Dự đoán
            with st.spinner("Đang dự đoán..."):
                try:
                    if st.session_state.model_type == "PhoBERT-CRF":
                        pred_result = st.session_state.predictor.predict([true_tokens])
                        pred_labels = pred_result[0] if pred_result else []
                    else:
                        text_input = " ".join(true_tokens)
                        pred_result = st.session_state.predictor.predict(text_input)
                        pred_labels = (
                            pred_result if isinstance(pred_result, list) else []
                        )

                    # Hiển thị dự đoán
                    st.subheader("Dự đoán của mô hình")
                    pred_highlighted = highlight_entities(true_tokens, pred_labels)
                    st.markdown(
                        f'<div class="prediction-box">{pred_highlighted}</div>',
                        unsafe_allow_html=True,
                    )

                    # So sánh kết quả
                    pred_json = predictions_to_json(true_tokens, pred_labels)
                    comparison = compare_predictions(
                        pred_json["entities"], true_entities
                    )

                    st.subheader("So sánh kết quả")
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Precision", f"{comparison['precision']:.3f}")
                    with col2:
                        st.metric("Recall", f"{comparison['recall']:.3f}")
                    with col3:
                        st.metric("F1-Score", f"{comparison['f1']:.3f}")
                    with col4:
                        st.metric(
                            "Accuracy",
                            f"{comparison['tp']}/{comparison['tp'] + comparison['fp'] + comparison['fn']}",
                        )

                    # Bảng so sánh chi tiết
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**Ground Truth Entities**")
                        if true_entities:
                            df_true = pd.DataFrame(true_entities)
                            st.dataframe(df_true)
                        else:
                            st.write("Không có entities")

                    with col2:
                        st.write("**Predicted Entities**")
                        if pred_json["entities"]:
                            df_pred = pd.DataFrame(pred_json["entities"])
                            st.dataframe(df_pred[["type", "text", "start", "end"]])
                        else:
                            st.write("Không có entities")

                except Exception as e:
                    st.error(f"Lỗi trong quá trình dự đoán: {e}")

        except Exception as e:
            st.error(f"Lỗi khi parse dữ liệu: {e}")

with tab3:
    st.subheader("Thống kê và phân tích")

    if st.session_state.sample_data is not None:
        df = st.session_state.sample_data

        # Thống kê tổng quan với layout cải thiện
        st.write("#### Tổng quan dữ liệu")

        # Parse và thống kê entities
        all_entities = []
        entity_types = []

        for _, row in df.iterrows():
            entities = eval(row["entities"])
            all_entities.extend(entities)
            entity_types.extend([e["type"] for e in entities])

        # Sử dụng columns với metric cards tùy chỉnh
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(
                f"""
            <div class="metric-card">
                <h3 style="color: #1f77b4; margin: 0; font-size: 2rem;">{len(df)}</h3>
                <p style="margin: 0; color: #666; font-weight: bold;">Tổng mẫu</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
            <div class="metric-card">
                <h3 style="color: #e74c3c; margin: 0; font-size: 2rem;">{len(all_entities)}</h3>
                <p style="margin: 0; color: #666; font-weight: bold;">Tổng entities</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                f"""
            <div class="metric-card">
                <h3 style="color: #27ae60; margin: 0; font-size: 2rem;">{len(set(entity_types))}</h3>
                <p style="margin: 0; color: #666; font-weight: bold;">Loại entities</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col4:
            avg_entities = len(all_entities) / len(df) if len(df) > 0 else 0
            st.markdown(
                f"""
            <div class="metric-card">
                <h3 style="color: #9b59b6; margin: 0; font-size: 2rem;">{avg_entities:.1f}</h3>
                <p style="margin: 0; color: #666; font-weight: bold;">Trung bình số entities/câu</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Biểu đồ phân bố loại entities với màu mới
        if entity_types:
            entity_counts = Counter(entity_types)

            st.write("#### Phân bố các loại entities")
            col1, col2 = st.columns(2)

            with col1:
                st.write("##### Tỷ lệ các loại entities")
                fig_pie = px.pie(
                    values=list(entity_counts.values()),
                    names=list(entity_counts.keys()),
                    title="",
                    color_discrete_sequence=[
                        "#e74c3c",
                        "#27ae60",
                        "#3498db",
                        "#f39c12",
                        "#9b59b6",
                    ],
                )
                fig_pie.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                st.write("##### Số lượng theo loại")
                fig_bar = px.bar(
                    x=list(entity_counts.keys()),
                    y=list(entity_counts.values()),
                    title="",
                    color=list(entity_counts.keys()),
                    color_discrete_sequence=[
                        "#e74c3c",
                        "#27ae60",
                        "#3498db",
                        "#f39c12",
                        "#9b59b6",
                    ],
                )
                fig_bar.update_xaxes(title="Loại entity")
                fig_bar.update_yaxes(title="Số lượng")
                fig_bar.update_layout(showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)

        # Bảng thống kê chi tiết
        st.write("#### Thống kê chi tiết")
        if entity_types:
            stats_data = []
            for entity_type in set(entity_types):
                count = entity_counts[entity_type]
                percentage = (count / len(all_entities)) * 100
                stats_data.append(
                    {
                        "Loại Entity": entity_type,
                        "Số lượng": count,
                        "Tỷ lệ (%)": f"{percentage:.1f}%",
                    }
                )

            df_stats = pd.DataFrame(stats_data)
            df_stats = df_stats.sort_values("Số lượng", ascending=False)
            st.dataframe(df_stats, use_container_width=True, hide_index=True)
    else:
        st.info("Vui lòng load dữ liệu mẫu để xem thống kê")
