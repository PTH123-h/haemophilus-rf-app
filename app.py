# -*- coding: utf-8 -*-
import json
import math
import struct
from pathlib import Path

import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "rf_model.json"

FEATURES = ["NEU", "MONO", "BASO", "Cl", "CK_MB", "ALP", "GLB", "ChE"]
FEATURE_LABELS = {
    "NEU": "NEU",
    "MONO": "MONO",
    "BASO": "BASO",
    "Cl": "Cl",
    "CK_MB": "CK-MB",
    "ALP": "ALP",
    "GLB": "GLB",
    "ChE": "ChE",
}
FEATURE_DESCRIPTIONS = {
    "NEU": "中性粒细胞绝对计数",
    "MONO": "单核细胞绝对计数",
    "BASO": "嗜碱性粒细胞绝对计数",
    "Cl": "氯离子",
    "CK_MB": "肌酸激酶同工酶",
    "ALP": "碱性磷酸酶",
    "GLB": "球蛋白",
    "ChE": "胆碱酯酶",
}
FEATURE_UNITS = {
    "NEU": "10^9/L",
    "MONO": "10^9/L",
    "BASO": "10^9/L",
    "Cl": "mmol/L",
    "CK_MB": "U/L",
    "ALP": "U/L",
    "GLB": "g/L",
    "ChE": "U/L",
}


def load_model_file(path):
    if not path.exists():
        raise FileNotFoundError("未找到 rf_model.json。")
    with path.open("r", encoding="utf-8") as model_file:
        model = json.load(model_file)
    required = {
        "model_type",
        "classes",
        "feature_names",
        "n_estimators",
        "fill_values",
        "decision_threshold",
        "trees",
    }
    missing = required.difference(model)
    if missing:
        raise ValueError(f"模型文件缺少字段：{', '.join(sorted(missing))}")
    if model["feature_names"] != FEATURES:
        raise ValueError("模型变量顺序与网页变量不一致。")
    if len(model["trees"]) != model["n_estimators"]:
        raise ValueError("随机森林树数量与模型元数据不一致。")
    return model


@st.cache_data(show_spinner=False)
def load_model():
    return load_model_file(MODEL_PATH)


def predict_tree_probability(tree, values, feature_names, positive_class_index):
    node = 0
    children_left = tree["children_left"]
    children_right = tree["children_right"]
    feature = tree["feature"]
    threshold = tree["threshold"]
    node_values = tree["value"]

    while children_left[node] != -1:
        feature_name = feature_names[feature[node]]
        if values[feature_name] <= threshold[node]:
            node = children_left[node]
        else:
            node = children_right[node]

    class_values = node_values[node]
    total = sum(class_values)
    if total <= 0:
        return 0.0
    return class_values[positive_class_index] / total


def predict_probability(model, values):
    clean_values = {}
    for feature in model["feature_names"]:
        default = float(model["fill_values"][feature])
        try:
            value = float(values.get(feature, default))
        except (TypeError, ValueError):
            value = default
        clean_values[feature] = value if math.isfinite(value) else default

    # scikit-learn converts tree inputs to float32 before threshold comparisons.
    # Matching that conversion is required for exact predictions at split boundaries.
    model_values = {
        feature: struct.unpack("f", struct.pack("f", value))[0]
        for feature, value in clean_values.items()
    }

    positive_class_index = model["classes"].index(1)
    probabilities = [
        predict_tree_probability(
            tree,
            model_values,
            model["feature_names"],
            positive_class_index,
        )
        for tree in model["trees"]
    ]
    return sum(probabilities) / len(probabilities), clean_values


def number_input(feature, default, step, value_format):
    return st.number_input(
        f"{FEATURE_LABELS[feature]} ({FEATURE_UNITS[feature]})",
        min_value=0.0,
        value=float(default),
        step=step,
        format=value_format,
        help=f"{FEATURE_DESCRIPTIONS[feature]}；单位：{FEATURE_UNITS[feature]}",
        key=f"input_{feature}",
    )


def main():
    st.set_page_config(
        page_title="流感嗜血杆菌耐药风险计算器",
        page_icon="H",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    st.markdown(
        """
        <style>
        .block-container {
            max-width: 980px;
            padding-top: 1rem;
            padding-bottom: 1.5rem;
        }
        h1, h2, h3, p, label, button {
            letter-spacing: 0 !important;
        }
        h1 {
            font-size: 2rem !important;
            line-height: 1.25 !important;
        }
        h2, h3 {
            font-size: 1.15rem !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.65rem;
        }
        [data-testid="stForm"] {
            padding: 0.7rem 0.9rem 0.65rem;
        }
        [data-testid="stForm"] [data-testid="stVerticalBlock"] {
            gap: 0.35rem;
        }
        [data-testid="stForm"] [data-testid="stHorizontalBlock"] {
            gap: 0.75rem;
        }
        [data-testid="stForm"] [data-testid="stNumberInput"] label p {
            font-size: 0.82rem;
            line-height: 1.1;
        }
        [data-testid="stFormSubmitButton"] button {
            min-height: 2.35rem;
            height: 2.35rem;
            border-radius: 6px;
            font-weight: 650;
        }
        [data-testid="stForm"] [data-testid="stNumberInput"] input,
        [data-testid="stForm"] [data-testid="stNumberInput"] button {
            min-height: 2.05rem;
            height: 2.05rem;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 6px;
        }
        @media (max-width: 640px) {
            .block-container {
                padding-top: 0.75rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }
            h1 {
                font-size: 1.55rem !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("流感嗜血杆菌耐药风险计算器")
    st.caption("随机森林模型 | 结局：阿莫西林/克拉维酸不敏感")

    try:
        model = load_model()
    except Exception as exc:
        st.error(f"模型加载失败：{exc}")
        st.stop()

    defaults = model["fill_values"]
    with st.form("prediction_form"):
        st.subheader("实验室指标")
        left, right = st.columns(2)
        with left:
            neu = number_input("NEU", defaults["NEU"], 0.10, "%.2f")
            mono = number_input("MONO", defaults["MONO"], 0.01, "%.2f")
            baso = number_input("BASO", defaults["BASO"], 0.01, "%.2f")
            chloride = number_input("Cl", defaults["Cl"], 0.10, "%.1f")
        with right:
            ck_mb = number_input("CK_MB", defaults["CK_MB"], 1.0, "%.1f")
            alp = number_input("ALP", defaults["ALP"], 1.0, "%.1f")
            glb = number_input("GLB", defaults["GLB"], 0.10, "%.1f")
            che = number_input("ChE", defaults["ChE"], 10.0, "%.1f")
        submitted = st.form_submit_button(
            "计算风险", type="primary", use_container_width=True
        )

    if submitted:
        values = {
            "NEU": neu,
            "MONO": mono,
            "BASO": baso,
            "Cl": chloride,
            "CK_MB": ck_mb,
            "ALP": alp,
            "GLB": glb,
            "ChE": che,
        }
        probability, _ = predict_probability(model, values)
        decision_threshold = float(model["decision_threshold"])
        threshold_reached = probability >= decision_threshold

        with st.container(border=True):
            st.subheader("模型结果")
            result_col, threshold_col = st.columns(2)
            result_col.metric("预测不敏感概率", f"{probability:.1%}")
            threshold_col.metric("固定判定阈值", f"{decision_threshold:.0%}")
            st.progress(min(max(probability, 0.0), 1.0))
            if threshold_reached:
                st.error("模型判定：达到阿莫西林/克拉维酸不敏感阈值。")
            else:
                st.success("模型判定：未达到阿莫西林/克拉维酸不敏感阈值。")
            st.caption("该概率是模型输出，不等同于微生物药敏试验结果。")

    st.divider()
    st.caption(
        "本工具用于科研演示。任何抗菌药物选择均应结合标准药敏试验、患者情况及临床判断。"
    )


if __name__ == "__main__":
    main()
