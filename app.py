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
    "NEU": "Absolute neutrophil count",
    "MONO": "Absolute monocyte count",
    "BASO": "Absolute basophil count",
    "Cl": "Serum chloride",
    "CK_MB": "Creatine kinase-MB",
    "ALP": "Alkaline phosphatase",
    "GLB": "Globulin",
    "ChE": "Cholinesterase",
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
        raise FileNotFoundError("rf_model.json was not found.")
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
        raise ValueError(f"Missing model fields: {', '.join(sorted(missing))}")
    if model["feature_names"] != FEATURES:
        raise ValueError("The model feature order does not match the application.")
    if len(model["trees"]) != model["n_estimators"]:
        raise ValueError("The number of trees does not match the model metadata.")
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
        help=f"{FEATURE_DESCRIPTIONS[feature]}; unit: {FEATURE_UNITS[feature]}",
        key=f"input_{feature}",
    )


def main():
    st.set_page_config(
        page_title="H. influenzae Resistance Risk Calculator",
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
            padding-left: 1.5rem;
            padding-right: 1.5rem;
        }
        h1, h2, h3, p, label, button {
            letter-spacing: 0 !important;
        }
        h1 {
            font-size: 1.75rem !important;
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

    st.title("Haemophilus influenzae Antimicrobial Resistance Risk Calculator")
    st.caption(
        "Random forest model | Endpoint: amoxicillin/clavulanate nonsusceptibility"
    )

    try:
        model = load_model()
    except Exception as exc:
        st.error(f"Model loading failed: {exc}")
        st.stop()

    defaults = model["fill_values"]
    with st.form("prediction_form"):
        st.subheader("Laboratory Parameters")
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
            "Calculate Risk", type="primary", use_container_width=True
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
            st.subheader("Model Output")
            result_col, threshold_col = st.columns(2)
            result_col.metric("Predicted Nonsusceptibility", f"{probability:.1%}")
            threshold_col.metric("Youden Cutoff", f"{decision_threshold:.0%}")
            st.progress(min(max(probability, 0.0), 1.0))
            if threshold_reached:
                st.error(
                    f"High predicted risk: The predicted probability is "
                    f"{probability:.1%}, at or above the {decision_threshold:.0%} "
                    "cutoff. The isolate may be resistant or intermediate to "
                    "amoxicillin/clavulanate."
                )
            else:
                st.success(
                    f"Lower predicted risk: The predicted probability is "
                    f"{probability:.1%}, below the {decision_threshold:.0%} cutoff. "
                    "The isolate is less likely to be resistant or intermediate to "
                    "amoxicillin/clavulanate."
                )
            st.caption(
                "The 0.69 cutoff was selected from the training data by maximizing "
                "Youden's J. The predicted probability does not replace standardized "
                "antimicrobial susceptibility testing."
            )

    st.divider()
    st.caption(
        "For research demonstration only. Antimicrobial selection should integrate "
        "standardized susceptibility testing, patient characteristics, and clinical "
        "judgment."
    )


if __name__ == "__main__":
    main()
