import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_image_comparison import image_comparison
from streamlit_option_menu import option_menu
from deep_inference import get_deep_change_heatmap
from model_comparison import compare_models
from risk_loader import load_risk_table
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from advanced_analytics import get_feature_importance, get_leaderboard, extract_regions  # noqa: E402
from inference import (  # noqa: E402
    compute_change_map,
    compute_risk_score,
    find_city_pair,
    get_city_label,
    get_city_row,
    list_cities,
    load_image,
    predict_city,
)

st.set_page_config(
    page_title="SpaceOps Sentinel",
    page_icon="🛰️",
    layout="wide",
)

st.markdown(
    """
    <link rel="stylesheet"
    href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1rem;
            max-width: 1450px;
        }
        .stMetric {
            background: rgba(255,255,255,0.02);
            padding: 10px;
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.05);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="
        text-align:center;
        margin-top: 0.4rem;
        margin-bottom: 0.2rem;
    ">
        <div style="
            font-size: 44px;
            font-weight: 900;
            line-height: 1.1;
            letter-spacing: 0.4px;
        ">
            <i class="fa-solid fa-satellite-dish" style="color:#ff4b4b; margin-right:10px;"></i>
            SpaceOps Sentinel
        </div>
        <div style="
            font-size: 18px;
            font-weight: 700;
            color: #e5e7eb;
            margin-top: 8px;
        ">
            Geo-Intelligence & Satellite Change Analytics Platform
        </div>
        <div style="
            font-size: 13px;
            color: #9aa0a6;
            margin-top: 8px;
            max-width: 1050px;
            margin-left: auto;
            margin-right: auto;
            line-height: 1.5;
        ">
            Compare multi-temporal satellite scenes, quantify spatial change, detect anomaly regions,
            and interpret risk with classical ML analytics.
        </div>
    </div>
    <hr style="opacity:0.25;">
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        """
        <div style="
            text-align:center;
            font-size:48px;
            font-weight:900;
            letter-spacing:1px;
            margin-top:10px;
            margin-bottom:14px;
        ">
            <i class="fa-solid fa-satellite-dish"
               style="color:#ff4b4b; margin-right:10px;"></i>
            SpaceOps
        </div>

        <div style="
            text-align:center;
            font-size:14px;
            color:#9aa0a6;
            margin-bottom:14px;
        ">
            Geo Intelligence Console
        </div>

        <hr style="opacity:0.25;">
        """,
        unsafe_allow_html=True,
    )

    selected = option_menu(
    "Navigation",
    [
        "Overview",
        "Change Analytics",
        "Model Insights",
        "Global Intelligence",
        "Live Detection"
    ],
    icons=[
        "activity",
        "bar-chart",
        "cpu",
        "globe",
        "camera"
    ],
    default_index=0,
        styles={
            "container": {
                "padding": "8px",
                "background-color": "#0e1117",
                "border-radius": "18px",
            },
            "icon": {"color": "#ff4b4b", "font-size": "20px"},
            "nav-link": {
                "font-size": "17px",
                "margin": "8px",
                "--hover-color": "#262730",
                "border-radius": "12px",
            },
            "nav-link-selected": {
                "background-color": "#ff4b4b",
                "color": "white",
            },
        },
    )

cities = list_cities()
if not cities:
    st.error("No cities found in feature table.")
    st.stop()

city = st.selectbox("Select City", cities)

img1_path, img2_path = find_city_pair(city)
if img1_path is None or img2_path is None:
    st.error("Satellite pair not found for this city.")
    st.stop()

img1 = load_image(img1_path)
img2 = load_image(img2_path)
diff_norm, binary_mask, threshold = compute_change_map(img1, img2)

predicted_ratio = predict_city(city)
actual_row = get_city_label(city)
city_row = get_city_row(city)
risk_score = compute_risk_score(city)


def centered_heatmap():
    st.subheader("Change Heatmap")
    colA, colB, colC = st.columns([1, 2, 1])
    with colB:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(diff_norm, cmap="inferno")
        ax.axis("off")
        ax.set_title("Change Intensity")
        st.pyplot(fig)



def centered_mask():
    st.subheader("Binary Mask")
    colA, colB, colC = st.columns([1, 2, 1])
    with colB:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(binary_mask, cmap="gray")
        ax.axis("off")
        st.pyplot(fig)



def region_visual():
    regions = extract_regions(binary_mask, min_area=250)

    st.write(f"Detected Regions: **{len(regions)}**")

    if not regions:
        st.info("No large connected change regions found for this city.")
        return

    colA, colB, colC = st.columns([1, 2, 1])
    with colB:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img1)

        for r in regions[:12]:
            minr, minc, maxr, maxc = r["bbox"]
            rect = patches.Rectangle(
                (minc, minr),
                maxc - minc,
                maxr - minr,
                linewidth=2,
                edgecolor="cyan",
                facecolor="none",
            )
            ax.add_patch(rect)

        ax.set_title("Detected Change Regions")
        ax.axis("off")
        st.pyplot(fig)

    st.dataframe(pd.DataFrame(regions), use_container_width=True)


if selected == "Overview":
    st.title("🌍 Satellite Temporal Change Explorer")
    st.caption(
        "This page compares two satellite scenes from the same city, visualizes the change map, "
        "and summarizes severity using classical ML."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Predicted Change", round(predicted_ratio, 4))

    if actual_row:
        c2.metric("Actual Change", round(actual_row["gt_change_ratio"], 4))
    else:
        c2.metric("Actual Change", "N/A")

    severity = "LOW"
    if predicted_ratio > 0.02:
        severity = "HIGH"
    elif predicted_ratio > 0.01:
        severity = "MEDIUM"

    c3.metric("Severity", severity)
    c4.metric("Risk Score", risk_score)

    st.subheader("Temporal Comparison")
    compA, compB, compC = st.columns([1, 2, 1])
    with compB:
        image_comparison(
            img1=img1,
            img2=img2,
            label1="Time-1",
            label2="Time-2",
        )

    centered_heatmap()
    st.markdown("---")
    st.markdown("### Deep Learning Spatial Change Detection")

    deep_heat = get_deep_change_heatmap(img1, img2)

    colA, colB, colC = st.columns([1,2,1])

    with colB:
        fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(deep_heat, cmap="turbo")
        ax.set_title("Deep Model Spatial Change")
        ax.axis("off")
        st.pyplot(fig)

    deep_ratio = np.mean(deep_heat > 0.6)

    st.metric(
        "Deep Change Ratio",
        f"{deep_ratio:.4f}"
    )

    st.markdown("---")
    st.markdown("### 🤖 Model Intelligence Comparison")

    cmp = compare_models(predicted_ratio, deep_heat)

    mc1, mc2, mc3 = st.columns(3)

    mc1.metric("Classical Ratio", round(predicted_ratio,4))
    mc2.metric("Deep Ratio", round(cmp["deep_ratio"],4))
    mc3.metric("Model Agreement", round(cmp["agreement"],3))

    st.info(f"🧠 Verdict: {cmp['verdict']}")

elif selected == "Change Analytics":
    st.title("📊 Change Feature Analytics")
    st.caption(
        "This page shows the engineered features used by the classical ML model and the connected anomaly regions."
    )

    vals = {
        "Diff Mean": city_row["diff_mean"],
        "Diff Std": city_row["diff_std"],
        "Change Ratio": city_row["change_ratio"],
        "Img1 Std": city_row["img1_std"],
        "Img2 Std": city_row["img2_std"],
    }

    fig = px.bar(
        x=list(vals.keys()),
        y=list(vals.values()),
        title="Feature Profile",
        labels={"x": "Feature", "y": "Value"},
    )
    st.plotly_chart(fig, use_container_width=True)

    centered_mask()

    st.subheader("Region Detection")
    region_visual()

elif selected == "Model Insights":
    st.title("🤖 Model Intelligence")
    st.caption(
        "This page explains the model inputs and shows which engineered features matter most for prediction."
    )

    st.json(city_row)

    imp = get_feature_importance()
    if not imp.empty:
        fig = px.bar(
            imp,
            x="importance",
            y="feature",
            orientation="h",
            title="Model Feature Importance",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance is not available for the current model.")

    st.subheader("Decision Context")
    st.info(
        f"""
        Threshold used: {round(threshold, 4)}

        Predicted change severity: {round(predicted_ratio, 4)}

        Risk score: {risk_score}
        """
    )

elif selected == "Global Intelligence":

    st.title("🌐 Global Risk Intelligence")
    st.caption(
        "This page ranks cities by risk and helps you inspect where the largest temporal changes are concentrated."
    )

    top, bottom = get_leaderboard(5)

    st.subheader("Highest Risk")
    st.dataframe(top, use_container_width=True)

    st.subheader("Lowest Risk")
    st.dataframe(bottom, use_container_width=True)

    fig = px.bar(
        top,
        x="city",
        y="risk_score",
        color="risk_score",
        title="Top Risk Cities",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detected Region Preview")
    region_visual()

    # ⭐ IMPORTANT → define risk_df here
    risk_df = load_risk_table()

    st.markdown("### 🎯 City Risk Radar")

    import plotly.graph_objects as go

    top5 = risk_df.head(5)

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=top5["risk_score"],
        theta=top5["city"],
        fill='toself',
        name='Risk Pattern'
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=False,
        height=450
    )

    st.plotly_chart(fig, use_container_width=True)

    # ⭐ FINAL ALERT METRIC
    st.markdown("### 🚨 Global Change Alert Summary")

    high_alert_count = len(risk_df[risk_df["risk_score"] > 2])

    st.metric(
        "Cities With Significant Change",
        high_alert_count
    )

elif selected == "Live Detection":

    st.title("📡 Real-Time Change Detection")
    st.caption(
        "Upload two satellite images to analyze temporal change using both classical and deep models."
    )

    col1, col2 = st.columns(2)

    with col1:
        file1 = st.file_uploader(
            "Upload Time-1 Image",
            type=["png", "jpg", "jpeg"]
        )

    with col2:
        file2 = st.file_uploader(
            "Upload Time-2 Image",
            type=["png", "jpg", "jpeg"]
        )

    if file1 and file2:

        img1_up = np.array(Image.open(file1).convert("RGB"))
        img2_up = np.array(Image.open(file2).convert("RGB"))

        st.subheader("Temporal Comparison")

        image_comparison(
            img1=img1_up,
            img2=img2_up,
            label1="Time-1",
            label2="Time-2"
        )

        # classical change
        classical_heat, binary_mask, classical_ratio = compute_change_map(img1, img2)

        st.subheader("Classical Change Map")

        fig, ax = plt.subplots(figsize=(5,5))
        ax.imshow(classical_heat, cmap="inferno")
        ax.axis("off")
        st.pyplot(fig)

        classical_ratio = np.mean(classical_heat > classical_heat.mean())

        # deep change
        deep_heat = get_deep_change_heatmap(img1_up, img2_up)

        st.subheader("Deep Learning Change Map")

        fig2, ax2 = plt.subplots(figsize=(5,5))
        ax2.imshow(deep_heat, cmap="turbo")
        ax2.axis("off")
        st.pyplot(fig2)

        deep_ratio = np.mean(deep_heat > 0.6)

        st.metric("Classical Change Ratio", f"{classical_ratio:.4f}")
        st.metric("Deep Change Ratio", f"{deep_ratio:.4f}")