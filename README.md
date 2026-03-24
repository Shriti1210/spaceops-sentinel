# 🛰️ SpaceOps Sentinel

**Geo-Intelligence & Satellite Change Analytics Platform**

SpaceOps Sentinel is an end-to-end satellite image intelligence dashboard that compares two images from different times, detects spatial change, estimates risk, and visualizes the result in an interactive Streamlit app.

It is built for remote sensing style analysis and demonstrates:
- data handling
- feature engineering
- classical machine learning
- deep learning change detection
- visual analytics
- deployment-ready dashboard design

---

## ✨ What the project does

This application takes two satellite images of the same location from different dates and analyzes how much the scene has changed.

It provides:
- a side-by-side temporal comparison slider
- classical change heatmaps
- deep learning based spatial change maps
- predicted change ratio
- actual change ratio from labels
- anomaly risk score
- region detection
- city-wise risk ranking
- global intelligence analytics
- real-time upload mode for custom image pairs

The goal is to help users quickly understand urban expansion, land change, and anomaly-like temporal changes in satellite imagery.

---

## 🚀 Key Features

- 🛰️ **Temporal Image Comparison**
  - Compare two satellite scenes from the same city.

- 🔥 **Classical Change Heatmap**
  - Uses pixel-wise image difference to show spatial change intensity.

- 🧠 **Deep Learning Change Map**
  - Uses a PyTorch CNN-based spatial model for change detection.

- 📊 **Risk Score & Severity**
  - Predicts change severity and assigns a risk score.

- 🌍 **Global Intelligence Panel**
  - Shows city-wise risk ranking and summary analytics.

- 🎯 **Region Detection**
  - Highlights connected change regions on the image.

- 📡 **Live Detection Mode**
  - Upload two images and analyze them directly in the app.

- 🎨 **Professional UI**
  - Dark theme, clean layout, sidebar navigation, and image comparison slider.

---

## 🧠 How it works

The project is built in two layers:

### 1) Classical ML Layer
The app first uses handcrafted features like:
- mean intensity
- standard deviation
- difference statistics
- change ratio

These features are passed into a trained classical ML model to estimate change severity.

### 2) Deep Learning Layer
A PyTorch-based CNN analyzes paired images to generate a spatial change map.  
This adds a deep-learning view of the same satellite scene, making the project more advanced and realistic.

---

## 🧪 Demo Mode vs Full Dataset Mode

### ✅ Demo Mode
The deployed Streamlit app uses a small demo dataset with a few cities:
- `abudhabi`
- `beihai`
- `mumbai`

This keeps the cloud deployment lightweight and stable.

### ✅ Full Dataset Mode
For local development, the project can use the full OSCD dataset.

If you want to run the full version locally:
1. Download the OSCD dataset
2. Place it in the correct raw data folder
3. Run the preprocessing scripts to generate feature and label tables

The repo is intentionally kept lightweight on GitHub so it can deploy faster and more reliably.

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Plotly**
- **Scikit-learn**
- **PyTorch**
- **OpenCV**
- **Pillow**
- **Scikit-image**
- **Joblib**

---

##🌐 Deployment

The project is deployed on Streamlit Cloud.

The public app link is the .streamlit.app URL shown in the browser after deployment.
