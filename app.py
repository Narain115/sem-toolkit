import streamlit as st
import pandas as pd
from PIL import Image
import os

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Semiconductor Process Intelligence Toolkit",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Sem Process Toolkit")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Select Module",
    [
        "🏠 Overview",
        "🔬 Fault Detection (SECOM)",
        "💡 Optical Metrology",
        "🗺️ Wafer Defect Classification",
        "🧪 Process DOE Optimization"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Built by:** Narain Karthikeyan")
st.sidebar.markdown("**Stack:** Python · scikit-learn · PyTorch · TMM · Streamlit")
st.sidebar.markdown("**Data:** SECOM (UCI) · WM-811K (TSMC/MIR Lab)")

# ── HELPER ────────────────────────────────────────────────────────────────────
def load_image(path):
    if os.path.exists(path):
        return Image.open(path)
    return None

def show_image(path, caption=""):
    img = load_image(path)
    if img:
        st.image(img, caption=caption, width=900)
    else:
        st.warning(f"Image not found: {path}")

# ── PAGE: OVERVIEW ────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.title("Semiconductor Process Intelligence Toolkit")
    st.markdown("""
    A four-module analytical platform covering the core data problems in 
    semiconductor manufacturing: yield prediction, optical metrology, 
    defect classification, and process optimization.
    """)

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("SECOM Sensors Analyzed", "442")
        st.metric("Fault Detection Accuracy", "98%")
        st.caption("Module 1: SECOM fault detection")

    with col2:
        st.metric("Film Stacks Modeled", "4")
        st.metric("Ellipsometry Error", "4 nm")
        st.caption("Module 2: Optical thin film")

    with col3:
        st.metric("Wafer Maps Trained", "2,249")
        st.metric("Defect Classification", "81%")
        st.caption("Module 3: WM-811K classifier")

    with col4:
        st.metric("DOE Runs", "20")
        st.metric("Optimal Desirability", "0.702")
        st.caption("Module 4: PECVD DOE")

    st.markdown("---")
    st.subheader("Data Sources")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **SECOM Semiconductor Manufacturing Dataset**  
        Source: UCI Machine Learning Repository  
        1,567 wafers · 591 sensor readings · Real fab process data  
        [Download Dataset](https://archive.ics.uci.edu/dataset/179/secom)
        """)

    with col2:
        st.markdown("""
        **WM-811K Wafer Map Dataset**  
        Source: MIR Lab (collected from TSMC production)  
        811,457 wafer maps · 8 defect pattern types · Real fab data  
        [Download Dataset](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)
        """)

    st.markdown("---")
    st.subheader("What this toolkit demonstrates")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **Semiconductor manufacturing analytics**
        - Yield loss root cause identification via SHAP
        - Imbalanced fault detection (14:1 class ratio)
        - Wafer-level defect pattern recognition
        - Spatial localization with Grad-CAM

        **Process engineering**
        - PECVD process DOE (2⁴ full factorial)
        - Multi-response optimization
        - Process window mapping
        - Pareto tradeoff analysis
        """)
    with col_b:
        st.markdown("""
        **Optical metrology**
        - Transfer Matrix Method (TMM) simulation
        - Spectroscopic ellipsometry fitting
        - Thin film thickness uniformity maps
        - Real semiconductor material optical constants

        **Tools and methods**
        - SPC · DOE · FMEA thinking
        - SMOTE for imbalanced manufacturing data
        - CNN + Grad-CAM for explainable AI
        - Physics-informed process modeling
        """)

# ── PAGE: MODULE 1 ────────────────────────────────────────────────────────────
elif page == "🔬 Fault Detection (SECOM)":
    st.title("Module 1: Wafer Fault Detection")
    st.markdown("""
    **Dataset:** SECOM semiconductor manufacturing  
    **Source:** [UCI ML Repository](https://archive.ics.uci.edu/dataset/179/secom) — McCann & Johnston (2008)  
    **Problem:** 1,567 wafers, 591 sensor readings each, 14:1 pass/fail imbalance  
    **Approach:** Feature selection → SMOTE balancing → Random Forest → SHAP explainability
    """)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("Sensors after cleaning", "442", "-149 zero-variance removed")
    col2.metric("Accuracy after SMOTE", "98%", "vs 93% naive baseline")
    col3.metric("Class balance", "1:1", "from 14:1 raw imbalance")

    st.markdown("---")
    st.subheader("Confusion Matrix")
    st.markdown("""
    Without SMOTE, a naive model predicts pass every time and still gets 93% 
    accuracy while missing every real failure. SMOTE creates synthetic failure 
    samples so the model genuinely learns to detect faults.
    """)
    show_image("outputs/module1_confusion_matrix.png")

    st.subheader("Top 15 Sensors Driving Wafer Failures")
    st.markdown("""
    SHAP (SHapley Additive exPlanations) quantifies each sensor's contribution 
    to failure predictions. This tells process engineers exactly which 
    measurements to investigate during a yield excursion, replacing manual 
    correlation analysis of 442 signals.
    """)
    show_image("outputs/module1_shap_importance.png")

    with st.expander("Engineering interpretation"):
        st.markdown("""
        In a real fab environment, this analysis would:
        - Cut root cause investigation time by 40-60% vs manual correlation
        - Direct metrology resources to the highest-signal sensors
        - Provide statistically defensible evidence for process changes
        - Feed directly into FMEA and 8D corrective action reports
        """)

# ── PAGE: MODULE 2 ────────────────────────────────────────────────────────────
elif page == "💡 Optical Metrology":
    st.title("Module 2: Optical Thin Film Metrology")
    st.markdown("""
    **Method:** Transfer Matrix Method (TMM) for multilayer thin film simulation  
    **Application:** Spectroscopic ellipsometry fitting, thickness uniformity mapping  
    **Relevance:** In-line non-destructive metrology used by KLA, AMAT, ASM equipment
    """)
    st.info("Optical constants (n, k) from Palik Handbook of Optical Constants of Solids. TMM physics based on Hecht, Optics (5th ed).")

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Film stacks modeled", "4", "SiO2, Si3N4, TiN, Al2O3")
    col2.metric("Ellipsometry fit error", "4 nm", "at 142 nm true thickness")
    col3.metric("Wavelength range", "300-800 nm", "UV to near-IR")

    st.markdown("---")
    st.subheader("Reflectance Spectra — Four Semiconductor Materials")
    st.markdown("""
    Each material has a unique optical fingerprint. TiN (barrier layer) absorbs 
    strongly across all wavelengths. SiO2 (gate oxide) and Al2O3 (ALD high-k) 
    are nearly transparent. These spectra are used in production to verify 
    film identity after deposition.
    """)
    show_image("outputs/module2_reflectance_spectra.png")

    st.subheader("Wafer Thickness Uniformity Map")
    st.markdown("""
    Simulates in-line metrology output for a 300mm wafer after CVD deposition. 
    The edge-thinning pattern is characteristic of diffusion-limited deposition. 
    Non-uniformity above 2% typically triggers a process review.
    """)
    show_image("outputs/module2_uniformity_map.png")

    st.subheader("Spectroscopic Ellipsometry Fit")
    st.markdown("""
    Ellipsometry measures Ψ (amplitude ratio) and Δ (phase difference) of 
    polarized light reflected from the film. Fitting a TMM model to these 
    spectra extracts thickness with sub-nm precision in production hardware.
    """)
    show_image("outputs/module2_ellipsometry_fit.png")

    with st.expander("Engineering interpretation"):
        st.markdown("""
        This module demonstrates:
        - Understanding of optical constants (n, k) and their physical meaning
        - Transfer matrix physics for multilayer film stacks
        - Model-based fitting methodology used in production ellipsometers
        - Uniformity analysis that directly feeds SPC control charts
        - Relevance to KLA, AMAT, ASM metrology tool development
        """)

# ── PAGE: MODULE 3 ────────────────────────────────────────────────────────────
elif page == "🗺️ Wafer Defect Classification":
    st.title("Module 3: Wafer Map Defect Classification")
    st.markdown("""
    **Dataset:** WM-811K wafer map dataset  
    **Source:** [Kaggle / MIR Lab](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map) — Wu et al. (2014)  
    **Size:** 811,457 wafer maps from real TSMC fabrication lines  
    **Approach:** CNN with Grad-CAM spatial explainability
    """)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Training samples", "2,249", "balanced across 8 classes")
    col2.metric("Test accuracy", "81%", "8-class classification")
    col3.metric("Best class", "Near-Full: 100%", "precision")

    st.markdown("---")
    st.subheader("Defect Pattern Types")
    st.markdown("""
    Each pattern type points to a different root cause in the fab process.
    Edge-Ring suggests deposition edge effects. Scratch indicates handling 
    damage. Center points to plasma or contamination issues at the chuck center.
    """)
    show_image("outputs/module3_wafer_samples.png")

    st.subheader("Classification Results")
    show_image("outputs/module3_confusion_matrix.png")

    st.subheader("Grad-CAM: Where the Model Looks")
    st.markdown("""
    Grad-CAM generates a heatmap showing which spatial regions of the wafer 
    map drove the classification decision. This confirms the model is detecting 
    real physical patterns, not learning artifacts from the dataset.
    """)
    show_image("outputs/module3_gradcam.png")

    st.subheader("Training History")
    show_image("outputs/module3_training_curve.png")

    with st.expander("Engineering interpretation"):
        st.markdown("""
        In production, this type of classifier:
        - Replaces manual wafer map review by experienced engineers
        - Reduces root cause identification time from hours to seconds
        - Enables automated feedback to upstream process equipment
        - Loc and Scratch have lower precision because their spatial 
          signatures are geometrically similar at 32x32 resolution
        """)

# ── PAGE: MODULE 4 ────────────────────────────────────────────────────────────
elif page == "🧪 Process DOE Optimization":
    st.title("Module 4: PECVD Process DOE & Optimization")
    st.markdown("""
    **Process:** PECVD Silicon Nitride (Si₃N₄) deposition  
    **Design:** 2⁴ full factorial + 4 center points (20 runs total)  
    **Factors:** Temperature · RF Power · Pressure · SiH4 Flow  
    **Responses:** Deposition rate · Film stress · Refractive index · Uniformity
    """)
    st.info("Physics-informed simulation based on published PECVD Si₃N₄ literature. Parameter ranges derived from real CVD process windows. Not proprietary fab data.")

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Optimal Temperature", "334 °C")
    col2.metric("Optimal RF Power", "126 W")
    col3.metric("Dep. Rate at Optimum", "62 nm/min")
    col4.metric("Desirability Score", "0.702")

    st.markdown("---")
    st.subheader("DOE Run Results")
    if os.path.exists("outputs/module4_doe_results.csv"):
        doe_df = pd.read_csv("outputs/module4_doe_results.csv")
        st.dataframe(
            doe_df.style.format({
                'Dep_Rate': '{:.1f}',
                'Stress': '{:.1f}',
                'Ref_Index': '{:.4f}',
                'Uniformity': '{:.3f}'
            }),
            use_container_width=True
        )

    st.subheader("Main Effects Plot")
    st.markdown("""
    Shows how each factor independently affects each response. 
    RF Power has the strongest effect on deposition rate and stress. 
    Temperature drives the stress sign change (tensile vs compressive).
    """)
    show_image("outputs/module4_main_effects.png")

    st.subheader("Response Surface")
    st.markdown("""
    3D surfaces show how two factors jointly control the response. 
    The saddle shape of the stress surface confirms a temperature-RF 
    interaction: high RF compensates for high-temperature tensile stress.
    """)
    show_image("outputs/module4_response_surface.png")

    st.subheader("Multi-Response Optimization & Process Window")
    st.markdown("""
    Left: Pareto tradeoff between deposition rate and stress. 
    Points colored by uniformity. Red star = optimal operating point.  
    Right: Process window map showing desirability across 
    temperature-RF space. Green regions = viable operating conditions.
    """)
    show_image("outputs/module4_optimization.png")

    with st.expander("Engineering interpretation"):
        st.markdown("""
        This DOE analysis demonstrates:
        - 2⁴ full factorial design with center points for curvature detection
        - Multi-response optimization using geometric mean desirability
        - Process window identification for manufacturing transfer
        - Tradeoff visualization essential for process engineering decisions
        - Directly applicable to CVD, ALD, PVD process development at 
          Micron, TSMC, Lam Research, Applied Materials
        """)