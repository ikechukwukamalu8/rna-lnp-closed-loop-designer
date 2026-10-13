import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Generative RNA-LNP Designer",
    page_icon="🧬",
    layout="wide"
)

# --- TITLE AND SCIENTIFIC HOOK ---
st.title("🧬 Closed-Loop Generative AI & Predictive RNA-LNP Platform")
st.markdown("""
**Author:** Ikechukwu Okechi Kamalu, MSc  
*An advanced generative inverse-design and predictive ensemble framework for targeted delivery systems.*

---
### 🔬 The Closed-Loop AI Paradigm
This upgraded platform mirrors the cutting-edge "Design-Test-Learn" pipelines utilized by leading AI4Science research groups:
1. **The Generative Engine (Inverse Design):** Instead of forcing a human to guess chemical parameter values, this component acts as a generative agent. It scans the chemical landscape to **automatically generate** the ideal LNP recipe needed to achieve a user-defined biological target.
2. **The Predictive Gatekeeper (Ensemble + Conformal Prediction):** The generated recipe is immediately passed to a blended ensemble of **Random Forest** and **XGBoost** wrapped in **Split-Conformal Prediction** to verify its statistical confidence boundaries.
""")

# --- CACHED DATA & ENSEMBLE TRAINING LOOP ---
@st.cache_data
def train_ensemble_pipeline():
    np.random.seed(42)
    n_samples = 1000
    data = {
        'ionizable_lipid_ratio': np.random.uniform(35, 60, n_samples),
        'particle_size_nm': np.random.normal(85, 18, n_samples),
        'zeta_potential': np.random.uniform(-10, 25, n_samples),
        'encapsulation_efficiency': np.random.uniform(75, 99, n_samples)
    }
    df = pd.DataFrame(data)
    size_deviation = np.abs(df['particle_size_nm'] - 85)
    ideal_size_factor = np.exp(-size_deviation / 30)
    true_signal = (
        0.35 * df['ionizable_lipid_ratio'] + 
        40.0 * ideal_size_factor + 
        0.25 * df['encapsulation_efficiency'] +
        0.15 * (df['zeta_potential'] + 10)
    )
    df['liver_transfection_efficiency'] = (true_signal - true_signal.min()) / (true_signal.max() - true_signal.min()) * 100

    X = df[['ionizable_lipid_ratio', 'particle_size_nm', 'zeta_potential', 'encapsulation_efficiency']]
    y = df['liver_transfection_efficiency']

    X_train_val, _, y_train_val, _ = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_cal, y_train, y_cal = train_test_split(X_train_val, y_train_val, test_size=0.20, random_state=42)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    xgb_model.fit(X_train, y_train)

    alpha = 0.05
    rf_cal_pred = rf_model.predict(X_cal)
    xgb_cal_pred = xgb_model.predict(X_cal)
    ensemble_cal_pred = (rf_cal_pred + xgb_cal_pred) / 2.0
    
    cal_scores = np.abs(y_cal - ensemble_cal_pred)
    n_cal = len(X_cal)
    q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    q_hat = float(np.quantile(cal_scores, q_level, method='higher'))
    
    return rf_model, xgb_model, q_hat

# Initialize predictive core
rf_model, xgb_model, q_hat = train_ensemble_pipeline()

# --- THE GENERATIVE ENGINE CORE (Stochastic Optimization) ---
def generate_optimal_formulation(target_efficiency, rf, xgb, steps=150):
    best_recipe = None
    min_error = float('inf')
    
    for _ in range(steps):
        candidate = {
            'ionizable_lipid_ratio': float(np.random.uniform(35.0, 60.0)),
            'particle_size_nm': float(np.random.uniform(30.0, 150.0)),
            'zeta_potential': float(np.random.uniform(-10.0, 25.0)),
            'encapsulation_efficiency': float(np.random.uniform(70.0, 99.0))
        }
        cand_df = pd.DataFrame([candidate])
        
        p_rf = float(rf.predict(cand_df))
        p_xgb = float(xgb.predict(cand_df))
        p_ensemble = (p_rf + p_xgb) / 2.0
        
        error = np.abs(target_efficiency - p_ensemble)
        if error < min_error:
            min_error = error
            best_recipe = candidate
            
    return best_recipe

# --- SIDEBAR CONTROL CENTER ---
st.sidebar.header("🕹️ Mode Selection Control Panel")
app_mode = st.sidebar.radio("Choose Workflow Mode:", ["1. Manual Predictive Testing", "2. Generative Inverse Design"])

if app_mode == "1. Manual Predictive Testing":
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧪 Manual Formulation Knobs")
    user_lipid = st.sidebar.slider("Ionizable Lipid Ratio (Molar %)", 35.0, 60.0, 45.0, step=0.1)
    user_size = st.sidebar.slider("Particle Size (Diameter in nm)", 30.0, 150.0, 85.0, step=0.5)
    user_zeta = st.sidebar.slider("Zeta Potential (Surface Charge in mV)", -10.0, 25.0, 5.0, step=0.1)
    user_encap = st.sidebar.slider("Encapsulation Efficiency (%)", 70.0, 99.0, 90.0, step=0.1)
    
    input_features = pd.DataFrame([{
        'ionizable_lipid_ratio': user_lipid,
        'particle_size_nm': user_size,
        'zeta_potential': user_zeta,
        'encapsulation_efficiency': user_encap
    }])

else:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Target Therapeutic Metrics")
    target_eff = st.sidebar.slider("Desired Liver Transfection Efficiency (%)", 10.0, 100.0, 85.0, step=0.5)
    
    with st.spinner("🧠 Generative AI Agent searching the macromolecular space..."):
        generated_recipe = generate_optimal_formulation(target_eff, rf_model, xgb_model)
        
    input_features = pd.DataFrame([generated_recipe])
    user_lipid = float(generated_recipe['ionizable_lipid_ratio'])
    user_size = float(generated_recipe['particle_size_nm'])
    user_zeta = float(generated_recipe['zeta_potential'])
    user_encap = float(generated_recipe['encapsulation_efficiency'])

# --- PREDICTION CORE ---
rf_pred = float(rf_model.predict(input_features))
xgb_pred = float(xgb_model.predict(input_features))
ensemble_pred = (rf_pred + xgb_pred) / 2.0
lower_bound = float(max(0, ensemble_pred - q_hat))
upper_bound = float(min(100, ensemble_pred + q_hat))

# --- VISUAL DASHBOARD LAYOUT ---
if app_mode == "2. Generative Inverse Design":
    st.subheader("🎨 Generative AI Design Matrix Output")
    st.markdown(f"The AI has generated the following custom biochemical molecular blueprint to hit your target of **{target_eff:.1f}%** efficiency:")
    
    g_col1, g_col2, g_col3, g_col4 = st.columns(4)
    g_col1.metric("Generated Lipid Ratio", f"{user_lipid:.2f} Molar %")
    g_col2.metric("Generated Particle Diameter", f"{user_size:.1f} nm")
    g_col3.metric("Generated Zeta Potential", f"{user_zeta:.2f} mV")
    g_col4.metric("Generated Encapsulation", f"{user_encap:.1f}%")
    st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Predictive Gatekeeper Outputs")
    st.metric(
        label="Blended Ensemble Transfection Consensus", 
        value=f"{ensemble_pred:.2f}%"
    )
    st.markdown(f"**95% Statistical Confidence Boundaries:**")
    st.success(f" Lower Bound Limit: **{lower_bound:.2f}%**")
    st.success(f" Upper Bound Limit: **{upper_bound:.2f}%**")
    st.caption(f"Ensemble Conformal Error Margin tolerance verified at: ±{q_hat:.3f}%")

with col2:
    st.subheader("📊 Algorithmic Validation Log")
    st.markdown("🧑‍💻 **Behind-the-Scenes Architecture Predictions:**")
    st.write(f"🌲 **Random Forest Component Model:** `{rf_pred:.2f}%`")
    st.write(f"⚡ **XGBoost Component Model:** `{xgb_pred:.2f}%`")
    
    st.markdown("#### 🛡️ Active Decision Support Guardrail")
    if ensemble_pred >= 70:
        st.balloons()
        st.info("🚀 **Recommendation: Synthesize Candidate!** High algorithm consensus indicates a highly stable, high-efficiency structure.")
    elif ensemble_pred < 40:
        st.error("🛑 **Recommendation: Redesign Formulation.** Consensus prediction falls deep below baseline preclinical delivery requirements.")
    else:
        st.warning("⚠️ **Recommendation: Proceed with Limited Wet-Lab Batches.** Borderline properties flag this as a risky formulation.")

st.markdown("---")
st.subheader("🌐 Real-time Pipeline Infrastructure Stream")
st.json({
    "Current Pipeline Mode": app_mode,
    "Active Active Structural Vector": {
        "Lipid Molar Percent": round(user_lipid, 4),
        "Diameter Metrics": round(user_size, 4),
        "Zeta Charge": round(user_zeta, 4),
        "Encapsulation Efficiency": round(user_encap, 4)
    },
    "Ensemble Framework Specifications": {
        "Random Forest Component Vector": round(rf_pred, 4),
        "XGBoost Component Vector": round(xgb_pred, 4),
        "Calculated Conformal Interval (q_hat)": round(q_hat, 4),
        "Absolute Variance Range Span": round(upper_bound - lower_bound, 4)
    }
})
