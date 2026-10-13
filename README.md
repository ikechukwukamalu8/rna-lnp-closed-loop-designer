# 🧬 Closed-Loop Generative AI & Predictive RNA-LNP Discovery Platform

### 🚀 An Inverse-Design Generative Agent coupled with Predictive Ensembles and Statistically Guaranteed Conformal Guardrails
**Author:** Ikechukwu Okechi Kamalu, MSc (Biostatistics) & BSc (Microbiology)  
**Live Interactive Application:** *[]*

---

## 🔬 Preclinical Context & The Delivery Bottleneck

In modern RNA therapeutics (including mRNA vaccines, siRNA gene silencing, and CRISPR pipelines), designing the genetic sequence payload can now be done digitally in days. However, the critical bottleneck remains **targeted cellular delivery**. Because raw RNA is instantly degraded by the human immune system, it must be encapsulated inside microscopic fat bubbles called **Lipid Nanoparticles (LNPs)**.

Optimizing an LNP envelope requires navigating a highly complex, multi-parameter chemical and physical landscape (e.g., ionizable lipid molar concentrations, particle diameters, surface charges, and manufacturing encapsulation parameters). 

In traditional discovery loops, physically synthesizing and screening failed formulations in a wet-lab costs thousands of dollars and days of manual automation. Standard machine learning simulators can guide screening but are typically **overconfident**, leading to high-risk preclinical laboratory failures and wasted research budgets.

---

## 🤖 The Solution: Closed-Loop Generative AI & Biostatistics

This repository implements a production-ready, closed-loop computational framework designed to act as an automated, risk-aware discovery funnel for experimental laboratories.

[User Input: Target %] ➔ [Generative AI Agent] ➔ [Candidate Recipe] ➔ [Predictive Ensemble (RF + XGBoost)] ➔ [Conformal Safety Filter] ➔ [Robot Synthesis]


### 1. The Generative Engine (Inverse Design Mode)
Instead of forcing human scientists into a blind trial-and-error slider loop, this platform deploys a **Generative Optimization Agent**. When a researcher inputs a desired biological target (e.g., *85% Liver Transfection Efficiency*), the generative engine automatically searches across millions of continuous parameter spaces in milliseconds to invent and output the exact chemical blueprint recipe required to meet that therapeutic goal.

### 2. The Predictive Gatekeeper (Multi-Algorithmic Core)
Once a formulation recipe is generated (or manually input), it is instantly passed to a **Blended Machine Learning Ensemble** trained on historical wet-lab data:
-   **Random Forest Regressor:** A robust bagging ensemble designed to smooth out random laboratory measurement noise by averaging randomized decision paths.
-   **XGBoost Regressor:** A sequential gradient boosting framework optimized to isolate sharp, non-linear chemical and physical threshold cuts.

### 3. Biostatistics Guardrails (Split-Conformal Prediction)
To enforce strict economic guardrails, the blended ensemble is wrapped in **Split-Conformal Prediction**:
- The framework utilizes a holdout **Calibration Partition** to evaluate empirical absolute conformity scores (residuals) between predictions and real laboratory outcomes.
- It computes a strict quantile error boundary (q̂), allowing the user interface to output a **statistically guaranteed prediction interval with a strict 95% coverage guarantee**.
- **The Practical Impact:** If the generative agent creates a recipe that falls into a sparse region of the historical data, the app's conformal boundaries widen dramatically—flagging the candidate as structurally high-risk and stopping the lab from spending money on a blind guess.

---

## 📊 App Pipeline Data Architecture

The models process four baseline structural features coming straight from physical wet-lab instruments:
1.  **Ionizable Lipid Ratio (Molar %):** Controls the structural endosomal escape phase inside an acidic cellular environment.
2.  **Particle Size (nm):** Governs physical organ fenestration access points (e.g., targeting the liver vs. extrahepatic target tissues).
3.  **Zeta Potential (mV):** Gauges the net mechanical surface charge to prevent cellular toxicity and particle aggregation.
4.  **Encapsulation Efficiency (%):** Evaluates chemical manufacturing quality control parameters during microfluidic mixing.

**Target Target Label:** Projected Liver Transfection Efficiency Percentage (%).

---

## 💻 Local Installation & Deployment

To run this computational pipeline locally on your machine, clone the repository and install the dependencies:

```bash
# Clone the closed-loop discovery pipeline repository
git clone https://github.com
cd rna-lnp-closed-loop-designer

# Install core algorithm dependencies
pip install -r requirements.txt

# Launch the interactive Streamlit dashboard
streamlit run app.py
```

### Repository File Structure
*   `app.py`: Main interactive dashboard script holding cached training loops, generative inverse-design models, and the Streamlit layout.
*   `requirements.txt`: System package requirements listing `streamlit`, `numpy`, `pandas`, `scikit-learn`, and `xgboost`.
*   `README.md`: Structural documentation of project logic and preclinical relevance.

---

## 🎯 Sample Pipeline Diagnostic Output

```text
📋 Dataset loaded successfully!
📊 Data split profiles -> Train: 680 | Calibrate: 170 | Test: 150
🤖 Base Blended Ensemble models trained successfully!
🎯 Conformal Error Boundary (q_hat): ±16.702% transfection variance.
🛡️ Statistical Guarantee: 95% of future predictions will fall within this window.

--- Generative AI Optimization Snapshot ---
Desired Liver Transfection Target: 85.00%
[🧠 AI Searching Macromolecular Space...]

🧬 Generated Custom Blueprint:
 -> Generated Lipid Ratio: 51.42 Molar %
 -> Generated Particle Diameter: 82.50 nm
 -> Generated Zeta Potential: 11.20 mV
 -> Generated Encapsulation: 94.10%

🌲 Random Forest Component Model: 84.12%
⚡ XGBoost Component Model:       85.86%
🧠 Ensemble Consensus Prediction: 84.99%
🔮 95% Conformal Confidence Bounds: [68.29%, 100.00%]
🚀 Recommendation: Synthesize Candidate!
```
