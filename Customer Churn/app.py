"""
Churn Cutie Dashboard
A soft pastel Streamlit dashboard for the Telco customer churn project.

Run with:   streamlit run app.py

Needs (in the same folder): Telco-Customer-Churn.csv, rf_churn_model.joblib, xgb_churn_model.joblib
(the two .joblib files are created by running the notebook churn_classifier.ipynb)
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------ page setup
st.set_page_config(page_title="Churn Cutie Dashboard", page_icon="🌸", layout="wide")

BASE = Path(__file__).parent
DATA_FILE = BASE / "Telco-Customer-Churn.csv"
MODEL_FILES = {
    "Random Forest": BASE / "rf_churn_model.joblib",
    "XGBoost": BASE / "xgb_churn_model.joblib",
}

# ------------------------------------------------------------------ palette
PINK, PINK_DEEP = "#FFB3C7", "#FF7AA2"
LAVENDER, LAVENDER_DEEP = "#D8C8FF", "#A98BFF"
MINT, MINT_DEEP = "#BDF0DC", "#5FCFA6"
PEACH, SKY, BUTTER = "#FFD6BA", "#BFE3FF", "#FFF0B3"
INK, INK_SOFT = "#5B4B7A", "#8F82A8"
STAY_COLOR, CHURN_COLOR = MINT_DEEP, PINK_DEEP
MODEL_COLORS = {"Random Forest": MINT_DEEP, "XGBoost": LAVENDER_DEEP}

# ------------------------------------------------------------------ styling
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@400;500;600&family=Quicksand:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stMarkdown, p, label, span, div {{
    font-family: 'Quicksand', 'Nunito', 'Segoe UI', Arial, sans-serif;
    color: {INK};
}}
h1, h2, h3, h4 {{ font-family: 'Fredoka', 'Quicksand', 'Segoe UI', Arial, sans-serif !important; color: {INK} !important; }}

.stApp {{
    background: linear-gradient(160deg, #FFF6FA 0%, #F6F1FF 45%, #EEF8FF 100%);
}}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.6rem; max-width: 1250px; }}

/* sidebar */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #FFE8F0 0%, #EFE6FF 60%, #E3F3FF 100%);
    border-right: 2px dashed {PINK};
}}

/* hero */
.hero {{
    background: linear-gradient(120deg, {PINK} 0%, {LAVENDER} 50%, {SKY} 100%);
    border-radius: 30px; padding: 28px 34px; margin-bottom: 18px;
    box-shadow: 0 10px 28px rgba(169,139,255,.25);
    position: relative; overflow: hidden;
}}
.stApp .hero h1, .stApp .hero h1 * {{ color: white !important; }}
.hero h1 {{ margin: 0; font-size: 2.3rem; text-shadow: 0 2px 8px rgba(120,90,180,.35); }}
.stApp .hero p {{ margin: 6px 0 0 0; font-size: 1.05rem; color: white !important; font-weight: 600; }}
.stApp .hero .notes {{ color: white !important; position: absolute; right: 28px; top: 14px; font-size: 2.1rem; opacity: .8; letter-spacing: 8px; }}

/* kpi cards */
.kpi {{
    border-radius: 24px; padding: 18px 20px; text-align: center;
    box-shadow: 0 6px 18px rgba(120,90,180,.12); border: 2px solid rgba(255,255,255,.8);
}}
.kpi .emoji {{ font-size: 1.7rem; }}
.kpi .value {{ font-family: 'Fredoka', sans-serif; font-size: 1.9rem; font-weight: 600; color: {INK}; line-height: 1.2; }}
.kpi .label {{ font-size: .9rem; font-weight: 700; color: {INK_SOFT}; text-transform: uppercase; letter-spacing: .6px; }}

/* soft section cards for charts */
div[data-testid="stPlotlyChart"] {{
    background: rgba(255,255,255,.78); border-radius: 24px; padding: 10px;
    box-shadow: 0 6px 18px rgba(120,90,180,.10); border: 2px solid rgba(255,255,255,.9);
}}

/* tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: 10px; border-bottom: none; }}
.stTabs button[role="tab"] {{
    background: rgba(255,255,255,.85); border-radius: 999px; padding: 8px 20px;
    border: 2px solid {PINK}; height: auto;
}}
.stTabs button[role="tab"] p {{ font-weight: 700; font-size: 1rem; }}
.stTabs button[role="tab"][aria-selected="true"] {{
    background: linear-gradient(90deg, {PINK}, {LAVENDER}); border-color: {LAVENDER_DEEP};
}}
.stTabs button[role="tab"][aria-selected="true"] p {{ color: white !important; }}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display: none; }}

/* pastel accents instead of Streamlit's default red */
span[data-baseweb="tag"] {{ background: {PINK} !important; border-radius: 999px !important; }}
span[data-baseweb="tag"] span {{ color: {INK} !important; font-weight: 700; }}
div[data-baseweb="slider"] div[role="slider"] {{ background-color: {PINK_DEEP} !important; border-color: white !important; }}
div[data-testid="stSliderThumbValue"], div[data-testid="stSliderTickBarMin"], div[data-testid="stSliderTickBarMax"] {{ color: {INK} !important; }}
div[data-baseweb="radio"] > div:first-child {{ border-color: {PINK_DEEP} !important; }}

/* buttons */
.stButton > button, .stFormSubmitButton > button {{
    background: linear-gradient(90deg, {PINK_DEEP}, {LAVENDER_DEEP}); color: white; border: none;
    border-radius: 999px; padding: 10px 30px; font-weight: 700; font-size: 1.05rem;
    box-shadow: 0 6px 16px rgba(255,122,162,.35);
}}
.stButton > button:hover, .stFormSubmitButton > button:hover {{ filter: brightness(1.06); color: white; }}

/* result card */
.result {{ border-radius: 28px; padding: 22px 26px; text-align: center; box-shadow: 0 8px 22px rgba(120,90,180,.15); }}
.result h2 {{ margin: 4px 0; }}
.tip {{
    background: rgba(255,255,255,.85); border-left: 6px solid {PINK_DEEP}; border-radius: 16px;
    padding: 10px 16px; margin: 8px 0; font-weight: 600;
}}
.cap {{ color: {INK_SOFT}; font-size: .85rem; }}
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ helpers
def show(fig):
    """Plotly chart that works on both new and old Streamlit versions."""
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def style(fig, height=380, title=None):
    fig.update_layout(
        height=height,
        title=dict(text=title, font=dict(family="Fredoka, Quicksand, Segoe UI, Arial, sans-serif", size=19, color=INK)) if title else None,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Quicksand, Nunito, Segoe UI, Arial, sans-serif", color=INK, size=13),
        margin=dict(l=20, r=20, t=55 if title else 20, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5, bgcolor="rgba(255,255,255,0)"),
    )
    fig.update_xaxes(gridcolor="rgba(169,139,255,.18)", zerolinecolor="rgba(169,139,255,.25)")
    fig.update_yaxes(gridcolor="rgba(169,139,255,.18)", zerolinecolor="rgba(169,139,255,.25)")
    return fig


def kpi(emoji, value, label, color):
    return (f'<div class="kpi" style="background:{color}"><div class="emoji">{emoji}</div>'
            f'<div class="value">{value}</div><div class="label">{label}</div></div>')


# ------------------------------------------------------------------ data + models
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["ChurnFlag"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


@st.cache_resource
def load_models():
    return {name: joblib.load(path) for name, path in MODEL_FILES.items()}


@st.cache_data
def evaluate():
    """Rebuild the exact same test split as the notebook and score both models."""
    df = load_data()
    X = df.drop(columns=["customerID", "Churn", "ChurnFlag"])
    y = df["ChurnFlag"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    models = load_models()
    proba = {n: m.predict_proba(X_test)[:, 1] for n, m in models.items()}
    return y_test.to_numpy(), proba


missing = [p.name for p in [DATA_FILE, *MODEL_FILES.values()] if not p.exists()]
if missing:
    st.error("🌧️ Missing file(s): " + ", ".join(missing) +
             ". Put them in the same folder as app.py (run the notebook first to create the .joblib models).")
    st.stop()

df = load_data()
models = load_models()
y_test, proba = evaluate()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## 🎀 Settings")
    model_name = st.radio("Choose a model", list(models), index=1)
    st.markdown(
        f'<p class="cap">Random Forest ✿ and XGBoost ♪ were both trained with SMOTE '
        f'to handle the small number of churners.</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🌷 Filter the overview")
    contracts = st.multiselect("Contract type", sorted(df["Contract"].unique()),
                               default=sorted(df["Contract"].unique()))
    st.markdown("---")
    st.markdown('<p class="cap" style="text-align:center">♪ ♫ made with love & pastel ♫ ♪</p>',
                unsafe_allow_html=True)

# ------------------------------------------------------------------ hero
st.markdown(
    """
<div class="hero">
  <div class="notes">♪ ♫ ✿</div>
  <h1>🌸 Churn Cutie Dashboard</h1>
  <p>Gentle insights to keep your customers smiling and staying ✨</p>
</div>
""",
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["🌸 Overview", "🎀 Model Performance", "🍓 Feature Importance", "🔮 Predict a Customer"])

# ================================================================== TAB 1: overview
with tab1:
    view = df[df["Contract"].isin(contracts)]
    if view.empty:
        st.info("Pick at least one contract type in the sidebar 🌷")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(kpi("👥", f"{len(view):,}", "Customers", PINK), unsafe_allow_html=True)
        c2.markdown(kpi("💔", f"{view['ChurnFlag'].mean():.1%}", "Churn rate", LAVENDER),
                    unsafe_allow_html=True)
        c3.markdown(kpi("⏳", f"{view['tenure'].mean():.0f} mo", "Avg. tenure", MINT),
                    unsafe_allow_html=True)
        c4.markdown(kpi("💰", f"${view['MonthlyCharges'].mean():.0f}", "Avg. monthly bill", BUTTER),
                    unsafe_allow_html=True)
        st.write("")

        left, right = st.columns(2)
        with left:
            counts = view["Churn"].value_counts()
            fig = go.Figure(go.Pie(
                labels=["Stayed", "Churned"],
                values=[counts.get("No", 0), counts.get("Yes", 0)],
                hole=0.62, marker=dict(colors=[STAY_COLOR, CHURN_COLOR],
                                       line=dict(color="white", width=4)),
                textinfo="percent", textfont=dict(size=16, color="white"),
            ))
            fig.add_annotation(text="🌸", showarrow=False, font=dict(size=40))
            show(style(fig, title="Who stays, who leaves?"))
        with right:
            g = view.groupby("Contract")["ChurnFlag"].mean().reset_index()
            fig = go.Figure(go.Bar(
                x=g["Contract"], y=g["ChurnFlag"], marker_color=[PINK_DEEP, LAVENDER_DEEP, MINT_DEEP],
                text=[f"{v:.0%}" for v in g["ChurnFlag"]], textposition="outside",
                marker=dict(cornerradius=12)))
            fig.update_yaxes(tickformat=".0%", title="Churn rate", range=[0, g["ChurnFlag"].max() * 1.25])
            show(style(fig, title="Churn rate by contract"))

        left, right = st.columns(2)
        with left:
            fig = go.Figure()
            for label, flag, color in [("Stayed", 0, STAY_COLOR), ("Churned", 1, CHURN_COLOR)]:
                fig.add_histogram(x=view.loc[view["ChurnFlag"] == flag, "tenure"], name=label,
                                  marker_color=color, opacity=0.75, nbinsx=36)
            fig.update_layout(barmode="overlay")
            fig.update_xaxes(title="Tenure (months)")
            fig.update_yaxes(title="Customers")
            show(style(fig, title="New customers leave the most"))
        with right:
            g = view.groupby("PaymentMethod")["ChurnFlag"].mean().sort_values().reset_index()
            fig = go.Figure(go.Bar(
                y=g["PaymentMethod"], x=g["ChurnFlag"], orientation="h",
                marker=dict(color=[SKY, LAVENDER, PEACH, PINK_DEEP][: len(g)], cornerradius=12),
                text=[f"{v:.0%}" for v in g["ChurnFlag"]], textposition="outside"))
            fig.update_xaxes(tickformat=".0%", title="Churn rate", range=[0, g["ChurnFlag"].max() * 1.2])
            show(style(fig, title="Churn rate by payment method"))

# ================================================================== TAB 2: performance
with tab2:
    st.markdown(f"### 🎀 How well does **{model_name}** guess?")
    thr = st.slider("Decision threshold (a customer counts as 'will churn' above this score)",
                    0.10, 0.90, 0.50, 0.05)
    p = proba[model_name]
    yhat = (p >= thr).astype(int)

    auc = roc_auc_score(y_test, p)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.markdown(kpi("🎯", f"{auc:.3f}", "ROC-AUC", PINK), unsafe_allow_html=True)
    m2.markdown(kpi("✅", f"{accuracy_score(y_test, yhat):.1%}", "Accuracy", LAVENDER), unsafe_allow_html=True)
    m3.markdown(kpi("🔍", f"{precision_score(y_test, yhat, zero_division=0):.1%}", "Precision", MINT),
                unsafe_allow_html=True)
    m4.markdown(kpi("🧲", f"{recall_score(y_test, yhat):.1%}", "Recall", BUTTER), unsafe_allow_html=True)
    m5.markdown(kpi("🌟", f"{f1_score(y_test, yhat):.1%}", "F1-score", SKY), unsafe_allow_html=True)
    if auc > 0.80:
        st.success("🎉 ROC-AUC is above the 0.80 goal!")
    st.write("")

    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        for name, pr in proba.items():
            fpr, tpr, _ = roc_curve(y_test, pr)
            fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC {roc_auc_score(y_test, pr):.3f})",
                            line=dict(color=MODEL_COLORS[name], width=4 if name == model_name else 2.5))
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                        line=dict(color=INK_SOFT, dash="dot"))
        fig.update_xaxes(title="False positive rate")
        fig.update_yaxes(title="True positive rate")
        fig = style(fig, height=430, title="ROC curves")
        fig.update_layout(legend=dict(orientation="v", x=0.45, y=0.05, yanchor="bottom", xanchor="left",
                                      bgcolor="rgba(255,255,255,.7)"))
        show(fig)
    with right:
        cm = confusion_matrix(y_test, yhat)
        labels = ["Stayed", "Churned"]
        fig = go.Figure(go.Heatmap(
            z=cm, x=labels, y=labels, showscale=False,
            colorscale=[[0, "#FFF1F6"], [0.5, "#E9DDFF"], [1, "#B79BFF"]],
            text=cm, texttemplate="%{text}", textfont=dict(size=26, color=INK), xgap=6, ygap=6))
        fig.update_xaxes(title="Predicted")
        fig.update_yaxes(title="Actual", autorange="reversed")
        show(style(fig, height=430, title=f"Confusion matrix (threshold {thr:.2f})"))
    st.markdown(
        f'<p class="cap">Lower the threshold → catch more leavers (higher recall) but more false alarms. '
        f'Scores are computed on the same 20% test set as the notebook ({len(y_test):,} customers).</p>',
        unsafe_allow_html=True)

# ================================================================== TAB 3: importance
with tab3:
    st.markdown(f"### 🍓 What does **{model_name}** care about most?")
    n_top = st.slider("How many features to show", 5, 20, 10)
    pipe = models[model_name]
    names = [n.replace("cat__", "").replace("num__", "")
             for n in pipe.named_steps["prep"].get_feature_names_out()]
    imp = pd.Series(pipe.named_steps["model"].feature_importances_, index=names)
    top = imp.sort_values(ascending=False).head(n_top)[::-1]
    pastel_cycle = [PINK_DEEP, LAVENDER_DEEP, MINT_DEEP, "#FFB48A", "#7CC4FF"]
    fig = go.Figure(go.Bar(
        x=top.values, y=top.index, orientation="h",
        marker=dict(color=[pastel_cycle[i % 5] for i in range(len(top))], cornerradius=10),
        text=[f"{v:.3f}" for v in top.values], textposition="outside"))
    fig.update_xaxes(title="Importance", range=[0, top.max() * 1.18])
    show(style(fig, height=max(380, 38 * n_top), title=f"Top {n_top} features: {model_name}"))
    st.markdown(
        '<div class="tip">🌷 Month-to-month contracts and missing online security / tech support '
        'are the biggest warning signs.</div>', unsafe_allow_html=True)

# ================================================================== TAB 4: predict
with tab4:
    st.markdown(f"### 🔮 Will this customer stay? (using **{model_name}**)")
    opts = {c: sorted(df[c].unique()) for c in df.columns
            if not pd.api.types.is_numeric_dtype(df[c]) and c not in ("customerID", "Churn")}

    with st.form("predict_form"):
        a, b, c = st.columns(3)
        with a:
            st.markdown("**🧸 About the customer**")
            gender = st.selectbox("Gender", opts["gender"])
            senior = st.selectbox("Senior citizen", ["No", "Yes"])
            partner = st.selectbox("Has partner", opts["Partner"])
            dependents = st.selectbox("Has dependents", opts["Dependents"])
            tenure = st.slider("Tenure (months)", 0, 72, 6)
        with b:
            st.markdown("**📞 Services**")
            phone = st.selectbox("Phone service", opts["PhoneService"], index=1)
            multi = st.selectbox("Multiple lines", opts["MultipleLines"])
            internet = st.selectbox("Internet service", opts["InternetService"], index=1)
            security = st.selectbox("Online security", opts["OnlineSecurity"])
            backup = st.selectbox("Online backup", opts["OnlineBackup"])
            protection = st.selectbox("Device protection", opts["DeviceProtection"])
        with c:
            st.markdown("**🧾 Plan & billing**")
            support = st.selectbox("Tech support", opts["TechSupport"])
            tv = st.selectbox("Streaming TV", opts["StreamingTV"])
            movies = st.selectbox("Streaming movies", opts["StreamingMovies"])
            contract = st.selectbox("Contract", opts["Contract"])
            paperless = st.selectbox("Paperless billing", opts["PaperlessBilling"], index=1)
            payment = st.selectbox("Payment method", opts["PaymentMethod"])
            monthly = st.slider("Monthly charges ($)", 18.0, 120.0, 70.0, 0.5)
        go_btn = st.form_submit_button("✨ Predict churn risk ✨")

    if go_btn:
        row = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone, "MultipleLines": multi,
            "InternetService": internet, "OnlineSecurity": security, "OnlineBackup": backup,
            "DeviceProtection": protection, "TechSupport": support, "StreamingTV": tv,
            "StreamingMovies": movies, "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment, "MonthlyCharges": monthly,
            "TotalCharges": round(monthly * tenure, 2),
        }])
        score = float(models[model_name].predict_proba(row)[0, 1])

        if score < 0.35:
            level, emoji, bg, msg = "Low risk", "🌈", MINT, "This customer looks happy and likely to stay!"
        elif score < 0.6:
            level, emoji, bg, msg = "Medium risk", "🌤️", BUTTER, "Worth keeping an eye on this one."
        else:
            level, emoji, bg, msg = "High risk", "🥺", PINK, "This customer might be about to leave."

        left, right = st.columns([1, 1])
        with left:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=score * 100,
                number=dict(suffix="%", font=dict(family="Fredoka, Quicksand, Arial, sans-serif", color=INK)),
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor=INK_SOFT),
                    bar=dict(color=LAVENDER_DEEP, thickness=0.28),
                    bgcolor="white", borderwidth=0,
                    steps=[dict(range=[0, 35], color=MINT),
                           dict(range=[35, 60], color=BUTTER),
                           dict(range=[60, 100], color=PINK)])))
            show(style(fig, height=330, title="Churn score"))
        with right:
            st.markdown(
                f'<div class="result" style="background:{bg}"><div style="font-size:3rem">{emoji}</div>'
                f'<h2>{level}</h2><p style="font-weight:600">{msg}</p></div>', unsafe_allow_html=True)
            st.write("")
            tips = []
            if contract == "Month-to-month":
                tips.append("🎁 Offer a discount to move to a 1- or 2-year contract.")
            if tenure < 12:
                tips.append("💌 New customer: send a friendly welcome check-in.")
            if security == "No" and internet != "No":
                tips.append("🛡️ Offer a free trial of online security.")
            if support == "No" and internet != "No":
                tips.append("🧸 Offer tech support as a low-cost add-on.")
            if payment == "Electronic check":
                tips.append("💳 Suggest automatic card or bank payment.")
            for t in (tips or ["🌸 Nothing special to fix, keep up the great service!"]):
                st.markdown(f'<div class="tip">{t}</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="cap">The models were trained on SMOTE-balanced data, so the score is best read as a '
            '<b>relative risk</b> (higher = more likely to leave), not an exact probability.</p>',
            unsafe_allow_html=True)
    else:
        st.markdown('<div class="tip">🌼 Fill in the details above and press the button to see the risk.</div>',
                    unsafe_allow_html=True)
