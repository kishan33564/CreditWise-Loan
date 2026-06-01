import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CreditWise Loan Predictor",
    page_icon="🏦",
    layout="wide",
)

st.markdown("""
<style>
    .main-title { font-size: 2rem; font-weight: 700; }
    .metric-box { background: #f0f2f6; border-radius: 10px; padding: 1rem; text-align: center; }
    .approved { background-color: #d4edda; border: 1px solid #28a745;
                border-radius: 10px; padding: 1.2rem; text-align: center; }
    .denied   { background-color: #f8d7da; border: 1px solid #dc3545;
                border-radius: 10px; padding: 1.2rem; text-align: center; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar: upload or use demo data ─────────────────────────────────────────
st.sidebar.title("🏦 CreditWise")
st.sidebar.markdown("---")
uploaded = st.sidebar.file_uploader(
    "Upload your loan_approval_data.csv", type=["csv"]
)
use_demo = st.sidebar.checkbox("Use demo / synthetic data", value=(uploaded is None))

# ── Load / generate data ─────────────────────────────────────────────────────
@st.cache_data
def generate_demo_data(n=500, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Applicant_ID":       [f"APP{i:04d}" for i in range(n)],
        "Applicant_Income":   rng.integers(15000, 200000, n),
        "Coapplicant_Income": rng.integers(0, 80000, n),
        "Loan_Amount":        rng.integers(50000, 5000000, n),
        "Credit_Score":       rng.integers(300, 850, n),
        "DTI_Ratio":          rng.uniform(5, 70, n).round(2),
        "Savings":            rng.integers(0, 2000000, n),
        "Loan_Term":          rng.choice([12, 24, 36, 60, 120, 180, 240, 360], n),
        "Gender":             rng.choice(["Male", "Female"], n),
        "Marital_Status":     rng.choice(["Married", "Single", "Divorced"], n),
        "Education_Level":    rng.choice(["Graduate", "Not Graduate", "Postgraduate"], n),
        "Employment_Status":  rng.choice(["Employed", "Self-Employed", "Unemployed"], n),
        "Employer_Category":  rng.choice(["Private", "Government", "NGO", "Self"], n),
        "Property_Area":      rng.choice(["Urban", "Semiurban", "Rural"], n),
        "Loan_Purpose":       rng.choice(["Home", "Education", "Business", "Personal"], n),
    })
    score = (
        (df["Credit_Score"] - 300) / 550 * 40
        + df["Applicant_Income"] / 200000 * 25
        + (1 - df["DTI_Ratio"] / 70) * 20
        + df["Savings"] / 2000000 * 15
        + rng.normal(0, 5, n)
    )
    df["Loan_Approved"] = np.where(score >= 45, "Y", "N")
    return df

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

if uploaded and not use_demo:
    df_raw = load_data(uploaded)
else:
    df_raw = generate_demo_data()

# ── Preprocess ───────────────────────────────────────────────────────────────
@st.cache_data
def preprocess(df_raw):
    df = df_raw.copy()
    df = df.drop(columns=["Applicant_ID"], errors="ignore")

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    num_cols = df.select_dtypes(include="number").columns.tolist()

    # Impute
    num_imp = SimpleImputer(strategy="mean")
    df[num_cols] = num_imp.fit_transform(df[num_cols])
    cat_imp = SimpleImputer(strategy="most_frequent")
    df[cat_cols] = cat_imp.fit_transform(df[cat_cols])

    # Encode target + education
    le = LabelEncoder()
    df["Loan_Approved"]   = le.fit_transform(df["Loan_Approved"])
    df["Education_Level"] = le.fit_transform(df["Education_Level"])

    # One-hot encode remaining categoricals (excluding target)
    ohe_cols = [c for c in cat_cols if c not in ("Loan_Approved", "Education_Level")]
    if ohe_cols:
        ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        encoded = ohe.fit_transform(df[ohe_cols])
        enc_df  = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(ohe_cols), index=df.index)
        df = pd.concat([df.drop(columns=ohe_cols), enc_df], axis=1)

    # Feature engineering
    df["DTI_Ratio_sq"]    = df["DTI_Ratio"] ** 2
    df["Credit_Score_sq"] = df["Credit_Score"] ** 2

    X = df.drop(columns=["Loan_Approved", "Credit_Score", "DTI_Ratio"], errors="ignore")
    y = df["Loan_Approved"]
    return X, y, df

X, y, df_processed = preprocess(df_raw)

# ── Train models ─────────────────────────────────────────────────────────────
@st.cache_resource
def train_models(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN (k=5)":           KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes":         GaussianNB(),
    }
    results = {}
    for name, m in models.items():
        m.fit(X_tr_s, y_tr)
        pred = m.predict(X_te_s)
        results[name] = {
            "model":     m,
            "precision": precision_score(y_te, pred),
            "recall":    recall_score(y_te, pred),
            "f1":        f1_score(y_te, pred),
            "accuracy":  accuracy_score(y_te, pred),
            "cm":        confusion_matrix(y_te, pred),
        }
    return results, scaler, X_tr.columns.tolist(), X_te_s, y_te

results, scaler, feature_cols, X_te_s, y_te = train_models(X, y)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="main-title">🏦 CreditWise — Loan Approval Predictor</p>', unsafe_allow_html=True)
st.markdown("Explore loan approval patterns, compare ML models, and predict outcomes in real time.")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict", "📊 EDA Charts", "🤖 Model Results", "📋 Data"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 · PREDICT
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Enter applicant details")

    col1, col2, col3 = st.columns(3)
    with col1:
        income      = st.slider("Applicant income (₹/month)", 10000, 200000, 60000, step=5000)
        credit      = st.slider("Credit score", 300, 850, 680, step=1)
        dti         = st.slider("DTI ratio (%)", 0, 80, 25, step=1)
    with col2:
        savings     = st.slider("Savings (₹)", 0, 2000000, 200000, step=50000)
        coapp       = st.slider("Coapplicant income (₹/month)", 0, 100000, 0, step=5000)
        loan_amt    = st.slider("Loan amount (₹)", 50000, 5000000, 500000, step=50000)
    with col3:
        emp_status  = st.selectbox("Employment status",  ["Employed", "Self-Employed", "Unemployed"])
        education   = st.selectbox("Education level",    ["Graduate", "Not Graduate", "Postgraduate"])
        prop_area   = st.selectbox("Property area",      ["Urban", "Semiurban", "Rural"])
        loan_purp   = st.selectbox("Loan purpose",       ["Home", "Education", "Business", "Personal"])
        model_name  = st.selectbox("Prediction model",   list(results.keys()))

    predict_btn = st.button("🔮 Predict loan approval", use_container_width=True)

    if predict_btn:
        # Build a one-row dataframe that mimics the training features
        row = {c: 0.0 for c in feature_cols}

        # Fill numeric features (present in training)
        mapping = {
            "Applicant_Income": income,
            "Coapplicant_Income": coapp,
            "Loan_Amount": loan_amt,
            "Savings": savings,
            "Loan_Term": 360,
            "Education_Level": {"Graduate": 0, "Not Graduate": 1, "Postgraduate": 2}.get(education, 0),
            "DTI_Ratio_sq": dti ** 2,
            "Credit_Score_sq": credit ** 2,
        }
        for k, v in mapping.items():
            if k in row:
                row[k] = v

        # One-hot flags (best-effort matching)
        emp_map = {"Employed": "Employment_Status_Self-Employed",
                   "Self-Employed": "Employment_Status_Unemployed"}
        for ohe_col, val in [
            (f"Employment_Status_{emp_status}", 1),
            (f"Property_Area_{prop_area}", 1),
            (f"Loan_Purpose_{loan_purp}", 1),
        ]:
            if ohe_col in row:
                row[ohe_col] = 1.0

        X_input = pd.DataFrame([row])[feature_cols]
        X_scaled = scaler.transform(X_input)

        chosen = results[model_name]["model"]
        pred   = chosen.predict(X_scaled)[0]
        prob   = chosen.predict_proba(X_scaled)[0] if hasattr(chosen, "predict_proba") else None

        st.markdown("---")
        res_col1, res_col2 = st.columns([2, 1])
        with res_col1:
            if pred == 1:
                st.markdown('<div class="approved"><h2>✅ Loan Approved</h2><p>The applicant meets the lending criteria.</p></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="denied"><h2>❌ Loan Denied</h2><p>The applicant does not meet the lending criteria.</p></div>', unsafe_allow_html=True)

            if prob is not None:
                confidence = prob[pred] * 100
                st.progress(int(confidence))
                st.caption(f"Model confidence: {confidence:.1f}%")

        with res_col2:
            st.markdown("**Key factors considered:**")
            factors = {
                "Credit score": credit,
                "Income (₹)": f"{income:,}",
                "DTI ratio": f"{dti}%",
                "Savings (₹)": f"{savings:,}",
                "Loan amount (₹)": f"{loan_amt:,}",
            }
            for k, v in factors.items():
                st.write(f"• **{k}:** {v}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 · EDA
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Exploratory data analysis")

    # Row 1
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Loan approval distribution**")
        fig, ax = plt.subplots(figsize=(4, 4))
        counts = df_raw["Loan_Approved"].value_counts()
        ax.pie(counts, labels=counts.index, autopct="%1.1f%%",
               colors=["#4caf50", "#f44336"], startangle=90)
        ax.set_title("Approved vs Denied")
        st.pyplot(fig); plt.close()

    with c2:
        st.markdown("**Education level breakdown**")
        fig, ax = plt.subplots(figsize=(4, 4))
        edu_cnt = df_raw["Education_Level"].value_counts()
        sns.barplot(x=edu_cnt.values, y=edu_cnt.index, palette="Blues_r", ax=ax)
        for p in ax.patches:
            ax.text(p.get_width() + 1, p.get_y() + p.get_height() / 2,
                    f"{int(p.get_width())}", va="center", fontsize=9)
        ax.set_xlabel("Count"); ax.set_title("Applicants by education")
        st.pyplot(fig); plt.close()

    st.markdown("---")
    # Row 2 — income & credit histograms
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Applicant income distribution**")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.histplot(data=df_raw, x="Applicant_Income", hue="Loan_Approved",
                     bins=20, multiple="dodge", palette={"Y": "#4caf50", "N": "#f44336"}, ax=ax)
        ax.set_xlabel("Income (₹)")
        st.pyplot(fig); plt.close()

    with c4:
        st.markdown("**Credit score distribution**")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.histplot(data=df_raw, x="Credit_Score", hue="Loan_Approved",
                     bins=20, multiple="dodge", palette={"Y": "#2196f3", "N": "#ff9800"}, ax=ax)
        ax.set_xlabel("Credit score")
        st.pyplot(fig); plt.close()

    st.markdown("---")
    # Row 3 — box plots
    st.markdown("**Box plots — key numeric features by approval status**")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    pairs = [
        ("Applicant_Income", (0, 0)),
        ("Credit_Score",     (0, 1)),
        ("DTI_Ratio",        (1, 0)),
        ("Savings",          (1, 1)),
    ]
    for col, (r, c) in pairs:
        if col in df_raw.columns:
            sns.boxplot(data=df_raw, x="Loan_Approved", y=col,
                        palette={"Y": "#4caf50", "N": "#f44336"}, ax=axes[r][c])
            axes[r][c].set_title(col.replace("_", " "))
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("---")
    # Row 4 — correlation heatmap
    st.markdown("**Correlation heatmap**")
    num_df = df_processed.select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 · MODEL RESULTS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Model performance comparison")

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    best_prec = max(results, key=lambda k: results[k]["precision"])
    best_acc  = max(results, key=lambda k: results[k]["accuracy"])
    best_rec  = max(results, key=lambda k: results[k]["recall"])
    best_f1   = max(results, key=lambda k: results[k]["f1"])
    m1.metric("Best precision",  f"{results[best_prec]['precision']:.1%}", best_prec)
    m2.metric("Best accuracy",   f"{results[best_acc]['accuracy']:.1%}",   best_acc)
    m3.metric("Best recall",     f"{results[best_rec]['recall']:.1%}",     best_rec)
    m4.metric("Best F1",         f"{results[best_f1]['f1']:.1%}",          best_f1)

    st.markdown("---")

    # Metrics table
    st.markdown("**Detailed metrics**")
    rows = []
    for name, r in results.items():
        rows.append({
            "Model":     name,
            "Precision": f"{r['precision']:.3f}",
            "Recall":    f"{r['recall']:.3f}",
            "F1 Score":  f"{r['f1']:.3f}",
            "Accuracy":  f"{r['accuracy']:.3f}",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

    st.markdown("---")

    # Bar chart comparison
    st.markdown("**Grouped bar chart — all metrics**")
    fig, ax = plt.subplots(figsize=(10, 4))
    metric_names = ["precision", "recall", "f1", "accuracy"]
    x = np.arange(len(metric_names))
    width = 0.25
    colors = ["#4285f4", "#34a853", "#ea4335"]
    for i, (name, r) in enumerate(results.items()):
        vals = [r[m] for m in metric_names]
        bars = ax.bar(x + i * width, vals, width, label=name, color=colors[i])
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.capitalize() for m in metric_names])
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("---")

    # Confusion matrices
    st.markdown("**Confusion matrices**")
    cms = st.columns(3)
    for idx, (name, r) in enumerate(results.items()):
        with cms[idx]:
            st.markdown(f"*{name}*")
            fig, ax = plt.subplots(figsize=(3.5, 3))
            sns.heatmap(r["cm"], annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Denied","Approved"],
                        yticklabels=["Denied","Approved"], ax=ax)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            plt.tight_layout()
            st.pyplot(fig); plt.close()

    st.markdown("---")
    # Correlation with target
    st.markdown("**Top features correlated with loan approval**")
    target_corr = (
        df_processed.select_dtypes(include="number")
        .corr()["Loan_Approved"]
        .drop("Loan_Approved")
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    colors_bar = ["#4caf50" if v > 0 else "#f44336" for v in target_corr.values]
    ax.bar(range(len(target_corr)), target_corr.values, color=colors_bar)
    ax.set_xticks(range(len(target_corr)))
    ax.set_xticklabels(target_corr.index, rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title("Feature correlation with Loan_Approved")
    ax.set_ylabel("Pearson r")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 · DATA
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Raw dataset")
    st.write(f"Shape: **{df_raw.shape[0]} rows × {df_raw.shape[1]} columns**")
    st.dataframe(df_raw.head(100), use_container_width=True)

    st.markdown("---")
    st.markdown("**Descriptive statistics**")
    st.dataframe(df_raw.describe(), use_container_width=True)

    st.markdown("**Missing values**")
    missing = df_raw.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        st.success("No missing values found.")
    else:
        st.dataframe(missing.rename("Missing count"), use_container_width=True)
