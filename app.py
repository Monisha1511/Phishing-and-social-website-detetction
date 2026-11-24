
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
import re
import ipaddress
import requests

from datetime import datetime
from urllib.parse import urlparse

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, silhouette_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

PROJECT_TITLE = "Phishing Website Detection using Machine Learning (XGBoost)"

st.set_page_config(
    page_title=PROJECT_TITLE,
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#e0f2f1,#004d40);
}
.main .block-container {
    background: rgba(255,255,255,0.97);
    padding: 2rem;
    border-radius: 18px;
}
.metric-card {
    padding: 20px;
    border-radius: 14px;
    color: white;
    text-align: center;
    background: linear-gradient(135deg,#26a69a,#00695c);
}
.metric-value {
    font-size: 30px;
    font-weight: bold;
}
.metric-label {
    font-size: 15px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    f"<h1 style='text-align:center;color:#004d40;'>"
    f"🛡️ {PROJECT_TITLE}</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;color:#00695c;'>"
    "Data Exploration • Model Training • Evaluation • "
    "Live Prediction • Website Intelligence • Future Forecast"
    "</p>",
    unsafe_allow_html=True
)

# ==========================================================
# 2. DATA LOADING
# ==========================================================

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

st.sidebar.header("📂 Dataset & Navigation")

uploaded_file = st.sidebar.file_uploader(
    "Upload Training Dataset",
    type=["csv"],
    key="training_dataset"
)

if uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    try:
        df = load_data("Phising_Training_Dataset.csv")
    except Exception:
        st.error(
            "Please keep Phising_Training_Dataset.csv "
            "in the same folder as app.py."
        )
        st.stop()

df.columns = df.columns.str.strip()

if "Result" not in df.columns:
    st.error("Dataset must contain Result column.")
    st.stop()

df["Result"] = pd.to_numeric(df["Result"], errors="coerce")
df = df.dropna(subset=["Result"]).copy()

# Original dataset labels:
# -1 = Legitimate
#  1 = Phishing

unique_values = set(df["Result"].unique())

if unique_values.issubset({-1, 1}):
    df["Result"] = df["Result"].map({
        -1: 0,
         1: 1
    })

if not set(df["Result"].unique()).issubset({0, 1}):
    st.error("Result must contain -1/1 or 0/1.")
    st.stop()

df["Result"] = df["Result"].astype(int)

df["Label"] = df["Result"].map({
    0: "Legitimate",
    1: "Phishing"
})

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

feature_cols = [
    c for c in numeric_cols
    if c not in ["Result", "key"]
]

if not feature_cols:
    st.error("No numeric features found.")
    st.stop()

df[feature_cols] = df[feature_cols].replace(
    [np.inf, -np.inf], np.nan
)

df[feature_cols] = df[feature_cols].fillna(
    df[feature_cols].median()
)

# ==========================================================
# 3. SIDEBAR NAVIGATION
# ==========================================================

pages = [
    "Dashboard",
    "Data Exploration",
    "Feature Engineering",
    "Model Training",
    "Performance Analysis",
    "Clustering",
    "Live Prediction",
    "Website Intelligence",
    "Related Domain Analysis",
    "Future Phishing Forecast",
    "Documentation",
    "About"
]

page = st.sidebar.radio("Go to", pages)

st.sidebar.divider()
st.sidebar.caption(PROJECT_TITLE)

# ==========================================================
# 4. COMMON FUNCTIONS
# ==========================================================

def get_importance():

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    )

    model.fit(df[feature_cols], df["Result"])

    return pd.DataFrame({
        "Feature": feature_cols,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)


def parse_url(url):

    url = url.strip()

    if not url:
        return None

    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url

    try:
        parsed = urlparse(url)

        if not parsed.hostname:
            return None

        return parsed

    except Exception:
        return None


def extract_url_features(url):

    parsed = parse_url(url)

    if parsed is None:
        return None

    host = parsed.hostname.lower()
    full_url = url.strip()

    try:
        ipaddress.ip_address(host)
        ip_value = 1
    except ValueError:
        ip_value = -1

    shorteners = [
        "bit.ly", "tinyurl.com", "t.co",
        "is.gd", "ow.ly", "cutt.ly"
    ]

    labels = host.split(".")
    subdomains = max(0, len(labels) - 2)

    values = {
        "having_IP": ip_value,
        "URL_Length": (
            1 if len(full_url) < 54
            else 0 if len(full_url) < 75
            else -1
        ),
        "Shortining_Service": (
            1 if any(x in host for x in shorteners) else -1
        ),
        "having_At_Symbol": 1 if "@" in full_url else -1,
        "double_slash_redirecting": (
            -1 if "//" in parsed.path else 1
        ),
        "Prefix_Suffix": -1 if "-" in host else 1,
        "having_Sub_Domain": (
            1 if subdomains == 0
            else 0 if subdomains == 1
            else -1
        ),
        "SSLfinal_State": 1 if parsed.scheme == "https" else -1,
        "HTTPS_token": -1 if "https" in host else 1
    }

    return parsed, values


def prepare_url_input(values):

    row = {
        c: float(df[c].median())
        for c in feature_cols
    }

    for key, value in values.items():
        if key in row:
            row[key] = float(value)

    return pd.DataFrame([row])[feature_cols]


# ==========================================================
# 5. DASHBOARD
# ==========================================================

if page == "Dashboard":

    st.subheader("📊 Dataset Overview")

    total = len(df)
    phishing = int((df["Result"] == 1).sum())
    legitimate = int((df["Result"] == 0).sum())

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>Total Websites</div>"
            f"<div class='metric-value'>{total:,}</div></div>",
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>Phishing Websites</div>"
            f"<div class='metric-value'>{phishing:,}</div></div>",
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>Legitimate Websites</div>"
            f"<div class='metric-value'>{legitimate:,}</div></div>",
            unsafe_allow_html=True
        )

    st.markdown("### Sample Records")
    st.dataframe(df.head(), use_container_width=True)

    fig = px.pie(
        names=["Legitimate", "Phishing"],
        values=[legitimate, phishing],
        hole=0.45,
        color_discrete_sequence=["#26a69a", "#ef5350"]
    )

    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# 6. DATA EXPLORATION
# ==========================================================

elif page == "Data Exploration":

    st.subheader("🔍 Data Exploration")

    selected = st.multiselect(
        "Select Features",
        feature_cols,
        default=feature_cols[:4]
    )

    for feature in selected:

        fig = px.histogram(
            df,
            x=feature,
            color="Label",
            barmode="overlay",
            title=f"Distribution of {feature}"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Correlation Heatmap")

    corr = df[feature_cols + ["Result"]].corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        aspect="auto"
    )

    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# 7. FEATURE ENGINEERING
# ==========================================================

elif page == "Feature Engineering":

    st.subheader("🧬 Feature Importance")

    with st.spinner("Analyzing features..."):
        importance = get_importance()

    st.dataframe(importance, use_container_width=True)

    fig = px.bar(
        importance.head(20),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 20 Important Features"
    )

    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# 8. MODEL TRAINING
# ==========================================================

elif page == "Model Training":

    st.subheader("🤖 Model Training")

    test_size = st.slider(
        "Test Size (%)", 10, 40, 25
    ) / 100

    scaler_choice = st.selectbox(
        "Scaling Method",
        ["StandardScaler", "MinMaxScaler", "None"]
    )

    if st.button("🚀 Train All Models"):

        X = df[feature_cols]
        y = df["Result"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=42
        )

        scaler = None

        if scaler_choice == "StandardScaler":
            scaler = StandardScaler()

        elif scaler_choice == "MinMaxScaler":
            scaler = MinMaxScaler()

        if scaler is not None:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values

        models = {
            "XGBoost": XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1
            ),

            "Random Forest": RandomForestClassifier(
                n_estimators=150,
                random_state=42,
                n_jobs=-1
            ),

            "Gradient Boosting": GradientBoostingClassifier(
                random_state=42
            ),

            "SVM": SVC(
                probability=True,
                random_state=42
            ),

            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                random_state=42
            ),

            "KNN": KNeighborsClassifier(),

            "Naive Bayes": GaussianNB(),

            "MLP": MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42
            )
        }

        results = {}
        trained_models = {}

        progress = st.progress(0)

        for i, (name, model) in enumerate(models.items()):

            model.fit(X_train_scaled, y_train)

            preds = model.predict(X_test_scaled)

            probs = model.predict_proba(X_test_scaled)[:, 1]

            results[name] = {
                "acc": accuracy_score(y_test, preds),
                "f1": f1_score(y_test, preds),
                "auc": roc_auc_score(y_test, probs),
                "preds": preds,
                "probs": probs
            }

            trained_models[name] = model

            progress.progress((i + 1) / len(models))

        best_name = max(
            results,
            key=lambda x: results[x]["auc"]
        )

        st.session_state.trained_models = trained_models
        st.session_state.results = results
        st.session_state.best_model_name = best_name
        st.session_state.scaler = scaler
        st.session_state.y_test = y_test

        st.success(f"Training completed! Best Model: {best_name}")

        result_df = pd.DataFrame(results).T[
            ["acc", "f1", "auc"]
        ].round(4)

        st.dataframe(
            result_df.sort_values("auc", ascending=False),
            use_container_width=True
        )

# ==========================================================
# 9. PERFORMANCE ANALYSIS
# ==========================================================

elif page == "Performance Analysis":

    st.subheader("📈 Performance Analysis")

    if "results" not in st.session_state:
        st.warning("Train models first.")
    else:

        results = st.session_state.results
        y_test = st.session_state.y_test

        tab1, tab2, tab3 = st.tabs([
            "Leaderboard",
            "ROC Curves",
            "Confusion Matrix"
        ])

        with tab1:

            result_df = pd.DataFrame(results).T[
                ["acc", "f1", "auc"]
            ].round(4)

            st.dataframe(
                result_df.sort_values("auc", ascending=False),
                use_container_width=True
            )

        with tab2:

            fig = go.Figure()

            for name, result in results.items():

                fpr, tpr, _ = roc_curve(
                    y_test,
                    result["probs"]
                )

                fig.add_trace(
                    go.Scatter(
                        x=fpr,
                        y=tpr,
                        mode="lines",
                        name=f"{name} AUC={result['auc']:.3f}"
                    )
                )

            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Random",
                    line=dict(dash="dash")
                )
            )

            fig.update_layout(
                title="ROC Curve Comparison",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate"
            )

            st.plotly_chart(fig, use_container_width=True)

        with tab3:

            selected_model = st.selectbox(
                "Select Model",
                list(results.keys())
            )

            cm = confusion_matrix(
                y_test,
                results[selected_model]["preds"],
                labels=[0, 1]
            )

            fig = px.imshow(
                cm,
                text_auto=True,
                x=["Predicted Legitimate", "Predicted Phishing"],
                y=["Actual Legitimate", "Actual Phishing"],
                color_continuous_scale="Blues"
            )

            st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# 10. CLUSTERING
# ==========================================================

elif page == "Clustering":

    st.subheader("🧩 K-Means Clustering")

    X = df[feature_cols].values

    if len(X) > 10000:
        rng = np.random.RandomState(42)
        indices = rng.choice(
            len(X), 10000, replace=False
        )
        X = X[indices]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(
        n_clusters=2,
        random_state=42,
        n_init=10
    )

    clusters = kmeans.fit_predict(X_scaled)

    fig = px.scatter(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        color=clusters.astype(str),
        labels={
            "x": "PC1",
            "y": "PC2",
            "color": "Cluster"
        },
        title="K-Means Clustering"
    )

    st.plotly_chart(fig, use_container_width=True)

    score = silhouette_score(X_scaled, clusters)

    st.metric("Silhouette Score", f"{score:.3f}")

# ==========================================================
# 11. LIVE PREDICTION
# ==========================================================

elif page == "Live Prediction":

    st.subheader("🔮 Live Prediction")

    if "trained_models" not in st.session_state:
        st.warning("Please train models first.")
    else:

        best_name = st.session_state.best_model_name
        model = st.session_state.trained_models[best_name]
        scaler = st.session_state.scaler

        st.write(f"Best Model: **{best_name}**")

        importance = get_importance()
        top_features = importance["Feature"].head(5).tolist()

        user_values = {}

        for feature in top_features:

            minimum = float(df[feature].min())
            maximum = float(df[feature].max())
            median = float(df[feature].median())

            step = (
                (maximum - minimum) / 100
                if maximum != minimum else 1.0
            )

            user_values[feature] = st.slider(
                feature,
                minimum,
                maximum,
                median,
                step=step,
                key=f"live_{feature}"
            )

        if st.button("⚡ Predict Website Type"):

            full_input = {
                feature: float(df[feature].median())
                for feature in feature_cols
            }

            full_input.update(user_values)

            input_df = pd.DataFrame(
                [full_input]
            )[feature_cols]

            model_input = (
                scaler.transform(input_df)
                if scaler is not None
                else input_df.values
            )

            pred = int(model.predict(model_input)[0])

            prob = float(
                model.predict_proba(model_input)[0][1]
            )

            label = "PHISHING" if pred == 1 else "LEGITIMATE"

            if pred == 1:
                st.error(f"⚠️ Prediction: {label}")
            else:
                st.success(f"✅ Prediction: {label}")

            st.metric(
                "Phishing Probability",
                f"{prob:.2%}"
            )

            st.download_button(
                "📥 Download Report",
                f"Prediction: {label}\nProbability: {prob:.2%}\nModel: {best_name}",
                file_name="prediction_report.txt"
            )

# ==========================================================
# 12. WEBSITE INTELLIGENCE
# ==========================================================

elif page == "Website Intelligence":

    st.subheader("🌐 Website Intelligence")

    website_url = st.text_input(
        "Enter Website URL",
        placeholder="https://example.com"
    )

    if st.button("🔍 Analyze Website"):

        result = extract_url_features(website_url)

        if result is None:
            st.error("Enter a valid URL.")
        else:

            parsed, values = result

            st.markdown("### Website Information")

            c1, c2, c3 = st.columns(3)

            c1.metric("Domain", parsed.hostname)
            c2.metric("Protocol", parsed.scheme.upper())
            c3.metric("URL Length", len(website_url))

            st.markdown("### Security Indicators")

            st.dataframe(
                pd.DataFrame([
                    {"Feature": k, "Value": v}
                    for k, v in values.items()
                ]),
                use_container_width=True
            )

            if "trained_models" not in st.session_state:

                st.warning("Train models first.")

            else:

                model = st.session_state.trained_models[
                    st.session_state.best_model_name
                ]

                scaler = st.session_state.scaler

                input_df = prepare_url_input(values)

                model_input = (
                    scaler.transform(input_df)
                    if scaler is not None
                    else input_df.values
                )

                pred = int(model.predict(model_input)[0])

                prob = float(
                    model.predict_proba(model_input)[0][1]
                )

                if pred == 1:
                    st.error("⚠️ Potential Phishing Website")
                else:
                    st.success("✅ Classified as Legitimate")

                st.metric(
                    "Estimated Phishing Probability",
                    f"{prob:.2%}"
                )

                st.info(
                    "This is an indicative prediction. Features "
                    "not extracted from the URL are filled using "
                    "training dataset medians. It is not a complete "
                    "live website security scan."
                )

# ==========================================================
# 13. RELATED DOMAIN ANALYSIS
# ==========================================================

elif page == "Related Domain Analysis":

    st.subheader("🧭 Related Domain Analysis")

    domain_input = st.text_input(
        "Enter Main Domain",
        placeholder="example.com"
    )

    if st.button("🔎 Find Related Domains"):

        parsed = parse_url(domain_input)

        if parsed is None:
            st.error("Enter a valid domain.")
        else:

            domain = parsed.hostname.lower()

            try:

                response = requests.get(
                    "https://crt.sh/",
                    params={
                        "q": "%." + domain,
                        "output": "json"
                    },
                    timeout=20
                )

                response.raise_for_status()

                records = response.json()
                related = set()

                for record in records:

                    names = record.get(
                        "name_value", ""
                    ).splitlines()

                    for name in names:

                        name = name.strip().lower().lstrip("*.")

                        if (
                            name.endswith("." + domain)
                            and name != domain
                        ):
                            related.add(name)

                if related:

                    st.success(
                        f"Found {len(related)} related domain candidates."
                    )

                    st.dataframe(
                        pd.DataFrame({
                            "Related Domain": sorted(related)
                        }),
                        use_container_width=True
                    )

                    st.info(
                        "Related certificate records do not prove "
                        "that a domain is malicious."
                    )

                else:
                    st.info("No related domains found.")

            except Exception as e:

                st.error(f"Unable to retrieve domain data: {e}")

# ==========================================================
# 14. FUTURE PHISHING FORECAST - SINGLE CSV UPLOAD
# ==========================================================

elif page == "Future Phishing Forecast":

    st.subheader("📅 Future Phishing Forecast")

    st.write(
        "Upload ONE historical CSV file to estimate "
        "the phishing-report trend for the next 7 days."
    )

    st.info(
        "Required CSV columns: Date and Result. "
        "You can use another date column name by selecting it below."
    )

    # IMPORTANT: One uploader only.
    # Do not use accept_multiple_files=True.

    forecast_file = st.file_uploader(
        "Upload ONE Historical CSV File",
        type=["csv"],
        accept_multiple_files=False,
        key="single_forecast_csv"
    )

    if forecast_file is not None:

        try:

            history = pd.read_csv(forecast_file)

            history.columns = history.columns.str.strip()

            if history.empty:
                st.error("The uploaded CSV is empty.")
                st.stop()

            st.success("CSV uploaded successfully!")

            st.markdown("### Uploaded Data Preview")

            st.dataframe(
                history.head(10),
                use_container_width=True
            )

            # Select date column

            date_candidates = [
                c for c in history.columns
                if "date" in c.lower() or "time" in c.lower()
            ]

            default_date_index = (
                history.columns.get_loc(date_candidates[0])
                if date_candidates else 0
            )

            date_column = st.selectbox(
                "Select Date Column",
                history.columns,
                index=default_date_index,
                key="forecast_date_column"
            )

            # Select label column

            label_candidates = [
                c for c in history.columns
                if c.lower() in [
                    "result", "label", "class",
                    "status", "phishing"
                ]
            ]

            default_label_index = (
                history.columns.get_loc(label_candidates[0])
                if label_candidates else 0
            )

            label_column = st.selectbox(
                "Select Phishing Label / Count Column",
                history.columns,
                index=default_label_index,
                key="forecast_label_column"
            )

            if st.button(
                "📈 Generate 7-Day Forecast",
                key="generate_forecast"
            ):

                work = history[
                    [date_column, label_column]
                ].copy()

                work[date_column] = pd.to_datetime(
                    work[date_column],
                    errors="coerce"
                )

                work = work.dropna(
                    subset=[date_column]
                )

                if work.empty:

                    st.error(
                        "No valid dates found. "
                        "Use date format YYYY-MM-DD."
                    )

                else:

                    labels = work[label_column]

                    # Convert labels to daily phishing counts

                    if pd.api.types.is_numeric_dtype(labels):

                        numeric_labels = pd.to_numeric(
                            labels,
                            errors="coerce"
                        )

                        unique_labels = set(
                            numeric_labels.dropna().unique()
                        )

                        if unique_labels.issubset({-1, 1}):

                            # Forecast CSV uses original dataset labels:
                            # -1 = Legitimate, 1 = Phishing

                            work["phishing_count"] = (
                                numeric_labels == 1
                            ).astype(int)

                        elif unique_labels.issubset({0, 1}):

                            # 0 = Legitimate, 1 = Phishing

                            work["phishing_count"] = (
                                numeric_labels == 1
                            ).astype(int)

                        else:

                            # Treat values as already aggregated counts

                            work["phishing_count"] = (
                                numeric_labels.clip(lower=0)
                            )

                    else:

                        work["phishing_count"] = (
                            labels.astype(str)
                            .str.lower()
                            .isin([
                                "phishing",
                                "malicious",
                                "1",
                                "true"
                            ])
                            .astype(int)
                        )

                    daily = (
                        work.groupby(
                            work[date_column].dt.date
                        )["phishing_count"]
                        .sum()
                        .sort_index()
                    )

                    daily.index = pd.to_datetime(daily.index)

                    if len(daily) < 3:

                        st.error(
                            "At least 3 different dates are required "
                            "to generate a forecast."
                        )

                    else:

                        # Linear trend estimation

                        x = np.arange(len(daily))

                        slope, intercept = np.polyfit(
                            x,
                            daily.values,
                            1
                        )

                        future_x = np.arange(
                            len(daily),
                            len(daily) + 7
                        )

                        forecast_values = np.maximum(
                            0,
                            slope * future_x + intercept
                        )

                        future_dates = pd.date_range(
                            daily.index.max()
                            + pd.Timedelta(days=1),
                            periods=7
                        )

                        forecast_df = pd.DataFrame({
                            "Date": future_dates,
                            "Estimated Phishing Reports": np.round(
                                forecast_values, 1
                            )
                        })

                        st.success("Forecast generated successfully!")

                        st.markdown("### Historical Phishing Trend")

                        historical_df = daily.reset_index()
                        historical_df.columns = [
                            "Date", "Phishing Reports"
                        ]

                        fig1 = px.line(
                            historical_df,
                            x="Date",
                            y="Phishing Reports",
                            markers=True,
                            title="Historical Phishing Reports"
                        )

                        st.plotly_chart(
                            fig1,
                            use_container_width=True
                        )

                        st.markdown("### Next 7 Days Forecast")

                        fig2 = px.line(
                            forecast_df,
                            x="Date",
                            y="Estimated Phishing Reports",
                            markers=True,
                            title="Estimated Future Phishing Trend"
                        )

                        st.plotly_chart(
                            fig2,
                            use_container_width=True
                        )

                        st.dataframe(
                            forecast_df,
                            use_container_width=True
                        )

                        st.download_button(
                            "📥 Download Forecast CSV",
                            data=forecast_df.to_csv(
                                index=False
                            ).encode("utf-8"),
                            file_name="future_phishing_forecast.csv",
                            mime="text/csv"
                        )

                        st.warning(
                            "This is a basic trend-based estimate "
                            "using historical reports. It does not "
                            "identify exact future phishing URLs "
                            "or guarantee future attacks."
                        )

        except pd.errors.EmptyDataError:

            st.error("The uploaded CSV contains no data.")

        except Exception as e:

            st.error(
                f"Unable to process CSV: {e}"
            )

# ==========================================================
# 15. DOCUMENTATION
# ==========================================================

elif page == "Documentation":

    st.subheader("📑 Project Documentation")

    st.markdown("""
### Project Title
Phishing Website Detection using Machine Learning (XGBoost)

### Problem Statement
Phishing websites imitate legitimate websites to steal
sensitive user information.

### Proposed Solution
Machine learning identifies patterns in website features
and classifies records as phishing or legitimate.

### Modules

1. Dashboard
2. Data Exploration
3. Feature Engineering
4. Model Training
5. Performance Analysis
6. Clustering
7. Live Prediction
8. Website Intelligence
9. Related Domain Analysis
10. Future Phishing Forecast

### Algorithms

- XGBoost
- Random Forest
- Gradient Boosting
- SVM
- Logistic Regression
- KNN
- Naive Bayes
- MLP

### Technologies

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Plotly

### Evaluation Metrics

- Accuracy
- F1-score
- AUC
- ROC Curve
- Confusion Matrix
""")

# ==========================================================
# 16. ABOUT
# ==========================================================

elif page == "About":

    st.subheader("ℹ️ About the Project")

    st.markdown("""
### Phishing Website Detection using Machine Learning (XGBoost)

This project is a machine-learning-based application
that classifies website records into phishing
and legitimate categories.

### Objectives

- Detect phishing website patterns.
- Classify website records using ML.
- Compare multiple ML algorithms.
- Analyze important website features.
- Perform URL-based indicative prediction.
- Identify related domain candidates.
- Estimate future phishing-report trends.

### Project Output

- Phishing / Legitimate classification
- Model performance comparison
- Website feature analysis
- Related domain information
- Future phishing trend estimation

**Note:** Model predictions are decision-support indicators,
not a guarantee of website safety.
""")

st.sidebar.divider()
st.sidebar.caption("Machine Learning Project")
