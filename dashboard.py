"""
Credit Risk ML Pipeline Dashboard

Run with: streamlit run dashboard.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# Page configuration
st.set_page_config(
    page_title="Credit Risk ML Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
FEATURE_STORE_DIR = PROJECT_ROOT / "feature_store" / "data" / "features"


@st.cache_data
def load_model_metadata():
    """Load the latest model metadata."""
    metadata_files = list(MODELS_DIR.glob("*_metadata.json"))
    if not metadata_files:
        return None
    latest = max(metadata_files, key=lambda x: x.stat().st_mtime)
    with open(latest) as f:
        return json.load(f)


@st.cache_data
def load_feature_importance():
    """Load feature importance data."""
    importance_files = list(MODELS_DIR.glob("*_feature_importance.csv"))
    if not importance_files:
        return None
    latest = max(importance_files, key=lambda x: x.stat().st_mtime)
    return pd.read_csv(latest)


@st.cache_data
def load_predictions():
    """Load prediction results."""
    pred_dir = DATA_DIR / "predictions"
    pred_files = list(pred_dir.glob("predictions_*.csv"))
    if not pred_files:
        return None
    latest = max(pred_files, key=lambda x: x.stat().st_mtime)
    return pd.read_csv(latest)


@st.cache_data
def load_training_data():
    """Load training features."""
    feature_file = DATA_DIR / "features" / "train_features.parquet"
    if not feature_file.exists():
        return None
    return pd.read_parquet(feature_file)


@st.cache_data
def load_raw_data():
    """Load raw training data for target distribution."""
    raw_file = DATA_DIR / "raw" / "application_train.csv"
    if not raw_file.exists():
        return None
    return pd.read_csv(raw_file)


def create_gauge_chart(value, title, max_val=1.0):
    """Create a gauge chart for metrics."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 16}},
        number={"font": {"size": 24}, "valueformat": ".3f"},
        gauge={
            "axis": {"range": [0, max_val], "tickwidth": 1},
            "bar": {"color": "#1f77b4"},
            "steps": [
                {"range": [0, max_val * 0.5], "color": "#ffcccb"},
                {"range": [max_val * 0.5, max_val * 0.7], "color": "#ffffcc"},
                {"range": [max_val * 0.7, max_val], "color": "#ccffcc"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": value,
            },
        },
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def main():
    # Header
    st.title("Credit Risk ML Pipeline Dashboard")
    st.markdown("---")

    # Load data
    metadata = load_model_metadata()
    importance_df = load_feature_importance()
    predictions_df = load_predictions()
    training_df = load_training_data()
    raw_df = load_raw_data()

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Overview", "Model Performance", "Feature Analysis", "Predictions", "Data Explorer"]
    )

    if page == "Overview":
        show_overview(metadata, predictions_df, training_df, raw_df)
    elif page == "Model Performance":
        show_model_performance(metadata)
    elif page == "Feature Analysis":
        show_feature_analysis(importance_df, training_df)
    elif page == "Predictions":
        show_predictions(predictions_df)
    elif page == "Data Explorer":
        show_data_explorer(training_df, raw_df)


def show_overview(metadata, predictions_df, training_df, raw_df):
    """Show overview page."""
    st.header("Pipeline Overview")

    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Training Samples",
            value=f"{len(training_df):,}" if training_df is not None else "N/A",
        )

    with col2:
        st.metric(
            label="Features Used",
            value=len(metadata["feature_columns"]) if metadata else "N/A",
        )

    with col3:
        st.metric(
            label="ROC-AUC Score",
            value=f"{metadata['metrics']['roc_auc']:.4f}" if metadata else "N/A",
        )

    with col4:
        st.metric(
            label="Predictions Made",
            value=f"{len(predictions_df):,}" if predictions_df is not None else "N/A",
        )

    st.markdown("---")

    # Two column layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Target Distribution (Training Data)")
        if raw_df is not None and "TARGET" in raw_df.columns:
            target_counts = raw_df["TARGET"].value_counts().reset_index()
            target_counts.columns = ["Target", "Count"]
            target_counts["Target"] = target_counts["Target"].map({0: "No Default", 1: "Default"})

            fig = px.pie(
                target_counts,
                values="Count",
                names="Target",
                color="Target",
                color_discrete_map={"No Default": "#2ecc71", "Default": "#e74c3c"},
                hole=0.4,
            )
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

            default_rate = raw_df["TARGET"].mean() * 100
            st.info(f"Default Rate: {default_rate:.2f}%")
        else:
            st.warning("Training data not available")

    with col2:
        st.subheader("Model Information")
        if metadata:
            info_data = {
                "Property": [
                    "Model Type",
                    "Training Date",
                    "Number of Features",
                    "Estimators",
                    "Max Depth",
                ],
                "Value": [
                    metadata["model_type"].replace("_", " ").title(),
                    metadata["training_date"][:19].replace("T", " "),
                    len(metadata["feature_columns"]),
                    metadata["model_params"].get("n_estimators", "N/A"),
                    metadata["model_params"].get("max_depth", "N/A"),
                ],
            }
            st.table(pd.DataFrame(info_data))
        else:
            st.warning("Model metadata not available")

    st.markdown("---")

    # Pipeline status
    st.subheader("Pipeline Components Status")

    components = [
        ("Raw Data", DATA_DIR / "raw", "application_train.csv"),
        ("Processed Features", DATA_DIR / "features", "train_features.parquet"),
        ("Feature Store", FEATURE_STORE_DIR, "credit_features.parquet"),
        ("Trained Model", MODELS_DIR, "*.joblib"),
        ("Predictions", DATA_DIR / "predictions", "predictions_*.csv"),
    ]

    cols = st.columns(len(components))
    for col, (name, path, pattern) in zip(cols, components):
        with col:
            if path.exists() and list(path.glob(pattern)):
                st.success(f"{name}")
            else:
                st.error(f"{name}")


def show_model_performance(metadata):
    """Show model performance page."""
    st.header("Model Performance Metrics")

    if not metadata:
        st.warning("Model metadata not available")
        return

    metrics = metadata["metrics"]

    # Gauge charts for main metrics
    st.subheader("Classification Metrics")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.plotly_chart(
            create_gauge_chart(metrics["roc_auc"], "ROC-AUC"),
            use_container_width=True,
        )

    with col2:
        st.plotly_chart(
            create_gauge_chart(metrics["accuracy"], "Accuracy"),
            use_container_width=True,
        )

    with col3:
        st.plotly_chart(
            create_gauge_chart(metrics["precision"], "Precision"),
            use_container_width=True,
        )

    with col4:
        st.plotly_chart(
            create_gauge_chart(metrics["recall"], "Recall"),
            use_container_width=True,
        )

    with col5:
        st.plotly_chart(
            create_gauge_chart(metrics["f1_score"], "F1-Score"),
            use_container_width=True,
        )

    st.markdown("---")

    # Metrics explanation
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Metrics Summary")
        metrics_df = pd.DataFrame({
            "Metric": list(metrics.keys()),
            "Value": [f"{v:.4f}" for v in metrics.values()],
        })
        st.table(metrics_df)

    with col2:
        st.subheader("Model Parameters")
        params = metadata["model_params"]
        params_df = pd.DataFrame({
            "Parameter": list(params.keys()),
            "Value": [str(v) for v in params.values()],
        })
        st.table(params_df)

    st.markdown("---")

    # Metrics interpretation
    st.subheader("Metrics Interpretation")

    roc_auc = metrics["roc_auc"]
    if roc_auc >= 0.8:
        st.success(f"ROC-AUC of {roc_auc:.4f} indicates good discriminative ability")
    elif roc_auc >= 0.7:
        st.info(f"ROC-AUC of {roc_auc:.4f} indicates acceptable discriminative ability")
    else:
        st.warning(f"ROC-AUC of {roc_auc:.4f} indicates room for improvement")

    recall = metrics["recall"]
    st.info(
        f"Recall of {recall:.4f} means the model identifies {recall*100:.1f}% of actual defaults"
    )


def show_feature_analysis(importance_df, training_df):
    """Show feature analysis page."""
    st.header("Feature Analysis")

    if importance_df is None:
        st.warning("Feature importance data not available")
        return

    # Top features bar chart
    st.subheader("Top 20 Most Important Features")

    top_20 = importance_df.head(20).copy()
    top_20 = top_20.sort_values("importance", ascending=True)

    fig = px.bar(
        top_20,
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        height=500,
        xaxis_title="Importance",
        yaxis_title="Feature",
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Feature importance distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Importance Distribution")
        fig = px.histogram(
            importance_df,
            x="importance",
            nbins=30,
            color_discrete_sequence=["#1f77b4"],
        )
        fig.update_layout(
            xaxis_title="Importance",
            yaxis_title="Count",
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Cumulative Importance")
        importance_df_sorted = importance_df.sort_values("importance", ascending=False)
        importance_df_sorted["cumulative"] = importance_df_sorted["importance"].cumsum()
        importance_df_sorted["cumulative_pct"] = (
            importance_df_sorted["cumulative"] / importance_df_sorted["importance"].sum() * 100
        )

        fig = px.line(
            importance_df_sorted.head(30),
            x=range(1, 31),
            y="cumulative_pct",
            markers=True,
        )
        fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="80%")
        fig.update_layout(
            xaxis_title="Number of Features",
            yaxis_title="Cumulative Importance (%)",
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Feature statistics
    if training_df is not None:
        st.subheader("Feature Statistics")

        # Select top features for display
        top_features = importance_df.head(10)["feature"].tolist()
        available_features = [f for f in top_features if f in training_df.columns]

        if available_features:
            stats_df = training_df[available_features].describe().T
            stats_df = stats_df.round(2)
            st.dataframe(stats_df, use_container_width=True)


def show_predictions(predictions_df):
    """Show predictions page."""
    st.header("Prediction Results")

    if predictions_df is None:
        st.warning("Prediction data not available")
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Predictions", f"{len(predictions_df):,}")

    with col2:
        default_rate = predictions_df["prediction"].mean() * 100
        st.metric("Predicted Default Rate", f"{default_rate:.2f}%")

    with col3:
        avg_prob = predictions_df["probability"].mean() * 100
        st.metric("Avg Default Probability", f"{avg_prob:.2f}%")

    with col4:
        high_risk = (predictions_df["risk_score"] > 500).sum()
        st.metric("High Risk Applications", f"{high_risk:,}")

    st.markdown("---")

    # Probability distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Probability Distribution")
        fig = px.histogram(
            predictions_df,
            x="probability",
            nbins=50,
            color_discrete_sequence=["#3498db"],
        )
        fig.update_layout(
            xaxis_title="Default Probability",
            yaxis_title="Count",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Risk Score Distribution")
        fig = px.histogram(
            predictions_df,
            x="risk_score",
            nbins=50,
            color_discrete_sequence=["#e74c3c"],
        )
        fig.update_layout(
            xaxis_title="Risk Score (0-1000)",
            yaxis_title="Count",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Risk category breakdown
    st.subheader("Risk Category Breakdown")

    predictions_df["risk_category"] = pd.cut(
        predictions_df["risk_score"],
        bins=[0, 200, 400, 600, 800, 1000],
        labels=["Very Low", "Low", "Medium", "High", "Very High"],
    )

    risk_counts = predictions_df["risk_category"].value_counts().reset_index()
    risk_counts.columns = ["Risk Category", "Count"]

    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(
            risk_counts,
            values="Count",
            names="Risk Category",
            color="Risk Category",
            color_discrete_map={
                "Very Low": "#27ae60",
                "Low": "#2ecc71",
                "Medium": "#f39c12",
                "High": "#e67e22",
                "Very High": "#e74c3c",
            },
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            risk_counts.sort_values("Risk Category"),
            x="Risk Category",
            y="Count",
            color="Risk Category",
            color_discrete_map={
                "Very Low": "#27ae60",
                "Low": "#2ecc71",
                "Medium": "#f39c12",
                "High": "#e67e22",
                "Very High": "#e74c3c",
            },
        )
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Sample predictions
    st.subheader("Sample Predictions")
    st.dataframe(
        predictions_df.head(20)[["SK_ID_CURR", "prediction", "probability", "risk_score"]],
        use_container_width=True,
    )


def show_data_explorer(training_df, raw_df):
    """Show data explorer page."""
    st.header("Data Explorer")

    if training_df is None:
        st.warning("Training data not available")
        return

    # Dataset info
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", f"{len(training_df):,}")

    with col2:
        st.metric("Columns", f"{len(training_df.columns):,}")

    with col3:
        memory = training_df.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("Memory Usage", f"{memory:.2f} MB")

    st.markdown("---")

    # Column selector
    st.subheader("Feature Distribution")

    numeric_cols = training_df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ["SK_ID_CURR", "TARGET", "event_timestamp"]
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    selected_feature = st.selectbox("Select Feature", feature_cols[:50])

    if selected_feature:
        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(
                training_df,
                x=selected_feature,
                nbins=50,
                color_discrete_sequence=["#3498db"],
            )
            fig.update_layout(
                title=f"Distribution of {selected_feature}",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.box(
                training_df,
                y=selected_feature,
                color_discrete_sequence=["#3498db"],
            )
            fig.update_layout(
                title=f"Box Plot of {selected_feature}",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Statistics
        stats = training_df[selected_feature].describe()
        st.write("**Statistics:**")
        st.dataframe(stats.to_frame().T, use_container_width=True)

    st.markdown("---")

    # Correlation with target
    if raw_df is not None and "TARGET" in raw_df.columns:
        st.subheader("Feature Correlation with Target")

        # Calculate correlations
        numeric_raw = raw_df.select_dtypes(include=[np.number])
        correlations = numeric_raw.corr()["TARGET"].drop("TARGET").sort_values(key=abs, ascending=False)

        top_corr = correlations.head(15)

        fig = px.bar(
            x=top_corr.values,
            y=top_corr.index,
            orientation="h",
            color=top_corr.values,
            color_continuous_scale="RdBu_r",
            color_continuous_midpoint=0,
        )
        fig.update_layout(
            title="Top 15 Features Correlated with Default",
            xaxis_title="Correlation",
            yaxis_title="Feature",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
