import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

# --- 1. UI ESTHETICS ---
st.set_page_config(page_title="Dataset Analysis Dashboard", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; background-color: #161b22; padding: 15px; border-radius: 15px; }
    .stTabs [data-baseweb="tab"] { height: 50px; border-radius: 8px; background-color: #21262d; color: #8b949e; font-weight: bold; }
    .stTabs [aria-selected="true"] { background-color: #238636 !important; color: white !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Dataset Analysis Dashboard")

# --- 2. DATA PERSISTENCE ---
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None

with st.sidebar:
    st.header("📍 Step 1: Definition")
    problem_type = st.radio("Problem Type", ["Classification", "Regression"])
    uploaded_file = st.file_uploader("Upload CSV", type="csv")

    if uploaded_file:
        if st.session_state.df_raw is None:
            st.session_state.df_raw = pd.read_csv(uploaded_file, sep=";")
        if st.button("🗑️ Reset Pipeline"):
            st.session_state.df_raw = pd.read_csv(uploaded_file, sep=";")
            st.rerun()

# --- 3. GLOBAL PROCESSING ---
if st.session_state.df_raw is not None:
    df = st.session_state.df_raw.copy()

    # Global Encoding for Math (keep original df for display)
    df_enc = df.copy()
    label_encoders = {}
    for col in df_enc.columns:
        if not pd.api.types.is_numeric_dtype(df_enc[col]):
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
            label_encoders[col] = le

    # Drop NaNs globally; keep aligned original df for display
    df_enc_clean = df_enc.dropna()
    df_orig_clean = df.loc[df_enc_clean.index]  # FIX: aligned original (with labels) for display
    target_options = list(df_enc_clean.columns)

    tabs = st.tabs(["📊 Data & PCA", "📈 Visual EDA", "🛠️ Engineering", "🎯 Selection", "🤖 Training", "🚀 Tuning"])

    # --- TAB 1: DATA & PCA ---
    with tabs[0]:
        st.subheader("Dataset Geometry & PCA")

        # Target selection lives here (original behaviour), but stored in session_state so other tabs can access it
        target_col = st.selectbox("Select Target (Y)", target_options,
                                  key="target_col",
                                  index=target_options.index(st.session_state.get("target_col", target_options[-1])))

        # Shared X, y — derived after target is chosen, stored for other tabs
        X_global = df_enc_clean.drop(columns=[target_col])
        y_global = df_enc_clean[target_col]
        st.session_state["X_global"] = X_global
        st.session_state["y_global"] = y_global

        feat_cols = st.multiselect(
            "Features for PCA",
            [c for c in df_enc_clean.columns if c != target_col],
            default=[c for c in df_enc_clean.columns if c != target_col][:5]
        )

        if len(feat_cols) >= 2:
            pca_input = StandardScaler().fit_transform(df_enc_clean[feat_cols])
            pca_res = PCA(n_components=2).fit_transform(pca_input)

            # FIX: Use ORIGINAL (unencoded) target column values for legend labels
            color_labels = df_orig_clean[target_col].astype(str)

            fig_pca = px.scatter(
                x=pca_res[:, 0],
                y=pca_res[:, 1],
                color=color_labels,
                labels={"x": "PC1", "y": "PC2", "color": target_col},  # FIX: proper axis & legend labels
                title="2D PCA Projection",
                template="plotly_dark",
                height=600
            )
            st.plotly_chart(fig_pca, use_container_width=True)

            # Bonus: Explained variance
            pca_full = PCA().fit(pca_input)
            explained = np.cumsum(pca_full.explained_variance_ratio_) * 100
            fig_var = px.line(
                x=list(range(1, len(explained) + 1)),
                y=explained,
                labels={"x": "Number of Components", "y": "Cumulative Explained Variance (%)"},
                title="Explained Variance by Components",
                template="plotly_dark",
                markers=True
            )
            fig_var.add_hline(y=95, line_dash="dash", line_color="red", annotation_text="95% threshold")
            st.plotly_chart(fig_var, use_container_width=True)
        else:
            st.warning("Please select at least 2 features for PCA.")

    # Helper: other tabs read from session_state (set in Tab 1)
    def get_globals():
        tc = st.session_state.get("target_col", target_options[-1])
        X = st.session_state.get("X_global", df_enc_clean.drop(columns=[tc]))
        y = st.session_state.get("y_global", df_enc_clean[tc])
        return tc, X, y

    # --- TAB 2: VISUAL EDA ---
    with tabs[1]:
        st.subheader("Exploratory Data Analysis")
        st.markdown("### 📋 Raw Data Preview")
        st.dataframe(df.head(15), use_container_width=True)

        target_col, X_global, y_global = get_globals()
        st.markdown("### 📊 Class Distribution")
        fig_dist = px.histogram(
            df_orig_clean,
            x=target_col,
            title=f"Distribution of {target_col}",
            template="plotly_dark",
            color=target_col
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        st.markdown("### 🗺️ Correlation Heatmap")
        fig_corr = px.imshow(
            df_enc_clean.corr(),
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale='Viridis',
            height=700
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # --- TAB 3: DATA ENGINEERING ---
    with tabs[2]:
        st.subheader("Cleaning & Outliers")
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### 🩹 Imputation")
            imp_choice = st.selectbox("Select Imputation Method", ["Mean", "Median", "Zero", "Drop NaNs"])
            if st.button("Apply Imputation"):
                if imp_choice == "Mean":
                    st.session_state.df_raw = st.session_state.df_raw.fillna(st.session_state.df_raw.mean(numeric_only=True))
                elif imp_choice == "Median":
                    st.session_state.df_raw = st.session_state.df_raw.fillna(st.session_state.df_raw.median(numeric_only=True))
                elif imp_choice == "Zero":
                    st.session_state.df_raw = st.session_state.df_raw.fillna(0)
                elif imp_choice == "Drop NaNs":
                    st.session_state.df_raw = st.session_state.df_raw.dropna()
                st.toast(f"Imputation ({imp_choice}) successful!", icon="✅")
                st.rerun()

        with col_b:
            st.markdown("#### 🕵️ Outlier Detection")
            out_alg = st.selectbox("Detection Algorithm", ["IQR", "Isolation Forest", "DBSCAN", "OPTICS"])
            if st.button("Detect & Remove Outliers"):
                num_only = st.session_state.df_raw.select_dtypes(include=[np.number]).dropna()
                idx_to_drop = []

                if out_alg == "IQR":
                    q1, q3 = num_only.quantile(0.25), num_only.quantile(0.75)
                    iqr = q3 - q1
                    idx_to_drop = num_only[((num_only < (q1 - 1.5 * iqr)) | (num_only > (q3 + 1.5 * iqr))).any(axis=1)].index
                elif out_alg == "Isolation Forest":
                    idx_to_drop = num_only.index[IsolationForest(random_state=42).fit_predict(num_only) == -1]
                elif out_alg in ["DBSCAN", "OPTICS"]:
                    scaled_out = StandardScaler().fit_transform(num_only)
                    clusterer = DBSCAN() if out_alg == "DBSCAN" else OPTICS()
                    idx_to_drop = num_only.index[clusterer.fit_predict(scaled_out) == -1]

                st.session_state.df_raw = st.session_state.df_raw.drop(idx_to_drop)
                st.success(f"Successfully removed {len(idx_to_drop)} outliers using {out_alg}!")
                st.rerun()

    # --- TAB 4: FEATURE SELECTION ---
    with tabs[3]:
        st.subheader("Feature Importance Analysis")
        target_col, X_global, y_global = get_globals()
        if st.button("Calculate Feature Importance"):
            scores = (
                mutual_info_classif(X_global, y_global)
                if problem_type == "Classification"
                else mutual_info_regression(X_global, y_global)
            )
            mi_df = pd.Series(scores, index=X_global.columns).sort_values(ascending=True)
            fig_fs = px.bar(
                mi_df,
                orientation='h',
                title="Mutual Information Scores",
                labels={"value": "MI Score", "index": "Feature"},
                template="plotly_dark"
            )
            st.plotly_chart(fig_fs, use_container_width=True)

    # --- TAB 5: TRAINING ---
    with tabs[4]:
        st.subheader("Model Configuration")
        target_col, X_global, y_global = get_globals()
        m_choice = st.selectbox("Select Model", ["Random Forest", "KNN", "SVM", "Decision Tree", "Linear/Logistic"])

        c1, c2 = st.columns(2)
        params = {}
        with c1:
            if m_choice == "KNN":
                params['k'] = st.slider("Neighbors (K)", 1, 50, 5)
            elif m_choice == "SVM":
                params['kernel'] = st.selectbox("Kernel", ["rbf", "linear", "poly"])
            elif m_choice == "Decision Tree":
                params['depth'] = st.slider("Depth", 1, 30, 10)
        with c2:
            k_fold_val = st.number_input("K-Fold Value", 2, 10, 5)

        if st.button("🚀 Train Model"):
            X_scaled = StandardScaler().fit_transform(X_global)

            if m_choice == "Random Forest":
                m = RandomForestClassifier(random_state=42) if problem_type == "Classification" else RandomForestRegressor(random_state=42)
            elif m_choice == "KNN":
                m = KNeighborsClassifier(n_neighbors=params['k']) if problem_type == "Classification" else KNeighborsRegressor(n_neighbors=params['k'])
            elif m_choice == "SVM":
                m = SVC(kernel=params['kernel']) if problem_type == "Classification" else SVR(kernel=params['kernel'])
            elif m_choice == "Decision Tree":
                m = DecisionTreeClassifier(max_depth=params['depth'], random_state=42) if problem_type == "Classification" else DecisionTreeRegressor(max_depth=params['depth'], random_state=42)
            else:
                m = LogisticRegression(max_iter=1000) if problem_type == "Classification" else LinearRegression()

            res = cross_validate(m, X_scaled, y_global, cv=k_fold_val, return_train_score=True)
            st.divider()
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("Mean Validation Score", f"{res['test_score'].mean():.4f}")
            col_m2.metric("Mean Training Score", f"{res['train_score'].mean():.4f}")

            # Bonus: per-fold scores chart
            fold_df = pd.DataFrame({
                "Fold": list(range(1, k_fold_val + 1)) * 2,
                "Score": list(res['test_score']) + list(res['train_score']),
                "Set": ["Validation"] * k_fold_val + ["Train"] * k_fold_val
            })
            fig_fold = px.line(fold_df, x="Fold", y="Score", color="Set",
                               title="Per-Fold Scores", template="plotly_dark", markers=True)
            st.plotly_chart(fig_fold, use_container_width=True)

    # --- TAB 6: TUNING ---
    with tabs[5]:
        st.subheader("Hyperparameter Tuning")
        target_col, X_global, y_global = get_globals()
        if st.button("🏁 Run Grid Search"):
            X_tune = StandardScaler().fit_transform(X_global)
            y_tune = y_global

            grid = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
            base = RandomForestClassifier(random_state=42) if problem_type == "Classification" else RandomForestRegressor(random_state=42)
            gs = GridSearchCV(base, grid, cv=3)
            with st.spinner("Tuning..."):
                gs.fit(X_tune, y_tune)
            st.success(f"Best Parameters: {gs.best_params_}")
            st.metric("Optimized Score", f"{gs.best_score_:.4f}")

            # Bonus: Grid search results heatmap
            results_df = pd.DataFrame(gs.cv_results_)
            pivot = results_df.pivot_table(
                values='mean_test_score',
                index='param_max_depth',
                columns='param_n_estimators'
            )
            fig_gs = px.imshow(pivot, text_auto=".4f", title="Grid Search Scores",
                               color_continuous_scale="Viridis", template="plotly_dark")
            st.plotly_chart(fig_gs, use_container_width=True)
else:
    st.info("⬆️ Upload a CSV file from the sidebar to get started.")