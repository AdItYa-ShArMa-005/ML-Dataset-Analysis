import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc,
    mean_absolute_error, mean_squared_error, r2_score
)

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

st.title("Dataset Analysis Dashboard")

# --- 2. DATA PERSISTENCE ---
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None

with st.sidebar:
    st.header("Step 1: Definition")
    problem_type = st.radio("Problem Type", ["Classification", "Regression"])
    uploaded_file = st.file_uploader("Upload CSV", type="csv")

    if uploaded_file:
        if st.session_state.df_raw is None:
            st.session_state.df_raw = pd.read_csv(uploaded_file, sep=";")
        if st.button("Reset Pipeline"):
            st.session_state.df_raw = pd.read_csv(uploaded_file, sep=";")
            for k in ["trained_model", "trained_model_name", "X_test_perf", "y_test_perf", "label_enc_target"]:
                st.session_state.pop(k, None)
            st.rerun()

# --- 3. GLOBAL PROCESSING ---
if st.session_state.df_raw is not None:
    df = st.session_state.df_raw.copy()

    df_enc = df.copy()
    label_encoders = {}
    for col in df_enc.columns:
        if not pd.api.types.is_numeric_dtype(df_enc[col]):
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
            label_encoders[col] = le

    df_enc_clean = df_enc.dropna()
    df_orig_clean = df.loc[df_enc_clean.index]
    target_options = list(df_enc_clean.columns)

    tabs = st.tabs([
        "Data & PCA", "Visual EDA", "Engineering",
        "Selection", "Training", "Tuning", "Performance"
    ])

    # --- TAB 1: DATA & PCA ---
    with tabs[0]:
        st.subheader("Dataset Geometry & PCA")

        target_col = st.selectbox(
            "Select Target (Y)", target_options,
            key="target_col",
            index=target_options.index(st.session_state.get("target_col", target_options[-1]))
        )

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
            color_labels = df_orig_clean[target_col].astype(str)

            fig_pca = px.scatter(
                x=pca_res[:, 0], y=pca_res[:, 1],
                color=color_labels,
                labels={"x": "PC1", "y": "PC2", "color": target_col},
                title="2D PCA Projection", template="plotly_dark", height=600
            )
            st.plotly_chart(fig_pca, use_container_width=True)

            pca_full = PCA().fit(pca_input)
            explained = np.cumsum(pca_full.explained_variance_ratio_) * 100
            fig_var = px.line(
                x=list(range(1, len(explained) + 1)), y=explained,
                labels={"x": "Number of Components", "y": "Cumulative Explained Variance (%)"},
                title="Explained Variance by Components", template="plotly_dark", markers=True
            )
            fig_var.add_hline(y=95, line_dash="dash", line_color="red", annotation_text="95% threshold")
            st.plotly_chart(fig_var, use_container_width=True)
        else:
            st.warning("Please select at least 2 features for PCA.")

    def get_globals():
        tc = st.session_state.get("target_col", target_options[-1])
        X = st.session_state.get("X_global", df_enc_clean.drop(columns=[tc]))
        y = st.session_state.get("y_global", df_enc_clean[tc])
        return tc, X, y

    # --- TAB 2: VISUAL EDA ---
    with tabs[1]:
        st.subheader("Exploratory Data Analysis")
        st.markdown("### Raw Data Preview")
        st.dataframe(df.head(15), use_container_width=True)

        target_col, X_global, y_global = get_globals()
        st.markdown("### Class Distribution")
        fig_dist = px.histogram(
            df_orig_clean, x=target_col,
            title=f"Distribution of {target_col}",
            template="plotly_dark", color=target_col
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        st.markdown("### Correlation Heatmap")
        fig_corr = px.imshow(
            df_enc_clean.corr(), text_auto=".2f", aspect="auto",
            color_continuous_scale='Viridis', height=700
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # --- TAB 3: DATA ENGINEERING ---
    with tabs[2]:
        st.subheader("Cleaning & Outliers")
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### Imputation")
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
            st.markdown("#### Outlier Detection")
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
                mi_df, orientation='h', title="Mutual Information Scores",
                labels={"value": "MI Score", "index": "Feature"}, template="plotly_dark"
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
            test_size = st.slider("Test Split Size", 0.1, 0.4, 0.2, 0.05,
                                  help="Fraction held out for the Performance tab")

        if st.button("Train Model"):
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_global)

            if m_choice == "Random Forest":
                m = RandomForestClassifier(random_state=42) if problem_type == "Classification" else RandomForestRegressor(random_state=42)
            elif m_choice == "KNN":
                m = KNeighborsClassifier(n_neighbors=params['k']) if problem_type == "Classification" else KNeighborsRegressor(n_neighbors=params['k'])
            elif m_choice == "SVM":
                # probability=True enables predict_proba for ROC curve
                m = SVC(kernel=params['kernel'], probability=True, random_state=42) if problem_type == "Classification" else SVR(kernel=params['kernel'])
            elif m_choice == "Decision Tree":
                m = DecisionTreeClassifier(max_depth=params['depth'], random_state=42) if problem_type == "Classification" else DecisionTreeRegressor(max_depth=params['depth'], random_state=42)
            else:
                m = LogisticRegression(max_iter=1000) if problem_type == "Classification" else LinearRegression()

            # Cross-validation
            res = cross_validate(m, X_scaled, y_global, cv=k_fold_val, return_train_score=True)
            st.divider()
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("Mean Validation Score", f"{res['test_score'].mean():.4f}")
            col_m2.metric("Mean Training Score", f"{res['train_score'].mean():.4f}")

            fold_df = pd.DataFrame({
                "Fold": list(range(1, k_fold_val + 1)) * 2,
                "Score": list(res['test_score']) + list(res['train_score']),
                "Set": ["Validation"] * k_fold_val + ["Train"] * k_fold_val
            })
            fig_fold = px.line(fold_df, x="Fold", y="Score", color="Set",
                               title="Per-Fold Scores", template="plotly_dark", markers=True)
            st.plotly_chart(fig_fold, use_container_width=True)

            # Final model fit on train split — saved for Performance tab
            strat = y_global if problem_type == "Classification" else None
            X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y_global, test_size=test_size,
                                                        random_state=42, stratify=strat)
            m.fit(X_tr, y_tr)
            st.session_state["trained_model"]      = m
            st.session_state["trained_model_name"] = m_choice
            st.session_state["X_test_perf"]        = X_te
            st.session_state["y_test_perf"]        = y_te
            st.session_state["label_enc_target"]   = label_encoders.get(target_col, None)
            st.success("Model trained! Head to the Performance tab for detailed metrics.")

    # --- TAB 6: TUNING ---
    with tabs[5]:
        st.subheader("Hyperparameter Tuning")
        target_col, X_global, y_global = get_globals()
        if st.button("Run Grid Search"):
            X_tune = StandardScaler().fit_transform(X_global)
            grid = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
            base = RandomForestClassifier(random_state=42) if problem_type == "Classification" else RandomForestRegressor(random_state=42)
            gs = GridSearchCV(base, grid, cv=3)
            with st.spinner("Tuning..."):
                gs.fit(X_tune, y_global)
            st.success(f"Best Parameters: {gs.best_params_}")
            st.metric("Optimized Score", f"{gs.best_score_:.4f}")

            results_df = pd.DataFrame(gs.cv_results_)
            pivot = results_df.pivot_table(values='mean_test_score',
                                           index='param_max_depth', columns='param_n_estimators')
            fig_gs = px.imshow(pivot, text_auto=".4f", title="Grid Search Scores",
                               color_continuous_scale="Viridis", template="plotly_dark")
            st.plotly_chart(fig_gs, use_container_width=True)

    # =========================================================
    # --- TAB 7: PERFORMANCE ---
    # =========================================================
    with tabs[6]:
        st.subheader("Model Performance")

        model      = st.session_state.get("trained_model", None)
        X_te       = st.session_state.get("X_test_perf", None)
        y_te       = st.session_state.get("y_test_perf", None)
        le_target  = st.session_state.get("label_enc_target", None)
        model_name = st.session_state.get("trained_model_name", "Model")

        if model is None or X_te is None:
            st.info("Train a model in the **Training** tab first, then return here for full performance analysis.")
        else:
            y_pred = model.predict(X_te)
            st.caption(f"Evaluating: **{model_name}** on **{len(y_te)}** test samples")

            # ==================================================
            # CLASSIFICATION
            # ==================================================
            if problem_type == "Classification":
                classes       = le_target.classes_ if le_target is not None else np.unique(y_te).astype(str)
                y_te_labels   = le_target.inverse_transform(y_te)   if le_target is not None else y_te.astype(str)
                y_pred_labels = le_target.inverse_transform(y_pred) if le_target is not None else y_pred.astype(str)

                report_dict = classification_report(y_te_labels, y_pred_labels, output_dict=True)
                accuracy    = report_dict["accuracy"]
                macro_f1    = report_dict["macro avg"]["f1-score"]
                macro_prec  = report_dict["macro avg"]["precision"]
                macro_rec   = report_dict["macro avg"]["recall"]

                # ── Metric cards ──
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Accuracy",        f"{accuracy:.4f}")
                m2.metric("Macro F1",        f"{macro_f1:.4f}")
                m3.metric("Macro Precision", f"{macro_prec:.4f}")
                m4.metric("Macro Recall",    f"{macro_rec:.4f}")

                st.divider()
                col_left, col_right = st.columns(2)

                # ── Confusion Matrix ──
                with col_left:
                    st.markdown("#### Confusion Matrix")
                    cm = confusion_matrix(y_te_labels, y_pred_labels, labels=classes)
                    fig_cm = px.imshow(
                        cm, x=list(classes), y=list(classes),
                        text_auto=True, color_continuous_scale="Blues",
                        labels={"x": "Predicted", "y": "Actual"},
                        title="Confusion Matrix", template="plotly_dark"
                    )
                    fig_cm.update_layout(height=420)
                    st.plotly_chart(fig_cm, use_container_width=True)

                # ── Per-class bar chart ──
                with col_right:
                    st.markdown("#### Per-Class Metrics")
                    rows = []
                    for cls in classes:
                        key = str(cls)
                        if key in report_dict:
                            rows.append({
                                "Class":     key,
                                "Precision": report_dict[key]["precision"],
                                "Recall":    report_dict[key]["recall"],
                                "F1-Score":  report_dict[key]["f1-score"]
                            })
                    per_class_df = pd.DataFrame(rows).melt(id_vars="Class", var_name="Metric", value_name="Score")
                    fig_pc = px.bar(
                        per_class_df, x="Class", y="Score", color="Metric",
                        barmode="group", title="Precision / Recall / F1 per Class",
                        template="plotly_dark"
                    )
                    fig_pc.update_layout(height=420)
                    st.plotly_chart(fig_pc, use_container_width=True)

                # ── ROC Curve ──
                st.markdown("#### ROC Curve")
                if hasattr(model, "predict_proba"):
                    y_score  = model.predict_proba(X_te)
                    n_cls    = len(classes)
                    fig_roc  = go.Figure()
                    fig_roc.update_layout(
                        template="plotly_dark", title="ROC Curve",
                        xaxis_title="False Positive Rate",
                        yaxis_title="True Positive Rate", height=450
                    )
                    if n_cls == 2:
                        fpr, tpr, _ = roc_curve((y_te == model.classes_[1]).astype(int), y_score[:, 1])
                        roc_auc = auc(fpr, tpr)
                        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                                     name=f"AUC = {roc_auc:.3f}",
                                                     line=dict(width=2)))
                    else:
                        for i, cls in enumerate(model.classes_):
                            cls_label = le_target.inverse_transform([cls])[0] if le_target else str(cls)
                            fpr, tpr, _ = roc_curve((y_te == cls).astype(int), y_score[:, i])
                            roc_auc = auc(fpr, tpr)
                            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                                         name=f"{cls_label} (AUC={roc_auc:.3f})",
                                                         line=dict(width=2)))
                    fig_roc.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1], mode='lines',
                        line=dict(dash='dash', color='gray'), name="Random Baseline"
                    ))
                    st.plotly_chart(fig_roc, use_container_width=True)
                else:
                    st.info("ROC Curve not available — this model does not support predict_proba.")

                # ── Full report table ──
                st.markdown("#### Full Classification Report")
                report_rows = []
                for cls in classes:
                    key = str(cls)
                    if key in report_dict:
                        report_rows.append({
                            "Class":     key,
                            "Precision": f"{report_dict[key]['precision']:.4f}",
                            "Recall":    f"{report_dict[key]['recall']:.4f}",
                            "F1-Score":  f"{report_dict[key]['f1-score']:.4f}",
                            "Support":   int(report_dict[key]['support'])
                        })
                st.dataframe(pd.DataFrame(report_rows), use_container_width=True)

            # ==================================================
            # REGRESSION
            # ==================================================
            else:
                mae  = mean_absolute_error(y_te, y_pred)
                mse  = mean_squared_error(y_te, y_pred)
                rmse = np.sqrt(mse)
                r2   = r2_score(y_te, y_pred)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("R² Score", f"{r2:.4f}")
                m2.metric("MAE",      f"{mae:.4f}")
                m3.metric("MSE",      f"{mse:.4f}")
                m4.metric("RMSE",     f"{rmse:.4f}")

                st.divider()
                col_left, col_right = st.columns(2)

                # ── Actual vs Predicted ──
                with col_left:
                    st.markdown("#### Actual vs Predicted")
                    fig_avp = px.scatter(
                        x=y_te, y=y_pred, opacity=0.6,
                        labels={"x": "Actual", "y": "Predicted"},
                        title="Actual vs Predicted", template="plotly_dark"
                    )
                    lo = min(float(y_te.min()), float(y_pred.min()))
                    hi = max(float(y_te.max()), float(y_pred.max()))
                    fig_avp.add_trace(go.Scatter(
                        x=[lo, hi], y=[lo, hi], mode='lines',
                        line=dict(dash='dash', color='red'), name="Perfect Fit"
                    ))
                    st.plotly_chart(fig_avp, use_container_width=True)

                # ── Residuals vs Predicted ──
                with col_right:
                    st.markdown("#### Residuals Plot")
                    residuals = y_te - y_pred
                    fig_res = px.scatter(
                        x=y_pred, y=residuals, opacity=0.6,
                        labels={"x": "Predicted", "y": "Residuals"},
                        title="Residuals vs Predicted", template="plotly_dark"
                    )
                    fig_res.add_hline(y=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_res, use_container_width=True)

                # ── Residuals Distribution ──
                st.markdown("#### Residuals Distribution")
                fig_hist = px.histogram(
                    x=residuals, nbins=50,
                    labels={"x": "Residual", "y": "Count"},
                    title="Distribution of Residuals", template="plotly_dark"
                )
                st.plotly_chart(fig_hist, use_container_width=True)

else:
    st.info("Upload a CSV file from the sidebar to get started.")