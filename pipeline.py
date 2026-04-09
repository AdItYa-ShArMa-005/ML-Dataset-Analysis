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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score

# --- UI ---
st.set_page_config(page_title="Dataset Analysis Dashboard", layout="wide")

st.title("🛡️ Dataset Analysis Dashboard")

# --- SESSION ---
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None

# --- SIDEBAR ---
with st.sidebar:
    st.header("📍 Step 1: Definition")
    problem_type = st.radio("Problem Type", ["Classification", "Regression"])
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    
    if uploaded_file:
        if st.session_state.df_raw is None:
            st.session_state.df_raw = pd.read_csv(uploaded_file)
        if st.button("🗑️ Reset Pipeline"):
            st.session_state.df_raw = pd.read_csv(uploaded_file)
            st.rerun()

# --- MAIN ---
if st.session_state.df_raw is not None:
    df = st.session_state.df_raw.copy()

    # Encoding
    df_enc = df.copy()
    for col in df_enc.columns:
        if not pd.api.types.is_numeric_dtype(df_enc[col]):
            df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))

    df_final = df_enc.dropna()
    target_options = list(df_final.columns)

    tabs = st.tabs([
        "📊 Data & PCA", "📈 Visual EDA", "🛠️ Engineering",
        "🎯 Selection", "🤖 Training", "📂 Data Split",
        "📉 Performance", "🚀 Tuning"
    ])

    # --- TAB 1 ---
    with tabs[0]:
        st.subheader("Dataset Geometry & PCA")
        target_col = st.selectbox("Target", target_options)
        feat_cols = st.multiselect("Features", [c for c in df_final.columns if c != target_col])

        if len(feat_cols) >= 2:
            pca_input = StandardScaler().fit_transform(df_final[feat_cols])
            pca_res = PCA(n_components=2).fit_transform(pca_input)
            fig = px.scatter(x=pca_res[:,0], y=pca_res[:,1],
                             color=df.loc[df_final.index, target_col].astype(str))
            st.plotly_chart(fig, use_container_width=True)

    # --- TAB 2 ---
    with tabs[1]:
        st.dataframe(df.head())
        fig_corr = px.imshow(df_final.corr())
        st.plotly_chart(fig_corr)

    # --- TAB 3 ---
    with tabs[2]:
        st.subheader("Data Cleaning")

        if st.button("Fill NA with Mean"):
            st.session_state.df_raw = st.session_state.df_raw.fillna(
                st.session_state.df_raw.mean(numeric_only=True)
            )
            st.rerun()

    # --- TAB 4 ---
    with tabs[3]:
        X_fs = df_final.drop(columns=[target_col])
        y_fs = df_final[target_col]

        if st.button("Feature Importance"):
            scores = mutual_info_classif(X_fs, y_fs) if problem_type == "Classification" else mutual_info_regression(X_fs, y_fs)
            st.bar_chart(pd.Series(scores, index=X_fs.columns))

    # --- TAB 5 TRAINING ---
    with tabs[4]:
        st.subheader("Training")

        if st.button("Train (CV)"):
            X = StandardScaler().fit_transform(df_final.drop(columns=[target_col]))
            y = df_final[target_col]

            model = RandomForestClassifier() if problem_type == "Classification" else RandomForestRegressor()
            res = cross_validate(model, X, y, cv=5)

            st.metric("Validation Score", res['test_score'].mean())

    # --- TAB 6 DATA SPLIT ---
    with tabs[5]:
        st.subheader("Train-Test Split")

        test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
        if st.button("Split Data"):
            X = df_final.drop(columns=[target_col])
            y = df_final[target_col]

            X = StandardScaler().fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test

            st.success("Data Split Done")

    # --- TAB 7 PERFORMANCE ---
    with tabs[6]:
        st.subheader("Performance Metrics")

        if 'X_train' in st.session_state:

            if st.button("Evaluate"):

                model = RandomForestClassifier() if problem_type == "Classification" else RandomForestRegressor()

                model.fit(st.session_state.X_train, st.session_state.y_train)
                y_pred = model.predict(st.session_state.X_test)

                if problem_type == "Classification":
                    st.metric("Accuracy", accuracy_score(st.session_state.y_test, y_pred))
                    st.metric("Precision", precision_score(st.session_state.y_test, y_pred, average='weighted'))
                    st.metric("Recall", recall_score(st.session_state.y_test, y_pred, average='weighted'))
                    st.metric("F1", f1_score(st.session_state.y_test, y_pred, average='weighted'))
                else:
                    st.metric("MSE", mean_squared_error(st.session_state.y_test, y_pred))
                    st.metric("R2", r2_score(st.session_state.y_test, y_pred))

        else:
            st.warning("Split data first!")

    # --- TAB 8 TUNING ---
    with tabs[7]:
        if st.button("Grid Search"):
            X = StandardScaler().fit_transform(df_final.drop(columns=[target_col]))
            y = df_final[target_col]

            model = RandomForestClassifier() if problem_type == "Classification" else RandomForestRegressor()
            grid = {'n_estimators':[50,100]}

            gs = GridSearchCV(model, grid, cv=3)
            gs.fit(X, y)

            st.write(gs.best_params_)
            st.metric("Best Score", gs.best_score_)

else:
    st.info("Upload dataset to start")