import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.api import OLS, add_constant
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from lime.lime_tabular import LimeTabularExplainer
import shap
import lightgbm as lgb
import xgboost as xgb
from sklearn.svm import SVR

# ------------------
# Page Configuration (must be the first Streamlit command)
# ------------------
st.set_page_config(
    page_title="Macroeconomic & Financial Forecasting Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon=":chart_with_upwards_trend:"
)

# ------------------
# Custom CSS Styling
# ------------------
custom_css = """
<style>
/* General styling */
body {
    font-family: 'Segoe UI', sans-serif;
}
h1, h2, h3, h4, h5, h6 {
    color: #2c3e50;
}
.stButton button {
    background-color: #2c3e50;
    color: white;
    border-radius: 5px;
    border: none;
    padding: 0.5em 1em;
    font-weight: bold;
}
.stAlert {
    background-color: #fefefe;
    border-left: 5px solid #2c3e50;
}
.sidebar .sidebar-content {
    background-image: linear-gradient(#2980b9, #6dd5fa);
    color: white;
}
/* Increase padding around sections */
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
}
/* Improve table styling */
table {
    font-size: 0.9rem;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ------------------
# Main Title
# ------------------
# st.title("Macroeconomic and Financial Forecasting Dashboard")

# ------------------
# Sidebar Configuration (single instance)
# ------------------
st.sidebar.image("logo.png", use_column_width=True)
st.sidebar.title("Macroeconomic & Financial Forecasting Dashboard")
selected_tab = st.sidebar.selectbox(
    "Select Tab",
    ["Upload Data", "Variable Selection", "Model Selection", "Results and Recommendations", 
     "What If Scenarios", "Scenario Forecast", "Stress Testing", "Stress Testing 2", "Stress Testing 3"],
    key="unique_tab_select"
)

st.sidebar.markdown("## Dashboard Summary")
st.sidebar.markdown(
    """
    **Upload Data:**  
    - Upload your Excel file and preview the monthly time series data.
    
    **Variable Selection:**  
    - Choose the target and predictor variables using automatic (model-based) or manual selection.  
    - Visualize feature importances (global & LIME analysis).

    **Model Selection:**  
    - Train and evaluate multiple models (e.g., SARIMAX, OLS, Random Forest, etc.) using RMSE and MAPE.  
    - Perform backtesting to compare forecast vs. actual data.

    **Results and Recommendations:**  
    - View detailed model outputs and get actionable recommendations based on the selected model.  
    - Explore use cases and backtesting results.

    **What If Scenarios:**  
    - Input custom values for independent variables to simulate forecasts and analyze potential outcomes.

    **Scenario Forecast:**  
    - Upload multiple scenario files to generate forecasts for each scenario and compare them with historical data.

    **Stress Testing:**  
    - Calculate stressed parameters (EAD, LGD, PD) and expected loss for risk assessment under adverse economic conditions.

    **Stress Testing 2 & 3:**  
    - Use the best model and scenario data to forecast losses and receive detailed risk insights.
    """
)

# ------------------
# Session State Initialization
# ------------------
if "data" not in st.session_state:
    st.session_state["data"] = None
if "valid_models" not in st.session_state:
    st.session_state["valid_models"] = {}
if "selected_features" not in st.session_state:
    st.session_state["selected_features"] = None
if "target_var" not in st.session_state:
    st.session_state["target_var"] = None

# ------------------
# Helper Functions
# ------------------
def calculate_feature_importance(model, X, y):
    if isinstance(model, (RandomForestRegressor, GradientBoostingRegressor)):
        model.fit(X, y)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        importances = np.abs(shap_values).mean(axis=0)
        return importances
    elif isinstance(model, OLS):
        X_const = add_constant(X)
        model_fit = model.fit(X_const, y)
        importances = np.abs(model_fit.params[1:])
        return importances
    return None

# ------------------
# Main Application Layout
# ------------------
# st.title(":chart_with_upwards_trend: Time Series Modeling and Analysis")

# ==================
# UPLOAD DATA TAB
# ==================
if selected_tab == "Upload Data":
    st.header("Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload an Excel file", type="xlsx")
    if uploaded_file:
        st.session_state["data"] = pd.read_excel(uploaded_file)
        st.session_state["data"].set_index('date', inplace=True)
        st.session_state["data"].index = pd.to_datetime(st.session_state["data"].index)
        st.session_state["data"] = st.session_state["data"].asfreq('MS')
        st.success("File uploaded successfully!")
        st.write("Data Preview:")
        st.dataframe(st.session_state["data"])

# =========================
# VARIABLE SELECTION TAB
# =========================
elif selected_tab == "Variable Selection":
    st.header("Variable Selection")
    if st.session_state["data"] is not None:
        target_var = st.selectbox("Select Target Variable", st.session_state["data"].columns)
        independent_vars = [col for col in st.session_state["data"].columns if col != target_var]
        X = st.session_state["data"][independent_vars]
        y = st.session_state["data"][target_var]

        # Compute feature importances using various models
        feature_importances = {}
        try:
            rf_model = RandomForestRegressor(random_state=42)
            feature_importances["Random Forest"] = calculate_feature_importance(rf_model, X, y)
        except Exception as e:
            st.error(f"Error in Random Forest Feature Importance: {str(e)}")

        try:
            gb_model = GradientBoostingRegressor(random_state=42)
            feature_importances["Gradient Boosting"] = calculate_feature_importance(gb_model, X, y)
        except Exception as e:
            st.error(f"Error in Gradient Boosting Feature Importance: {str(e)}")

        try:
            X_const = add_constant(X)
            ols_model = OLS(y, X_const).fit()
            feature_importances["OLS"] = np.abs(ols_model.params[1:]).values
        except Exception as e:
            st.error(f"Error in OLS Feature Importance: {str(e)}")

        try:
            from sklearn.linear_model import Lasso
            lasso_model = Lasso(alpha=0.01, random_state=42)
            lasso_model.fit(X, y)
            feature_importances["LASSO"] = np.abs(lasso_model.coef_)
        except Exception as e:
            st.error(f"Error in LASSO Feature Importance: {str(e)}")

        try:
            from sklearn.linear_model import Ridge
            ridge_model = Ridge(alpha=0.01, random_state=42)
            ridge_model.fit(X, y)
            feature_importances["Ridge"] = np.abs(ridge_model.coef_)
        except Exception as e:
            st.error(f"Error in Ridge Regression Feature Importance: {str(e)}")

        try:
            from sklearn.linear_model import ElasticNet
            elasticnet_model = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42)
            elasticnet_model.fit(X, y)
            feature_importances["Elastic Net"] = np.abs(elasticnet_model.coef_)
        except Exception as e:
            st.error(f"Error in Elastic Net Feature Importance: {str(e)}")

        try:
            shap.initjs()
            rf_model_shap = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=42)
            rf_model_shap.fit(X, y)
            explainer = shap.TreeExplainer(rf_model_shap)
            shap_values = explainer.shap_values(X)
            feature_importances["SHAP"] = np.abs(shap_values).mean(axis=0)
        except Exception as e:
            st.error(f"Error in SHAP Feature Importance: {str(e)}")

        try:
            from sklearn.feature_selection import mutual_info_regression
            mutual_info = mutual_info_regression(X, y, random_state=42)
            feature_importances["Mutual Information"] = mutual_info
        except Exception as e:
            st.error(f"Error in Mutual Information Feature Importance: {str(e)}")

        # st.subheader("Global Feature Importance by Model")
        try:
            model_specific_features = {}
            for model_name, importances in feature_importances.items():
                sorted_idx = np.argsort(importances)[::-1]
                model_specific_features[model_name] = [independent_vars[i] for i in sorted_idx]

            st.subheader("Feature Selection Mode")
            selection_mode = st.radio("Choose Feature Selection Mode", options=["Automatic (Model-Based)", "Manual"],
                                      index=0, horizontal=True, key="feature_selection_mode")
            selected_top_features = []
            sorted_importances = []

            if selection_mode == "Automatic (Model-Based)":
                selected_model = st.selectbox("Select Model for Feature Selection", list(model_specific_features.keys()),
                                              key="automatic_model_select")
                if selected_model in ["LASSO", "Ridge", "Elastic Net"]:
                    if selected_model == "LASSO":
                        lasso_alpha = st.slider("LASSO Alpha", min_value=0.0001, max_value=1.0, value=0.01, step=0.0001, key="lasso_alpha")
                        try:
                            from sklearn.linear_model import Lasso
                            lasso_model = Lasso(alpha=lasso_alpha, random_state=42)
                            lasso_model.fit(X, y)
                            feature_importances["LASSO"] = np.abs(lasso_model.coef_)
                            sorted_idx = np.argsort(feature_importances["LASSO"])[::-1]
                            model_specific_features["LASSO"] = [independent_vars[i] for i in sorted_idx]
                        except Exception as e:
                            st.error(f"Error in LASSO Feature Importance with parameter: {str(e)}")
                    elif selected_model == "Ridge":
                        ridge_alpha = st.slider("Ridge Alpha", min_value=0.0001, max_value=1.0, value=0.01, step=0.0001, key="ridge_alpha")
                        try:
                            from sklearn.linear_model import Ridge
                            ridge_model = Ridge(alpha=ridge_alpha, random_state=42)
                            ridge_model.fit(X, y)
                            feature_importances["Ridge"] = np.abs(ridge_model.coef_)
                            sorted_idx = np.argsort(feature_importances["Ridge"])[::-1]
                            model_specific_features["Ridge"] = [independent_vars[i] for i in sorted_idx]
                        except Exception as e:
                            st.error(f"Error in Ridge Regression Feature Importance with parameter: {str(e)}")
                    elif selected_model == "Elastic Net":
                        elasticnet_alpha = st.slider("ElasticNet Alpha", min_value=0.0001, max_value=1.0, value=0.01, step=0.0001, key="elasticnet_alpha")
                        elasticnet_l1_ratio = st.slider("ElasticNet L1 Ratio", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="elasticnet_l1_ratio")
                        try:
                            from sklearn.linear_model import ElasticNet
                            elasticnet_model = ElasticNet(alpha=elasticnet_alpha, l1_ratio=elasticnet_l1_ratio, random_state=42)
                            elasticnet_model.fit(X, y)
                            feature_importances["Elastic Net"] = np.abs(elasticnet_model.coef_)
                            sorted_idx = np.argsort(feature_importances["Elastic Net"])[::-1]
                            model_specific_features["Elastic Net"] = [independent_vars[i] for i in sorted_idx]
                        except Exception as e:
                            st.error(f"Error in Elastic Net Feature Importance with parameter: {str(e)}")

                model_top_features = model_specific_features[selected_model]
                num_features = st.slider(f"Select Number of Top Features from {selected_model}",
                                         min_value=1, max_value=min(len(model_top_features), 100),
                                         value=10, key="automatic_num_features")
                selected_top_features = model_top_features[:num_features]
                sorted_importances = np.array(feature_importances[selected_model])[np.argsort(feature_importances[selected_model])[::-1]][:num_features]
                st.session_state["selected_features"] = selected_top_features
                st.session_state["target_var"] = target_var
                st.success(f"Top {num_features} Features Selected from '{selected_model}': {selected_top_features}")

            elif selection_mode == "Manual":
                st.subheader("Manual Feature Selection")
                selected_top_features = st.multiselect("Select Variables You Believe Are Important:", independent_vars,
                                                       default=[], key="manual_selected_features")
                sorted_importances = []
                for feature in selected_top_features:
                    if feature in independent_vars:
                        feature_idx = independent_vars.index(feature)
                        sorted_importances.append(np.mean([feature_importances[m][feature_idx] for m in feature_importances.keys()]))
                    else:
                        sorted_importances.append(0)
                st.session_state["selected_features"] = selected_top_features
                st.session_state["target_var"] = target_var
                st.success(f"Manually Selected Features: {selected_top_features}")

            if selected_top_features:
                importance_df = pd.DataFrame({"Feature": selected_top_features, "Importance": sorted_importances}).sort_values(by="Importance", ascending=True)
                plt.figure(figsize=(10, 6))
                plt.barh(importance_df["Feature"], importance_df["Importance"], color=plt.cm.viridis(np.linspace(0, 1, len(importance_df))))
                plt.title(f"Feature Importances ({'Manual' if selection_mode == 'Manual' else selected_model})", fontsize=14, fontweight="bold", color="darkblue")
                plt.xlabel("Importance", fontsize=12, color="darkred")
                plt.ylabel("Features", fontsize=12, color="darkred")
                plt.xticks(fontsize=10)
                plt.yticks(fontsize=10)
                plt.grid(axis="x", linestyle="--", alpha=0.7)
                st.pyplot(plt)
            else:
                st.info("No features selected for plotting.")
        except Exception as e:
            st.error(f"Error in feature selection or plotting: {str(e)}")

        st.subheader("LIME Feature Importance")
        try:
            selected_features = st.session_state["selected_features"]
            lime_explainer = LimeTabularExplainer(training_data=X.values, feature_names=X.columns.tolist(), mode="regression")
            available_dates = X.index.to_list()
            selected_date = st.selectbox("Select a Date for Explanation", available_dates)
            selected_instance = X.loc[selected_date]
            st.write(f"**Explaining Instance for Date:** {selected_date}")
            st.write(selected_instance)
            lime_exp = lime_explainer.explain_instance(selected_instance.values, rf_model.predict, num_features=len(selected_features))
            lime_importances = dict(lime_exp.as_list())
            importance_df = pd.DataFrame({"Feature": list(lime_importances.keys()), "Importance": list(lime_importances.values())}).sort_values(by="Importance", ascending=False)
            st.write("**LIME Feature Importances**")
            importance_df.index = [selected_date] * len(importance_df)
            st.dataframe(importance_df)
            plt.figure(figsize=(10, 6))
            plt.barh(importance_df["Feature"][:50][::-1], importance_df["Importance"][:50][::-1], color="skyblue")
            plt.title(f"LIME Feature Importances for {selected_date}")
            plt.xlabel("Importance")
            plt.ylabel("Features")
            st.pyplot(plt)
            st.subheader("Interpretation of LIME Results")
            interpreted_features = set()
            for _, row in importance_df.iterrows():
                if row["Feature"] not in interpreted_features:
                    if row["Importance"] > 0:
                        st.write(f"- **{row['Feature']}**: Positively contributes by {row['Importance']:.2f}")
                    elif row["Importance"] < 0:
                        st.write(f"- **{row['Feature']}**: Negatively contributes by {row['Importance']:.2f}")
                    else:
                        st.write(f"- **{row['Feature']}**: No significant contribution")
                    interpreted_features.add(row["Feature"])
        except Exception as e:
            st.error(f"Error in LIME Feature Importance: {str(e)}")
    else:
        st.warning("Please upload data first.")



# ==================
# MODEL SELECTION TAB
# ==================

elif selected_tab == "Model Selection":
    st.title("🔍 Model Selection and Evaluation")
    if st.session_state["selected_features"] is not None and st.session_state["target_var"]:
        st.markdown(f"**Target Variable:** `{st.session_state['target_var']}`")
        st.markdown(f"**Selected Features:** `{', '.join(st.session_state['selected_features'])}`")
        data = st.session_state["data"]
        X = data[st.session_state["selected_features"]]
        y = data[st.session_state["target_var"]]

        st.subheader("Adjust Training and Testing Sample Split")
        train_size_ratio = st.slider("Select Training Sample Percentage (Remaining for Testing):", 
                                     min_value=50, max_value=90, value=70, step=5) / 100.0
        train_size = int(len(data) * train_size_ratio)
        test_size = len(data) - train_size
        st.write(f"Training Samples: {train_size}, Testing Samples: {test_size}")

        # Split the data
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        
        # --- SCALE THE PREDICTOR DATA ---
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        # Note: We keep y_train and y_test in their original scale

        metrics = {}
        col1, col2 = st.columns(2)
        with col1:
            try:
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                sarimax_model = SARIMAX(y_train, exog=X_train_scaled, order=(1, 1, 1),
                                        seasonal_order=(1, 1, 1, 12)).fit(disp=False)
                sarimax_forecast = sarimax_model.forecast(steps=len(y_test), exog=X_test_scaled)
                metrics["SARIMAX"] = {"RMSE": np.sqrt(mean_squared_error(y_test, sarimax_forecast)),
                                      "MAPE": mean_absolute_percentage_error(y_test, sarimax_forecast)}
            except Exception as e:
                metrics["SARIMAX"] = {"Error": str(e)}
                st.error(f"Error in SARIMAX: {str(e)}")

            try:
                from statsmodels.api import OLS, add_constant
                X_train_const = add_constant(X_train_scaled)
                ols_model = OLS(y_train, X_train_const).fit()
                X_test_const = add_constant(X_test_scaled)
                ols_forecast = ols_model.predict(X_test_const)
                metrics["OLS"] = {"RMSE": np.sqrt(mean_squared_error(y_test, ols_forecast)),
                                  "MAPE": mean_absolute_percentage_error(y_test, ols_forecast)}
            except Exception as e:
                metrics["OLS"] = {"Error": str(e)}
                st.error(f"Error in OLS: {str(e)}")
        with col2:
            try:
                from sklearn.ensemble import RandomForestRegressor
                rf_model = RandomForestRegressor(n_estimators=200, max_depth=10,
                                                 min_samples_leaf=5, random_state=42)
                rf_model.fit(X_train_scaled, y_train)
                rf_forecast = rf_model.predict(X_test_scaled)
                metrics["Random Forest"] = {"RMSE": np.sqrt(mean_squared_error(y_test, rf_forecast)),
                                            "MAPE": mean_absolute_percentage_error(y_test, rf_forecast)}
            except Exception as e:
                st.error(f"Error in Random Forest: {str(e)}")

            try:
                from sklearn.ensemble import GradientBoostingRegressor
                gb_model = GradientBoostingRegressor(random_state=42)
                gb_model.fit(X_train_scaled, y_train)
                gb_forecast = gb_model.predict(X_test_scaled)
                metrics["Gradient Boosting"] = {"RMSE": np.sqrt(mean_squared_error(y_test, gb_forecast)),
                                                "MAPE": mean_absolute_percentage_error(y_test, gb_forecast)}
            except Exception as e:
                st.error(f"Error in Gradient Boosting: {str(e)}")

        try:
            import xgboost as xgb
            xgb_model = xgb.XGBRegressor(random_state=42)
            xgb_model.fit(X_train_scaled, y_train)
            xgb_forecast = xgb_model.predict(X_test_scaled)
            metrics["XGBoost"] = {"RMSE": np.sqrt(mean_squared_error(y_test, xgb_forecast)),
                                  "MAPE": mean_absolute_percentage_error(y_test, xgb_forecast)}
        except Exception as e:
            st.error(f"Error in XGBoost: {str(e)}")

        try:
            import lightgbm as lgb
            lgb_model = lgb.LGBMRegressor(random_state=42)
            lgb_model.fit(X_train_scaled, y_train)
            lgb_forecast = lgb_model.predict(X_test_scaled)
            metrics["LightGBM"] = {"RMSE": np.sqrt(mean_squared_error(y_test, lgb_forecast)),
                                   "MAPE": mean_absolute_percentage_error(y_test, lgb_forecast)}
        except Exception as e:
            st.error(f"Error in LightGBM: {str(e)}")

        try:
            from sklearn.svm import SVR
            svr_model = SVR()
            svr_model.fit(X_train_scaled, y_train)
            svr_forecast = svr_model.predict(X_test_scaled)
            metrics["SVR"] = {"RMSE": np.sqrt(mean_squared_error(y_test, svr_forecast)),
                              "MAPE": mean_absolute_percentage_error(y_test, svr_forecast)}
        except Exception as e:
            st.error(f"Error in SVR: {str(e)}")

        try:
            from sklearn.linear_model import Lasso
            lasso_model = Lasso(alpha=st.session_state.get("lasso_alpha", 0.01), random_state=42)
            lasso_model.fit(X_train_scaled, y_train)
            lasso_forecast = lasso_model.predict(X_test_scaled)
            metrics["LASSO"] = {"RMSE": np.sqrt(mean_squared_error(y_test, lasso_forecast)),
                                  "MAPE": mean_absolute_percentage_error(y_test, lasso_forecast)}
        except Exception as e:
            metrics["LASSO"] = {"Error": str(e)}
            st.error(f"Error in LASSO: {str(e)}")

        try:
            from sklearn.linear_model import Ridge
            ridge_model = Ridge(alpha=st.session_state.get("ridge_alpha", 0.01), random_state=42)
            ridge_model.fit(X_train_scaled, y_train)
            ridge_forecast = ridge_model.predict(X_test_scaled)
            metrics["Ridge"] = {"RMSE": np.sqrt(mean_squared_error(y_test, ridge_forecast)),
                                  "MAPE": mean_absolute_percentage_error(y_test, ridge_forecast)}
        except Exception as e:
            metrics["Ridge"] = {"Error": str(e)}
            st.error(f"Error in Ridge: {str(e)}")

        try:
            from sklearn.linear_model import ElasticNet
            elasticnet_model = ElasticNet(alpha=st.session_state.get("elasticnet_alpha", 0.01), 
                                           l1_ratio=st.session_state.get("elasticnet_l1_ratio", 0.5), 
                                           random_state=42)
            elasticnet_model.fit(X_train_scaled, y_train)
            elasticnet_forecast = elasticnet_model.predict(X_test_scaled)
            metrics["Elastic Net"] = {"RMSE": np.sqrt(mean_squared_error(y_test, elasticnet_forecast)),
                                      "MAPE": mean_absolute_percentage_error(y_test, elasticnet_forecast)}
        except Exception as e:
            metrics["Elastic Net"] = {"Error": str(e)}
            st.error(f"Error in Elastic Net: {str(e)}")

        # For time series models like VAR, VECM, and VARX, we use the original (unscaled) data.
        try:
            from statsmodels.tsa.vector_ar.var_model import VAR
            var_model = VAR(st.session_state["data"])
            num_obs = st.session_state["data"].shape[0]
            num_vars = st.session_state["data"].shape[1]
            max_possible_lags = (num_obs - 1) // num_vars
            if max_possible_lags < 1:
                max_possible_lags = 1
            maxlags_to_use = min(2, max_possible_lags)
            var_fit = var_model.fit(maxlags=maxlags_to_use, ic='aic')
            var_forecast = var_fit.forecast(st.session_state["data"].values[-var_fit.k_ar:], steps=len(y_test))
            metrics["VAR"] = {"RMSE": np.sqrt(mean_squared_error(y_test, var_forecast[:, 0])),
                              "MAPE": mean_absolute_percentage_error(y_test, var_forecast[:, 0])}
        except Exception as e:
            metrics["VAR"] = {"Error": str(e)}
            st.error(f"Error in VAR: {str(e)}")

        try:
            from statsmodels.tsa.vector_ar.vecm import VECM
            vecm_model = VECM(st.session_state["data"], k_ar_diff=1, coint_rank=1)
            vecm_fit = vecm_model.fit()
            vecm_forecast = vecm_fit.predict(steps=len(y_test))
            metrics["VECM"] = {"RMSE": np.sqrt(mean_squared_error(y_test, vecm_forecast[:, 0])),
                               "MAPE": mean_absolute_percentage_error(y_test, vecm_forecast[:, 0])}
        except Exception as e:
            metrics["VECM"] = {"Error": str(e)}
            st.error(f"Error in VECM: {str(e)}")

        # --- VARX MODEL ---
        try:
            from statsmodels.tsa.statespace.varmax import VARMAX
            target_var = st.session_state["target_var"]
            endog = st.session_state["data"][[target_var]]
            exog = st.session_state["data"][st.session_state["selected_features"]]
            
            if endog.shape[1] < 2:
                st.info("Endogenous series is univariate. Using SARIMAX (ARIMAX) as a VARX equivalent.")
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                sarimax_varx_model = SARIMAX(endog, exog=exog, order=(2, 0, 0))
                sarimax_varx_fit = sarimax_varx_model.fit(disp=False)
                exog_future = exog.iloc[-len(y_test):]
                varx_forecast = sarimax_varx_fit.forecast(steps=len(y_test), exog=exog_future)
            else:
                varx_model = VARMAX(endog, exog=exog, order=(2, 0)).fit(disp=False)
                exog_future = exog.iloc[-len(y_test):]
                varx_forecast = varx_model.forecast(steps=len(y_test), exog=exog_future)
            
            if isinstance(varx_forecast, pd.Series):
                forecast_target = varx_forecast
            else:
                forecast_target = varx_forecast.iloc[:, 0]
            
            metrics["VARX"] = {"RMSE": np.sqrt(mean_squared_error(y_test, forecast_target)),
                               "MAPE": mean_absolute_percentage_error(y_test, forecast_target)}
        except Exception as e:
            metrics["VARX"] = {"Error": str(e)}
            st.error(f"Error in VARX: {str(e)}")

        # --- ECM MODEL ---
        try:
            # Here, we implement an Error Correction Model (ECM) using a VECM framework.
            # Note: In multivariate settings, a VECM can be interpreted as an ECM.
            from statsmodels.tsa.vector_ar.vecm import VECM
            ecm_model = VECM(st.session_state["data"], k_ar_diff=1, coint_rank=1)
            ecm_fit = ecm_model.fit()
            ecm_forecast = ecm_fit.predict(steps=len(y_test))
            metrics["ECM"] = {"RMSE": np.sqrt(mean_squared_error(y_test, ecm_forecast[:, 0])),
                              "MAPE": mean_absolute_percentage_error(y_test, ecm_forecast[:, 0])}
        except Exception as e:
            metrics["ECM"] = {"Error": str(e)}
            st.error(f"Error in ECM: {str(e)}")

        # Remove models with an "Error" key so that only valid metrics are shown
        cleaned_metrics = {model: vals for model, vals in metrics.items() if "Error" not in vals}
        st.session_state["valid_models"] = cleaned_metrics

        st.subheader("📊 Model Metrics")
        metrics_df = pd.DataFrame(cleaned_metrics).T
        st.table(metrics_df)

        # Identify the best model using both RMSE and MAPE.
        if not metrics_df.empty and "RMSE" in metrics_df.columns and "MAPE" in metrics_df.columns:
            best_rmse = metrics_df["RMSE"].idxmin()
            best_mape = metrics_df["MAPE"].idxmin()
            if best_rmse == best_mape:
                best_model = best_rmse
            else:
                # If they differ, default to RMSE as it is more sensitive to large errors.
                best_model = best_rmse
            st.write(f"**Best Model Based on RMSE/MAPE:** {best_model}")
            st.session_state["best_model"] = best_model
        else:
            st.write("No valid RMSE/MAPE metrics available to identify the best model.")

    else:
        st.warning("⚠️ Please complete the Variable Selection step first.")






##################################
##### Results and Recommendations
##################################

elif selected_tab == "Results and Recommendations":
    st.header("Results and Recommendations")
    valid_models = st.session_state.get("valid_models", {})

    # Retrieve the best model from the Model Selection tab (saved in session state)
    best_model = st.session_state.get("best_model", None)
    if best_model:
        st.markdown(
            f"""<div style="background-color:#e0f7fa; padding:10px; border-radius:5px; 
            font-size:16px; font-weight:bold; text-align:center;">
            Best Model Based on RMSE/MAPE: {best_model}
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.info("No best model was identified.")

    if valid_models:
        st.subheader("Select a Model for Recommendations")
        selected_model = st.selectbox(
            "Choose a model for recommendations",
            options=list(valid_models.keys()),
            index=0,
            key="results_rec_select"
        )
        st.success(f"Selected Model for Recommendations: {selected_model}")

        st.write("Metrics:")
        st.json(valid_models[selected_model])

        st.subheader(f"Model Details and Outputs: {selected_model}")
        # ----- SARIMAX -----
        if selected_model == "SARIMAX":
            try:
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                sarimax_model = SARIMAX(
                    st.session_state["data"][st.session_state["target_var"]],
                    exog=st.session_state["data"][st.session_state["selected_features"]],
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, 12)
                ).fit(disp=False)
                st.write(sarimax_model.summary())
            except Exception as e:
                st.error(f"Error retrieving SARIMAX details: {e}")

        # ----- OLS -----
        elif selected_model == "OLS":
            try:
                from statsmodels.api import OLS, add_constant
                X = st.session_state["data"][st.session_state["selected_features"]]
                y = st.session_state["data"][st.session_state["target_var"]]
                X_const = add_constant(X)
                ols_model = OLS(y, X_const).fit()
                st.markdown("#### OLS Regression Results")
                st.text(ols_model.summary())
            except Exception as e:
                st.error(f"Error retrieving OLS details: {e}")

        # ----- LASSO/Ridge/Elastic Net -----
        elif selected_model in ["LASSO", "Ridge", "Elastic Net"]:
            try:
                X = st.session_state["data"][st.session_state["selected_features"]]
                y = st.session_state["data"][st.session_state["target_var"]]
                if selected_model == "LASSO":
                    from sklearn.linear_model import Lasso
                    model = Lasso(alpha=st.session_state.get("lasso_alpha", 0.01), random_state=42)
                elif selected_model == "Ridge":
                    from sklearn.linear_model import Ridge
                    model = Ridge(alpha=st.session_state.get("ridge_alpha", 0.01), random_state=42)
                elif selected_model == "Elastic Net":
                    from sklearn.linear_model import ElasticNet
                    model = ElasticNet(alpha=st.session_state.get("elasticnet_alpha", 0.01),
                                       l1_ratio=st.session_state.get("elasticnet_l1_ratio", 0.5),
                                       random_state=42)
                model.fit(X, y)
                import numpy as np
                y_pred = model.predict(X)
                r2 = model.score(X, y)
                rmse = np.sqrt(np.mean((y - y_pred) ** 2))
                coefs = model.coef_
                intercept = model.intercept_
                summary_df = pd.DataFrame({"Coefficient": coefs}, index=X.columns)
                summary_str = f"Intercept: {intercept}\n\n" + summary_df.to_string()
                summary_str += f"\n\nR-squared: {r2:.4f}\nRMSE: {rmse:.4f}"
                st.markdown(f"#### {selected_model} Regression Results")
                st.text(summary_str)
            except Exception as e:
                st.error(f"Error retrieving {selected_model} details: {e}")

        # ----- Random Forest -----
        elif selected_model == "Random Forest":
            st.write(
                f"### Random Forest Model\nThe **Random Forest** model is robust for capturing nonlinear relationships and interactions."
                f"\n\n**Performance:** RMSE: {valid_models[selected_model].get('RMSE', 'N/A'):.2f}, "
                f"MAPE: {valid_models[selected_model].get('MAPE', 'N/A'):.2f}"
            )
            st.info("**Use Case:** Suitable for datasets with complex and diverse predictors.")

        # ----- Gradient Boosting -----
        elif selected_model == "Gradient Boosting":
            st.write(
                f"### Gradient Boosting Model\nThe **Gradient Boosting** model builds sequentially to improve predictions by correcting previous errors."
                f"\n\n**Performance:** RMSE: {valid_models[selected_model].get('RMSE', 'N/A'):.2f}, "
                f"MAPE: {valid_models[selected_model].get('MAPE', 'N/A'):.2f}"
            )
            st.info("**Use Case:** Ideal for high-accuracy prediction tasks such as pricing optimization.")

        # ----- XGBoost -----
        elif selected_model == "XGBoost":
            st.write(
                f"### Extreme Gradient Boosting (XGBoost)\nXGBoost is optimized for performance and efficiency, handling large datasets well."
                f"\n\n**Performance:** RMSE: {valid_models[selected_model].get('RMSE', 'N/A'):.2f}, "
                f"MAPE: {valid_models[selected_model].get('MAPE', 'N/A'):.2f}"
            )
            st.info("**Use Case:** Effective for demand forecasting, risk assessment, and marketing analytics.")

        # ----- LightGBM -----
        elif selected_model == "LightGBM":
            st.write(
                f"### LightGBM Model\nLightGBM is designed for speed and scalability with a histogram-based approach."
                f"\n\n**Performance:** RMSE: {valid_models[selected_model].get('RMSE', 'N/A'):.2f}, "
                f"MAPE: {valid_models[selected_model].get('MAPE', 'N/A'):.2f}"
            )
            st.info("**Use Case:** Suitable for high-dimensional data and tasks like inventory management.")

        # ----- SVR -----
        elif selected_model == "SVR":
            st.write(
                f"### Support Vector Regression (SVR)\nSVR uses kernel functions to capture nonlinear patterns and is robust to outliers."
                f"\n\n**Performance:** RMSE: {valid_models[selected_model].get('RMSE', 'N/A'):.2f}, "
                f"MAPE: {valid_models[selected_model].get('MAPE', 'N/A'):.2f}"
            )
            st.info("**Use Case:** Best for small-to-medium datasets with complex nonlinear relationships.")

        # ----- VAR MODEL -----
        elif selected_model == "VAR":
            st.subheader("Additional VAR Outputs for Selected Features")
            if st.button("Compute VAR Model on Selected Features", key="compute_var"):
                try:
                    from statsmodels.tsa.vector_ar.var_model import VAR
                    var_data = st.session_state["data"][st.session_state["selected_features"]]
                    st.session_state["var_fit"] = VAR(var_data).fit(maxlags=2, ic='aic')
                    st.success("VAR model computed successfully.")
                except Exception as e:
                    st.error(f"Error computing VAR model: {e}")
            if "var_fit" in st.session_state:
                if st.button("Show VAR Model Details", key="show_var_details"):
                    st.write("#### VAR Model Summary")
                    st.write(st.session_state["var_fit"].summary())
                if st.button("Show VAR Impulse Response Functions (IRF)", key="show_var_irf"):
                    try:
                        irf = st.session_state["var_fit"].irf(10)
                        st.write("#### VAR Impulse Response Functions")
                        fig = irf.plot(orth=False)
                        st.pyplot(fig.figure)
                    except Exception as e:
                        st.error(f"Error displaying VAR IRF: {e}")

        # ----- VARX MODEL -----
        elif selected_model == "VARX":
            st.subheader("Additional VARX Outputs for Selected Features")
            if st.button("Compute VARX Model on Selected Features", key="compute_varx"):
                try:
                    import pandas as pd
                    target_var = st.session_state["target_var"]
                    data_df = st.session_state["data"]
                    endog = pd.concat([data_df[target_var], data_df[target_var].shift(1)], axis=1).dropna()
                    endog.columns = [target_var, target_var + "_lag1"]
                    exog = data_df[st.session_state["selected_features"]].loc[endog.index]
                    from statsmodels.tsa.statespace.varmax import VARMAX
                    varx_model = VARMAX(endog, exog=exog, order=(2, 0)).fit(disp=False)
                    st.session_state["varx_fit"] = varx_model
                    st.session_state["varx_model_type"] = "VARMAX_multivariate"
                    st.success("VARX model computed successfully using a multivariate endogenous system.")
                except Exception as e:
                    st.error(f"Error computing VARX model: {e}")
            if "varx_fit" in st.session_state:
                if st.button("Show VARX Model Details", key="show_varx_details"):
                    st.write("#### VARX Model Summary")
                    st.write(st.session_state["varx_fit"].summary())
                if st.button("Show VARX Impulse Response Functions (IRF)", key="show_varx_irf"):
                    try:
                        irf = st.session_state["varx_fit"].irf(10)
                        st.write("#### VARX Impulse Response Functions")
                        fig = irf.plot(orth=False)
                        st.pyplot(fig.figure)
                    except Exception as e:
                        st.error(f"Error displaying VARX IRF: {e}")

        # ----- VECM MODEL -----
        elif selected_model == "VECM":
            st.subheader("Additional VECM Outputs for Selected Features")
            if st.button("Show VECM Model Details", key="show_vecm_details"):
                try:
                    from statsmodels.tsa.vector_ar.vecm import VECM
                    vecm_data = st.session_state["data"][st.session_state["selected_features"]]
                    vecm_model = VECM(vecm_data, k_ar_diff=1, coint_rank=1)
                    vecm_fit = vecm_model.fit()
                    st.session_state["vecm_fit"] = vecm_fit
                    st.write("#### VECM Model Summary")
                    st.write(vecm_fit.summary())
                except Exception as e:
                    st.error(f"Error displaying VECM details: {e}")
            if "vecm_fit" in st.session_state:
                if st.button("Show VECM Impulse Response Functions (IRF)", key="show_vecm_irf"):
                    try:
                        irf = st.session_state["vecm_fit"].irf(10)
                        st.write("#### VECM Impulse Response Functions")
                        fig = irf.plot(orth=False)
                        st.pyplot(fig.figure)
                    except Exception as e:
                        st.error(f"Error displaying VECM IRF: {e}")

        # ----- ECM MODEL (Backtesting version using differences and converting back to levels) -----
        elif selected_model == "ECM":
            st.subheader("ECM Model Details and Outputs")
            try:
                import statsmodels.api as sm
                selected_features = st.session_state["selected_features"]
                target = st.session_state["target_var"]
                data_levels = st.session_state["data"].copy()
                # Estimate cointegration regression using all selected features (levels)
                X_ecm = sm.add_constant(data_levels[selected_features])
                coint_model = sm.OLS(data_levels[target], X_ecm).fit()
                # Compute the error correction term (ECT)
                data_levels["ECT"] = data_levels[target] - X_ecm.dot(coint_model.params)
                # Compute first differences for target and all selected features
                data_diff = data_levels.diff().dropna()
                # Include lagged ECT (lag 1)
                data_diff["lagECT"] = data_levels["ECT"].shift(1).loc[data_diff.index]
                # Also include the current differences for each selected feature
                for feat in selected_features:
                    data_diff["diff_" + feat] = data_levels[feat].diff().loc[data_diff.index]
                data_diff = data_diff.dropna()
                # Set up ECM regression: change in target ~ constant + lagECT + differences of selected features
                X_ecm_reg = sm.add_constant(data_diff[["lagECT"] + [ "diff_" + feat for feat in selected_features]])
                y_ecm_reg = data_levels[target].diff().loc[data_diff.index]
                ecm_model = sm.OLS(y_ecm_reg, X_ecm_reg).fit()
                st.markdown("#### ECM Model Summary")
                st.text(ecm_model.summary())
                st.markdown("""<div style="font-size:14px; color:#555;">
                The ECM model above estimates the change in the target variable as a function of a constant, 
                the lagged error correction term, and the current differences of the selected features.
                </div>""", unsafe_allow_html=True)
                st.session_state["ecm_model"] = ecm_model
                st.session_state["data_ecm"] = data_levels
            except Exception as e:
                st.error(f"Error retrieving ECM details: {e}")

        # ---------------- Backtesting Block ----------------
        st.subheader("Backtesting Results")
        try:
            data = st.session_state["data"]
            target_var = st.session_state["target_var"]

            ml_flat_models = ["Random Forest", "Gradient Boosting", "XGBoost", "LightGBM", "SVR"]
            if selected_model in ml_flat_models:
                data_aug = data.copy()
                lag_name = target_var + "_lag1"
                data_aug[lag_name] = data_aug[target_var].shift(1)
                data_aug = data_aug.dropna()
                if lag_name not in st.session_state["selected_features"]:
                    features_for_ml = st.session_state["selected_features"] + [lag_name]
                else:
                    features_for_ml = st.session_state["selected_features"]
                X = data_aug[features_for_ml]
                y = data_aug[target_var]
            else:
                X = data[st.session_state["selected_features"]]
                y = data[target_var]

            train_size_ratio = 0.7
            train_size = int(len(X) * train_size_ratio)
            X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
            y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

            forecast = None

            if selected_model == "SARIMAX":
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                sarimax_model = SARIMAX(y_train, exog=X_train, order=(1, 1, 1),
                                        seasonal_order=(1, 1, 1, 12)).fit(disp=False)
                forecast = sarimax_model.forecast(steps=len(y_test), exog=X_test)
            elif selected_model == "OLS":
                from statsmodels.api import OLS, add_constant
                X_train_const = add_constant(X_train)
                X_test_const = add_constant(X_test)
                ols_model = OLS(y_train, X_train_const).fit()
                forecast = ols_model.predict(X_test_const)
            elif selected_model == "Random Forest":
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=200, max_depth=10,
                                              min_samples_leaf=5, random_state=42)
                model.fit(X_train, y_train)
                forecast = model.predict(X_test)
            elif selected_model == "Gradient Boosting":
                from sklearn.ensemble import GradientBoostingRegressor
                model = GradientBoostingRegressor(random_state=42)
                model.fit(X_train, y_train)
                forecast = model.predict(X_test)
            elif selected_model == "XGBoost":
                import xgboost as xgb
                model = xgb.XGBRegressor(random_state=42)
                model.fit(X_train, y_train)
                forecast = model.predict(X_test)
            elif selected_model == "LightGBM":
                import lightgbm as lgb
                model = lgb.LGBMRegressor(random_state=42)
                model.fit(X_train, y_train)
                forecast = model.predict(X_test)
            elif selected_model == "SVR":
                from sklearn.svm import SVR
                model = SVR()
                model.fit(X_train, y_train)
                forecast = model.predict(X_test)
            elif selected_model in ["LASSO", "Ridge", "Elastic Net"]:
                if selected_model == "LASSO":
                    from sklearn.linear_model import Lasso
                    model = Lasso(alpha=st.session_state.get("lasso_alpha", 0.01), random_state=42)
                elif selected_model == "Ridge":
                    from sklearn.linear_model import Ridge
                    model = Ridge(alpha=st.session_state.get("ridge_alpha", 0.01), random_state=42)
                elif selected_model == "Elastic Net":
                    from sklearn.linear_model import ElasticNet
                    model = ElasticNet(alpha=st.session_state.get("elasticnet_alpha", 0.01),
                                       l1_ratio=st.session_state.get("elasticnet_l1_ratio", 0.5),
                                       random_state=42)
                model.fit(X_train, y_train)
                forecast = model.predict(X_test)
            elif selected_model == "VAR":
                from statsmodels.tsa.vector_ar.var_model import VAR
                var_model = VAR(data)
                var_fit = var_model.fit(maxlags=2, ic='aic')
                forecast = var_fit.forecast(data.values[-var_fit.k_ar:], steps=len(y_test))
                forecast = forecast[:, 0]
            elif selected_model == "VARX":
                if "varx_fit" in st.session_state:
                    exog_future = data[st.session_state["selected_features"]].iloc[-len(y_test):]
                    model_type = st.session_state.get("varx_model_type", "VARMAX")
                    forecast = st.session_state["varx_fit"].forecast(steps=len(y_test), exog=exog_future)
                    if model_type == "VARMAX_multivariate":
                        if isinstance(forecast, pd.DataFrame):
                            forecast = forecast.iloc[:, 0].values
                        else:
                            forecast = forecast
                    elif model_type == "SARIMAX":
                        if isinstance(forecast, pd.Series):
                            forecast = forecast.values
                        else:
                            forecast = forecast
                else:
                    st.error("VARX model has not been computed.")
            elif selected_model == "VECM":
                from statsmodels.tsa.vector_ar.vecm import VECM
                vecm_model = VECM(data, k_ar_diff=1, coint_rank=1)
                vecm_fit = vecm_model.fit()
                forecast = vecm_fit.predict(steps=len(y_test))
                forecast = forecast[:, 0]
            elif selected_model == "ECM":
                try:
                    import statsmodels.api as sm
                    data_ecm = data.copy()
                    selected_features = st.session_state["selected_features"]
                    target = st.session_state["target_var"]
                    # Estimate cointegration regression in levels
                    X_ecm = sm.add_constant(data_ecm[selected_features])
                    coint_model = sm.OLS(data_ecm[target], X_ecm).fit()
                    data_ecm["ECT"] = data_ecm[target] - X_ecm.dot(coint_model.params)
                    # Compute first differences for target and selected features
                    data_diff = data_ecm.diff().dropna()
                    # Include lagged ECT
                    data_diff["lagECT"] = data_ecm["ECT"].shift(1).loc[data_diff.index]
                    # Include one lag of differences for each selected feature
                    for feat in selected_features:
                        data_diff["lag_d_" + feat] = data_diff[feat].shift(1)
                    data_diff = data_diff.dropna()
                    # Use the same train-test split on original data
                    original_train_size = int(len(data_ecm) * 0.7)
                    test_index_ecm = data_diff.index.intersection(data_ecm.index[original_train_size:])
                    # Estimate ECM on training portion of data_diff
                    X_ecm_reg = sm.add_constant(data_diff[["lagECT"] + selected_features + ["lag_d_" + feat for feat in selected_features]])
                    y_ecm_reg = data_diff[target]
                    train_index = data_diff.index[:original_train_size - 2]  # adjust for differencing and lag
                    ecm_model = sm.OLS(y_ecm_reg.loc[train_index], X_ecm_reg.loc[train_index]).fit()
                    st.session_state["ecm_model"] = ecm_model
                    # Backtesting ECM: predict on test_index_ecm and accumulate forecast
                    pred_dtarget = ecm_model.predict(X_ecm_reg.loc[test_index_ecm])
                    # Use the last observed level from the original series at index just before test period
                    last_level_index = data_ecm.index[data_ecm.index.get_loc(test_index_ecm[0]) - 1]
                    last_level = data_ecm[target].loc[last_level_index]
                    forecast_ecm = last_level + pred_dtarget.cumsum()
                    forecast = forecast_ecm.values
                except Exception as e:
                    st.error(f"Error during ECM backtesting: {e}")
                    forecast = None

            if forecast is not None and len(forecast) == len(y_test):
                forecast_series = pd.Series(forecast, index=y_test.index)
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 6))
                plt.plot(y_test.index, y_test, label="Actual", marker="o")
                plt.plot(y_test.index, forecast_series, label="Forecast", marker="x")
                plt.xlabel("Date")
                plt.ylabel("Target Variable")
                plt.title(f"Backtesting Results - {selected_model}")
                plt.legend()
                st.pyplot(plt)
            elif selected_model == "ECM":
                st.info("ECM backtesting completed. Please review the ECM forecast summary and plot below.")
                if forecast is not None:
                    forecast_series = pd.Series(forecast, index=y_test.index)
                    plt.figure(figsize=(12, 6))
                    plt.plot(y_test.index, y_test, label="Actual", marker="o")
                    plt.plot(y_test.index, forecast_series, label="ECM Forecast", marker="x")
                    plt.xlabel("Date")
                    plt.ylabel("Target Variable")
                    plt.title(f"Backtesting Results - {selected_model}")
                    plt.legend()
                    st.pyplot(plt)
            else:
                st.error("Forecast and y_test lengths do not match or forecast is None.")
        except Exception as e:
            st.error(f"Error during backtesting: {str(e)}")
    else:
        st.warning("No valid models available. Please complete the Model Selection step.")







##################################
##### What If Scenario 
##################################

elif selected_tab == "What If Scenarios":
    st.title("🧠 What-If Scenarios")
    
    # Display the best model from the Model Selection tab (if available)
    best_model = st.session_state.get("best_model", None)
    if best_model:
        st.info(f"Best Model from Model Selection: {best_model}")
    
    if "valid_models" in st.session_state and st.session_state["valid_models"]:
        valid_models = st.session_state["valid_models"]
        selected_model = st.selectbox(
            "Select a Model for What-If Scenarios",
            options=list(valid_models.keys()) + ["Ensemble"],
            key="whatif_model_select"
        )
        st.success(f"Selected Model for What-If Analysis: {selected_model}")

        X = st.session_state["data"][st.session_state["selected_features"]]
        st.subheader("Provide Input Values for Independent Variables")
        user_inputs = {}
        for feature in st.session_state["selected_features"]:
            default_value = X[feature].mean()
            user_inputs[feature] = st.number_input(
                f"Enter value for {feature}",
                value=float(default_value),
                step=0.1,
                key=f"input_{feature}"
            )
        input_df = pd.DataFrame([user_inputs])

        st.subheader("Forecast Results")
        forecasts = []
        try:
            forecast = None
            # SARIMAX
            if selected_model in ["SARIMAX", "Ensemble"]:
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                sarimax_model = SARIMAX(
                    st.session_state["data"][st.session_state["target_var"]],
                    exog=X,
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, 12)
                ).fit(disp=False)
                forecast = sarimax_model.get_forecast(steps=1, exog=input_df).predicted_mean.iloc[0]
                forecasts.append(forecast)
                if selected_model == "SARIMAX":
                    st.success(f"Forecasted Target Variable (SARIMAX): {forecast:.2f}")

            # OLS
            if selected_model in ["OLS", "Ensemble"]:
                from statsmodels.api import OLS, add_constant
                X_const = add_constant(X)
                ols_model = OLS(st.session_state["data"][st.session_state["target_var"]], X_const).fit()
                input_const = add_constant(input_df, has_constant="add")
                forecast = ols_model.predict(input_const).iloc[0]
                forecasts.append(forecast)
                if selected_model == "OLS":
                    st.success(f"Forecasted Target Variable (OLS): {forecast:.2f}")

            # Random Forest
            if selected_model in ["Random Forest", "Ensemble"]:
                from sklearn.ensemble import RandomForestRegressor
                rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=42)
                rf_model.fit(X, st.session_state["data"][st.session_state["target_var"]])
                forecast = rf_model.predict(input_df)[0]
                forecasts.append(forecast)
                if selected_model == "Random Forest":
                    st.success(f"Forecasted Target Variable (Random Forest): {forecast:.2f}")

            # Gradient Boosting
            if selected_model in ["Gradient Boosting", "Ensemble"]:
                from sklearn.ensemble import GradientBoostingRegressor
                gb_model = GradientBoostingRegressor(random_state=42)
                gb_model.fit(X, st.session_state["data"][st.session_state["target_var"]])
                forecast = gb_model.predict(input_df)[0]
                forecasts.append(forecast)
                if selected_model == "Gradient Boosting":
                    st.success(f"Forecasted Target Variable (Gradient Boosting): {forecast:.2f}")

            # XGBoost
            if selected_model in ["XGBoost", "Ensemble"]:
                import xgboost as xgb
                xgb_model = xgb.XGBRegressor(random_state=42)
                xgb_model.fit(X, st.session_state["data"][st.session_state["target_var"]])
                forecast = xgb_model.predict(input_df)[0]
                forecasts.append(forecast)
                if selected_model == "XGBoost":
                    st.success(f"Forecasted Target Variable (XGBoost): {forecast:.2f}")

            # LightGBM
            if selected_model in ["LightGBM", "Ensemble"]:
                import lightgbm as lgb
                lgb_model = lgb.LGBMRegressor(random_state=42)
                lgb_model.fit(X, st.session_state["data"][st.session_state["target_var"]])
                forecast = lgb_model.predict(input_df)[0]
                forecasts.append(forecast)
                if selected_model == "LightGBM":
                    st.success(f"Forecasted Target Variable (LightGBM): {forecast:.2f}")

            # SVR
            if selected_model in ["SVR", "Ensemble"]:
                from sklearn.svm import SVR
                svr_model = SVR()
                svr_model.fit(X, st.session_state["data"][st.session_state["target_var"]])
                forecast = svr_model.predict(input_df)[0]
                forecasts.append(forecast)
                if selected_model == "SVR":
                    st.success(f"Forecasted Target Variable (SVR): {forecast:.2f}")

            # LASSO, Ridge, Elastic Net
            if selected_model in ["LASSO", "Ridge", "Elastic Net"]:
                if selected_model == "LASSO":
                    from sklearn.linear_model import Lasso
                    model = Lasso(alpha=st.session_state.get("lasso_alpha", 0.01), random_state=42)
                elif selected_model == "Ridge":
                    from sklearn.linear_model import Ridge
                    model = Ridge(alpha=st.session_state.get("ridge_alpha", 0.01), random_state=42)
                elif selected_model == "Elastic Net":
                    from sklearn.linear_model import ElasticNet
                    model = ElasticNet(alpha=st.session_state.get("elasticnet_alpha", 0.01),
                                       l1_ratio=st.session_state.get("elasticnet_l1_ratio", 0.5),
                                       random_state=42)
                model.fit(X, st.session_state["data"][st.session_state["target_var"]])
                forecast = model.predict(input_df)[0]
                forecasts.append(forecast)
                st.success(f"Forecasted Target Variable ({selected_model}): {forecast:.2f}")

            # ECM
            if selected_model in ["ECM", "Ensemble"]:
                import statsmodels.api as sm
                data_levels = st.session_state["data"].copy()
                selected_features = st.session_state["selected_features"]
                target = st.session_state["target_var"]
                # Estimate cointegration regression using levels
                X_ecm = sm.add_constant(data_levels[selected_features])
                coint_model = sm.OLS(data_levels[target], X_ecm).fit()
                data_levels["ECT"] = data_levels[target] - X_ecm.dot(coint_model.params)
                # Compute first differences for target and selected features
                data_diff = data_levels.diff().dropna()
                # Include lagged ECT (lag 1)
                data_diff["lagECT"] = data_levels["ECT"].shift(1).loc[data_diff.index]
                # Include one lag of differences for each selected feature
                for feat in selected_features:
                    data_diff["lag_d_" + feat] = data_diff[feat].shift(1)
                data_diff = data_diff.dropna()
                # Estimate ECM on all available differenced data
                X_ecm_reg = sm.add_constant(data_diff[["lagECT"] + selected_features + ["lag_d_" + feat for feat in selected_features]])
                y_ecm_reg = data_levels[target].diff().loc[data_diff.index]
                ecm_model = sm.OLS(y_ecm_reg, X_ecm_reg).fit()
                # For one-step forecast:
                # Use the last observation from the levels data
                last_obs = data_levels.iloc[-1]
                # Construct a new row based on user input: compute differences from last observed levels
                new_row = {}
                new_row["lagECT"] = last_obs["ECT"]
                for feat in selected_features:
                    diff_val = input_df[feat].iloc[0] - last_obs[feat]
                    new_row[feat] = diff_val
                    new_row["lag_d_" + feat] = diff_val
                new_df = pd.DataFrame([new_row])
                new_df = sm.add_constant(new_df, has_constant="add")
                # Reindex to ensure columns align with the ECM training matrix
                new_df = new_df.reindex(columns=X_ecm_reg.columns, fill_value=0)
                pred_diff = ecm_model.predict(new_df)[0]
                forecast = last_obs[target] + pred_diff
                forecasts.append(forecast)
                if selected_model == "ECM":
                    st.success(f"Forecasted Target Variable (ECM): {forecast:.2f}")

            # VAR
            if selected_model == "VAR":
                from statsmodels.tsa.vector_ar.var_model import VAR
                var_model = VAR(st.session_state["data"])
                data_temp = st.session_state["data"]
                num_obs = data_temp.shape[0]
                num_vars = data_temp.shape[1]
                max_possible_lags = (num_obs - 1) // num_vars
                if max_possible_lags < 1:
                    max_possible_lags = 1
                maxlags_to_use = min(2, max_possible_lags)
                var_results = var_model.fit(maxlags=maxlags_to_use)
                f = var_results.forecast(data_temp.values[-var_results.k_ar:], steps=1)
                forecast = f[0][0]
                forecasts.append(forecast)
                if selected_model == "VAR":
                    st.success(f"Forecasted Target Variable (VAR): {forecast:.2f}")

            # VECM
            if selected_model == "VECM":
                from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
                jres = coint_johansen(st.session_state["data"], det_order=0, k_ar_diff=1)
                crit = jres.cvt[:, 1]
                cointegration_rank = sum(jres.lr1 > crit)
                if cointegration_rank > 0:
                    vecm_model = VECM(st.session_state["data"], coint_rank=cointegration_rank, k_ar_diff=1).fit()
                    vecm_forecast = vecm_model.predict(steps=1)
                    forecast = vecm_forecast[0, 0]
                    forecasts.append(forecast)
                    if selected_model == "VECM":
                        st.success(f"Forecasted Target Variable (VECM): {forecast:.2f}")
                else:
                    st.error("No cointegration found for VECM.")
            
            # VARX
            if selected_model == "VARX":
                import pandas as pd
                data_df = st.session_state["data"]
                target_var = st.session_state["target_var"]
                # Create a multivariate endogenous system using the target variable and its first lag.
                endog = pd.concat([data_df[target_var], data_df[target_var].shift(1)], axis=1).dropna()
                endog.columns = [target_var, target_var + "_lag1"]
                # Align exogenous variables with the new index
                exog = data_df[st.session_state["selected_features"]].loc[endog.index]
                from statsmodels.tsa.statespace.varmax import VARMAX
                varx_model = VARMAX(endog, exog=exog, order=(2, 0)).fit(disp=False)
                forecast_result = varx_model.forecast(steps=1, exog=input_df)
                forecast = forecast_result.iloc[0, 0]
                forecasts.append(forecast)
                if selected_model == "VARX":
                    st.success(f"Forecasted Target Variable (VARX): {forecast:.2f}")

            if selected_model == "Ensemble":
                if forecasts:
                    import numpy as np
                    ensemble_forecast = np.mean(forecasts)
                    st.success(f"Ensemble Forecasted Target Variable: {ensemble_forecast:.2f}")
                else:
                    st.warning("No forecasts available for ensemble calculation.")

        except Exception as e:
            st.error(f"Error during forecasting with {selected_model}: {str(e)}")
    else:
        st.warning("No valid models available for What-If Scenarios.")






#######################################
####### Scenario Forecast 
########################################
elif selected_tab == "Scenario Forecast":
    st.title("📊 Scenario Forecast")

    # Remind users of the best model from the Model Selection step
    best_model = st.session_state.get("best_model", None)
    if best_model:
        st.info(f"Best Model from Model Selection: {best_model}")

    if "valid_models" in st.session_state and st.session_state["valid_models"]:
        st.subheader("Selected Variables for Forecasting")
        if st.session_state["selected_features"] is not None:
            selected_features_str = ", ".join(st.session_state["selected_features"])
            st.info(f"The selected variables for forecasting are: **{selected_features_str}**")
        else:
            st.warning("No variables selected. Please complete the Variable Selection step.")

        valid_models = st.session_state["valid_models"]
        selected_model = st.selectbox(
            "Select a Model for Scenario Forecasting", 
            options=list(valid_models.keys()) + ["Ensemble"],
            key="scenario_forecast_model"
        )
        st.success(f"Selected Model for Scenario Forecasting: {selected_model}")

        # Define blend sliders if applicable
        if selected_model == "VAR":
            blend_var = st.slider("Blend Weight (VAR)", min_value=0.0, max_value=1.0,
                                  value=0.5, step=0.01, key="blend_var_scen")
        elif selected_model == "VECM":
            blend_vecm = st.slider("Blend Weight (VECM)", min_value=0.0, max_value=1.0,
                                   value=1.0, step=0.01, key="blend_vecm_scen")
        elif selected_model == "Ensemble":
            ens_var_blend = st.slider("Blend Weight for Ensemble (VAR)", min_value=0.0,
                                      max_value=1.0, value=0.5, step=0.01, key="ens_var_blend_scen")
            ens_vecm_blend = st.slider("Blend Weight for Ensemble (VECM)", min_value=0.0,
                                       max_value=1.0, value=0.5, step=0.01, key="ens_vecm_blend_scen")

        st.subheader("Upload Scenario Data")
        uploaded_scenarios = st.file_uploader(
            "Upload Excel files for scenarios (one file per scenario)",
            type="xlsx", accept_multiple_files=True
        )
        if uploaded_scenarios:
            scenario_forecasts = {}
            scenario_dates = None
            scenario_data = {}
            for scenario_file in uploaded_scenarios:
                scenario_name = st.text_input(
                    f"Rename Scenario ({scenario_file.name})", 
                    value=scenario_file.name.split(".")[0],
                    key=f"rename_{scenario_file.name}"
                )
                scenario_df = pd.read_excel(scenario_file)
                if not set(st.session_state["selected_features"]).issubset(scenario_df.columns):
                    st.error(f"The file {scenario_file.name} must contain the following columns: {st.session_state['selected_features']}")
                    continue
                if "date" in scenario_df.columns:
                    scenario_dates = pd.to_datetime(scenario_df["date"], errors="coerce").dropna()
                    st.write(f"Using {len(scenario_dates)} dates from the uploaded scenario data:")
                    st.write(scenario_dates)
                else:
                    st.error(f"The file {scenario_file.name} must contain a 'date' column for scenario forecasting.")
                    scenario_dates = None

                scenario_df = scenario_df[st.session_state["selected_features"]]
                st.write(f"Uploaded Data for {scenario_name}:")
                st.dataframe(scenario_df)
                scenario_data[scenario_name] = scenario_df

                scenario_results = []
                for i, row in scenario_df.iterrows():
                    input_df = pd.DataFrame([row])
                    forecast = None
                    # SARIMAX branch
                    if selected_model == "SARIMAX":
                        from statsmodels.tsa.statespace.sarimax import SARIMAX
                        sarimax_model = SARIMAX(
                            st.session_state["data"][st.session_state["target_var"]],
                            exog=st.session_state["data"][st.session_state["selected_features"]],
                            order=(1, 1, 1),
                            seasonal_order=(1, 1, 1, 12)
                        ).fit(disp=False)
                        forecast = sarimax_model.get_forecast(steps=1, exog=input_df).predicted_mean.iloc[0]
                    # OLS branch
                    elif selected_model == "OLS":
                        from statsmodels.api import OLS, add_constant
                        X_const = add_constant(st.session_state["data"][st.session_state["selected_features"]])
                        ols_model = OLS(st.session_state["data"][st.session_state["target_var"]], X_const).fit()
                        input_const = add_constant(input_df, has_constant="add")
                        forecast = ols_model.predict(input_const).iloc[0]
                    # Random Forest branch
                    elif selected_model == "Random Forest":
                        from sklearn.ensemble import RandomForestRegressor
                        rf_model = RandomForestRegressor(random_state=42)
                        rf_model.fit(
                            st.session_state["data"][st.session_state["selected_features"]],
                            st.session_state["data"][st.session_state["target_var"]]
                        )
                        forecast = rf_model.predict(input_df)[0]
                    # Gradient Boosting branch
                    elif selected_model == "Gradient Boosting":
                        from sklearn.ensemble import GradientBoostingRegressor
                        gb_model = GradientBoostingRegressor(random_state=42)
                        gb_model.fit(
                            st.session_state["data"][st.session_state["selected_features"]],
                            st.session_state["data"][st.session_state["target_var"]]
                        )
                        forecast = gb_model.predict(input_df)[0]
                    # LASSO branch
                    elif selected_model == "LASSO":
                        from sklearn.linear_model import Lasso
                        model = Lasso(alpha=st.session_state.get("lasso_alpha", 0.01), random_state=42)
                        model.fit(
                            st.session_state["data"][st.session_state["selected_features"]],
                            st.session_state["data"][st.session_state["target_var"]]
                        )
                        forecast = model.predict(input_df)[0]
                    # Ridge branch
                    elif selected_model == "Ridge":
                        from sklearn.linear_model import Ridge
                        model = Ridge(alpha=st.session_state.get("ridge_alpha", 0.01), random_state=42)
                        model.fit(
                            st.session_state["data"][st.session_state["selected_features"]],
                            st.session_state["data"][st.session_state["target_var"]]
                        )
                        forecast = model.predict(input_df)[0]
                    # Elastic Net branch
                    elif selected_model == "Elastic Net":
                        from sklearn.linear_model import ElasticNet
                        model = ElasticNet(
                            alpha=st.session_state.get("elasticnet_alpha", 0.01),
                            l1_ratio=st.session_state.get("elasticnet_l1_ratio", 0.5),
                            random_state=42
                        )
                        model.fit(
                            st.session_state["data"][st.session_state["selected_features"]],
                            st.session_state["data"][st.session_state["target_var"]]
                        )
                        forecast = model.predict(input_df)[0]
                    # ECM branch
                    elif selected_model == "ECM":
                        import statsmodels.api as sm
                        data_levels = st.session_state["data"].copy()
                        selected_features = st.session_state["selected_features"]
                        target = st.session_state["target_var"]
                        # Estimate cointegration regression using levels
                        X_ecm = sm.add_constant(data_levels[selected_features])
                        coint_model = sm.OLS(data_levels[target], X_ecm).fit()
                        data_levels["ECT"] = data_levels[target] - X_ecm.dot(coint_model.params)
                        # Compute first differences for target and selected features
                        data_diff = data_levels.diff().dropna()
                        # Include lagged ECT (lag 1)
                        data_diff["lagECT"] = data_levels["ECT"].shift(1).loc[data_diff.index]
                        # Include one lag of differences for each selected feature
                        for feat in selected_features:
                            data_diff["lag_d_" + feat] = data_diff[feat].shift(1)
                        data_diff = data_diff.dropna()
                        # Estimate ECM on all available differenced data
                        X_ecm_reg = sm.add_constant(data_diff[["lagECT"] + selected_features + ["lag_d_" + feat for feat in selected_features]])
                        y_ecm_reg = data_levels[target].diff().loc[data_diff.index]
                        ecm_model = sm.OLS(y_ecm_reg, X_ecm_reg).fit()
                        # For one-step forecast:
                        # Use the last observation from the levels data
                        last_obs = data_levels.iloc[-1]
                        # Construct a new row based on scenario input: compute differences from last observed levels
                        new_row = {}
                        new_row["lagECT"] = last_obs["ECT"]
                        for feat in selected_features:
                            diff_val = row[feat] - last_obs[feat]
                            new_row[feat] = diff_val
                            new_row["lag_d_" + feat] = diff_val
                        new_df = pd.DataFrame([new_row])
                        new_df = sm.add_constant(new_df, has_constant="add")
                        # Reindex to match the training matrix
                        new_df = new_df.reindex(columns=X_ecm_reg.columns, fill_value=0)
                        pred_diff = ecm_model.predict(new_df)[0]
                        forecast = last_obs[target] + pred_diff
                    # VAR branch with blending
                    elif selected_model == "VAR":
                        from statsmodels.tsa.vector_ar.var_model import VAR
                        var_model = VAR(st.session_state["data"])
                        var_fit = var_model.fit(maxlags=2, ic='aic')
                        p = var_fit.k_ar
                        historical_last = st.session_state["data"].values[-p:].copy()
                        scenario_input = input_df.reindex(columns=st.session_state["data"].columns)
                        scenario_input = scenario_input.fillna(pd.DataFrame(st.session_state["data"]).mean()).values[0]
                        blended = (1 - blend_var) * historical_last[-1] + blend_var * scenario_input
                        historical_last[-1] = blended
                        forecast = var_fit.forecast(historical_last, steps=1)[0][0]
                    # VECM branch with replacement of last observation by scenario input (no blending)
                    elif selected_model == "VECM":
                        from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
                        jres = coint_johansen(st.session_state["data"], det_order=0, k_ar_diff=1)
                        cointegration_rank = sum(jres.lr1 > jres.cvt[:, 1])
                        if cointegration_rank > 0:
                            new_data = st.session_state["data"].copy()
                            scenario_input = input_df.reindex(columns=new_data.columns)
                            scenario_input = scenario_input.fillna(pd.DataFrame(new_data).mean()).values[0]
                            new_data.iloc[-1] = scenario_input
                            vecm_model = VECM(new_data, k_ar_diff=1, coint_rank=cointegration_rank)
                            vecm_fit = vecm_model.fit()
                            fc = vecm_fit.predict(steps=1)
                            forecast = fc[0, new_data.columns.get_loc(st.session_state["target_var"])]
                        else:
                            forecast = np.nan
                            st.warning("No cointegration found for VECM.")
                    # VARX branch
                    elif selected_model == "VARX":
                        import pandas as pd
                        data_df = st.session_state["data"]
                        target_var = st.session_state["target_var"]
                        endog = pd.concat([data_df[target_var], data_df[target_var].shift(1)], axis=1).dropna()
                        endog.columns = [target_var, target_var + "_lag1"]
                        exog = data_df[st.session_state["selected_features"]].loc[endog.index]
                        from statsmodels.tsa.statespace.varmax import VARMAX
                        varx_model = VARMAX(endog, exog=exog, order=(2, 0)).fit(disp=False)
                        forecast_result = varx_model.forecast(steps=1, exog=input_df)
                        forecast = forecast_result.iloc[0, 0]
                    # Ensemble branch: Average forecasts from available methods
                    elif selected_model == "Ensemble":
                        ensemble_forecasts = []
                        try:
                            from statsmodels.tsa.statespace.sarimax import SARIMAX
                            sarimax_model = SARIMAX(
                                st.session_state["data"][st.session_state["target_var"]],
                                exog=st.session_state["data"][st.session_state["selected_features"]],
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12)
                            ).fit(disp=False)
                            ensemble_forecasts.append(sarimax_model.get_forecast(steps=1, exog=input_df).predicted_mean.iloc[0])
                        except Exception:
                            pass
                        try:
                            from statsmodels.api import OLS, add_constant
                            X_const = add_constant(st.session_state["data"][st.session_state["selected_features"]])
                            ols_model = OLS(st.session_state["data"][st.session_state["target_var"]], X_const).fit()
                            input_const = add_constant(input_df, has_constant="add")
                            ensemble_forecasts.append(ols_model.predict(input_const).iloc[0])
                        except Exception:
                            pass
                        try:
                            from statsmodels.tsa.vector_ar.var_model import VAR
                            var_model = VAR(st.session_state["data"])
                            var_fit = var_model.fit(maxlags=2, ic='aic')
                            p = var_fit.k_ar
                            historical_last = st.session_state["data"][st.session_state["selected_features"]].values[-p:].copy()
                            scenario_input = input_df.reindex(columns=st.session_state["selected_features"])
                            scenario_input = scenario_input.fillna(pd.DataFrame(st.session_state["data"][st.session_state["selected_features"]]).mean()).values[0]
                            blended = (1 - ens_var_blend) * historical_last[-1] + ens_var_blend * scenario_input
                            historical_last[-1] = blended
                            f = var_fit.forecast(historical_last, steps=1)
                            ensemble_forecasts.append(f[0][0])
                        except Exception:
                            pass
                        try:
                            from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
                            jres = coint_johansen(st.session_state["data"], det_order=0, k_ar_diff=1)
                            cointegration_rank = sum(jres.lr1 > jres.cvt[:, 1])
                            if cointegration_rank > 0:
                                new_data = st.session_state["data"].copy()
                                scenario_input = input_df.reindex(columns=new_data.columns).fillna(new_data.mean()).values[0]
                                blended = (1 - ens_vecm_blend) * new_data.iloc[-1].values + ens_vecm_blend * scenario_input
                                new_data.iloc[-1] = blended
                                vecm_model = VECM(new_data, k_ar_diff=1, coint_rank=cointegration_rank)
                                vecm_fit = vecm_model.fit()
                                f = vecm_fit.predict(steps=1)
                                ensemble_forecasts.append(f[0,0])
                            else:
                                pass
                        except Exception:
                            pass
                        if ensemble_forecasts:
                            forecast = np.mean(ensemble_forecasts)
                        else:
                            forecast = np.nan
                    scenario_results.append(forecast)
                scenario_forecasts[scenario_name] = scenario_results
            st.session_state["scenario_forecasts"] = scenario_forecasts
            st.session_state["scenario_data"] = scenario_data
            results_df = pd.DataFrame(scenario_forecasts)
            st.write("Scenario Forecast Results:")
            st.dataframe(results_df)
            st.subheader("Include Historical Data")
            num_hist_points = st.slider("Select number of last historical points to include in the plot:",
                                        min_value=0, max_value=len(st.session_state["data"]), value=10)
            historical_data = st.session_state["data"][st.session_state["target_var"]].iloc[-num_hist_points:]
            historical_dates = historical_data.index
            historical_values = historical_data.values
            plt.figure(figsize=(10, 6))
            if num_hist_points > 0:
                plt.plot(historical_dates, historical_values, marker="o", label="Historical Data", color="blue")
            if scenario_dates is not None:
                for scenario_name, forecasts in scenario_forecasts.items():
                    plt.plot(scenario_dates, forecasts, label=scenario_name)
                plt.title("Scenario Forecasts with Historical Data")
                plt.xlabel("Date")
                plt.ylabel("Forecast Value")
                plt.xticks(rotation=45)
                plt.legend()
                plt.grid()
                st.pyplot(plt)
            else:
                st.warning("No historical dates available in the uploaded data.")
        else:
            st.warning("No economic scenarios available. Please complete the Scenario Forecast step.")
    else:
        st.warning("No valid models available. Please complete the Model Selection step.")




###################################
####### Stress Testing
#######################################

elif selected_tab == "Stress Testing":
    st.title("💥 Stress Testing")
    st.subheader("Bank Assets or Portfolio")
    total_assets = st.number_input("Enter Total Assets (in millions)", value=1000.0, step=0.1, key="total_assets")
    ead = st.number_input("Enter Exposure at Default (EAD) (in millions)", value=500.0, step=0.1, key="ead")
    lgd = st.number_input("Enter Loss Given Default (LGD) (as a percentage)", value=40.0, step=0.1, key="lgd") / 100
    pd_ = st.number_input("Enter Probability of Default (PD) (as a percentage)", value=5.0, step=0.1, key="pd") / 100
    
    def forecast_ead(gdp_growth):
        return ead * (1 + gdp_growth / 100)
    def forecast_lgd(unemployment_rate):
        return lgd * (1 + unemployment_rate / 100)
    def forecast_pd(interest_rate):
        return pd_ * (1 + interest_rate / 100)
    
    if "scenario_data" in st.session_state:
        scenario_data = st.session_state["scenario_data"]
        st.subheader("Select Economic Scenario")
        scenario_name = st.selectbox("Select Scenario", options=list(scenario_data.keys()))
        selected_scenario = scenario_data[scenario_name]
        st.subheader("Select Variables for Forecasting")
        available_variables = list(selected_scenario.columns)
        gdp_growth_var = st.selectbox("Select Variable for GDP Growth", options=available_variables)
        unemployment_rate_var = st.selectbox("Select Variable for Unemployment Rate", options=available_variables)
        interest_rate_var = st.selectbox("Select Variable for Interest Rate", options=available_variables)
        gdp_growth_value = st.number_input(f"Enter Value for {gdp_growth_var}", value=float(selected_scenario[gdp_growth_var].iloc[0]), key="gdp_growth_value")
        unemployment_rate_value = st.number_input(f"Enter Value for {unemployment_rate_var}", value=float(selected_scenario[unemployment_rate_var].iloc[0]), key="unemployment_rate_value")
        interest_rate_value = st.number_input(f"Enter Value for {interest_rate_var}", value=float(selected_scenario[interest_rate_var].iloc[0]), key="interest_rate_value")
        stressed_ead = forecast_ead(gdp_growth_value)
        stressed_lgd = forecast_lgd(unemployment_rate_value)
        stressed_pd = forecast_pd(interest_rate_value)
        expected_loss = stressed_ead * stressed_lgd * stressed_pd
        result_type = st.radio("Select Result Type", ("Aggregated", "Detailed"))
        gdp_growth_mean = selected_scenario[gdp_growth_var].mean()
        unemployment_rate_mean = selected_scenario[unemployment_rate_var].mean()
        interest_rate_mean = selected_scenario[interest_rate_var].mean()
        stressed_ead_mean = forecast_ead(gdp_growth_mean)
        stressed_lgd_mean = forecast_lgd(unemployment_rate_mean)
        stressed_pd_mean = forecast_pd(interest_rate_mean)
        aggregated_expected_loss = stressed_ead_mean * stressed_lgd_mean * stressed_pd_mean
    
        if result_type == "Aggregated":
            st.markdown("""
                <div style="background-color:#f9f9f9; padding:10px; border-radius:5px; border:1px solid #ddd;">
                    <h2 style="color:#4CAF50; text-align:center;">Aggregated Stress Test Results</h2>
                    <p style="color:#555; text-align:center;">A summary of the stress test results based on the selected economic scenario.</p>
                </div>
            """, unsafe_allow_html=True)
            st.subheader("Aggregated Stress Test Results")
            st.write(f"**Scenario:** {scenario_name}")
            st.write(f"Average GDP Growth: {gdp_growth_mean:.2f}%")
            st.write(f"Average Unemployment Rate: {unemployment_rate_mean:.2f}%")
            st.write(f"Average Interest Rate: {interest_rate_mean:.2f}%")
            st.write(f"Stressed EAD: {stressed_ead_mean:.2f} million")
            st.write(f"Stressed LGD: {stressed_lgd_mean:.2%}")
            st.write(f"Stressed PD: {stressed_pd_mean:.2%}")
            st.write(f"Expected Loss: {aggregated_expected_loss:.2f} million")
            st.markdown(
                """
                **Note:** 
                The Aggregated Stress Test Results provide a simplified snapshot of potential financial risk under the selected economic scenario. 
                This high-level summary is useful for strategic decision-making, risk mitigation planning, and compliance with regulatory stress testing requirements.
                For more granular insights, explore the "Stress Test Results for Each Data Point" section.
                """
            )
            st.subheader("Automated Insights and Recommendations")
            insights = []
            if gdp_growth_mean > 100:
                insights.append(
                    f"The average GDP growth of {gdp_growth_mean:.2f}% is unusually high, indicating potential input data issues or an extreme economic scenario. Consider verifying data integrity."
                )
            elif gdp_growth_mean < 0:
                insights.append(
                    "Negative average GDP growth indicates a contracting economy. Consider reducing exposure to high-risk sectors."
                )
            elif gdp_growth_mean < 2:
                insights.append(
                    "Low GDP growth suggests a sluggish economy. Diversifying the portfolio may mitigate potential risks."
                )
            else:
                insights.append(
                    f"The average GDP growth of {gdp_growth_mean:.2f}% reflects a stable economic environment. Focus on growth opportunities in resilient sectors."
                )
            if unemployment_rate_mean > 50:
                insights.append(
                    f"The average unemployment rate of {unemployment_rate_mean:.2f}% is abnormally high, which may indicate input anomalies. Verify data for accuracy."
                )
            elif unemployment_rate_mean > 7:
                insights.append(
                    "High average unemployment rates could increase default probabilities. Monitor sectors with high unemployment sensitivity."
                )
            elif unemployment_rate_mean > 4:
                insights.append(
                    "Moderate unemployment rates may pose risks to consumer credit portfolios. Strengthen risk assessment for personal loans."
                )
            else:
                insights.append(
                    f"The average unemployment rate of {unemployment_rate_mean:.2f}% indicates a stable labor market, reducing default risks. Explore opportunities for lending expansion."
                )
            if interest_rate_mean > 50:
                insights.append(
                    f"The average interest rate of {interest_rate_mean:.2f}% is abnormally high, likely signaling input errors. Verify and correct data."
                )
            elif interest_rate_mean > 5:
                insights.append(
                    "High average interest rates could increase borrowing costs, impacting repayment abilities. Adjust lending strategies accordingly."
                )
            elif interest_rate_mean > 2:
                insights.append(
                    "Moderate interest rates may create mixed impacts. Optimize portfolio allocations to balance risk and return."
                )
            else:
                insights.append(
                    f"The average interest rate of {interest_rate_mean:.2f}% provides favorable borrowing conditions. Consider expanding credit offerings to qualified borrowers."
                )
            if stressed_lgd_mean > 1 or stressed_pd_mean > 1:
                insights.append(
                    "One or more stress test parameters (LGD or PD) exceed 100%, indicating potential anomalies. Verify model outputs and input assumptions."
                )
            elif aggregated_expected_loss > 100:
                insights.append(
                    f"The expected loss of {aggregated_expected_loss:.2f} million is significantly high. Immediate action is required to restructure risky assets and bolster reserves."
                )
            elif aggregated_expected_loss > 50:
                insights.append(
                    f"The expected loss of {aggregated_expected_loss:.2f} million indicates moderate risks. Strengthen monitoring of underperforming sectors."
                )
            else:
                insights.append(
                    f"The expected loss of {aggregated_expected_loss:.2f} million reflects strong portfolio resilience. Maintain current strategies while exploring growth opportunities."
                )
            st.markdown("### Automated Insights")
            for i, insight in enumerate(insights, start=1):
                st.write(f"**Insight {i}:** {insight}")
            st.subheader("Detailed Analysis of Expected Loss")
            st.write(f"The expected loss of **{aggregated_expected_loss:.2f} million** is derived from the stressed parameters:")
            st.write(f"- **Stressed EAD (Exposure at Default):** {stressed_ead_mean:.2f} million")
            st.write(f"- **Stressed LGD (Loss Given Default):** {stressed_lgd_mean:.2%}")
            st.write(f"- **Stressed PD (Probability of Default):** {stressed_pd_mean:.2%}")
            st.write("""
                The expected loss represents the potential financial impact on the portfolio under the selected economic scenario. 
                It is crucial to monitor these parameters and take proactive measures to mitigate risks.
            """)
            st.markdown(
                """
                **Note:**  
                The Automated Insights are generated based on the aggregated results of the stress test analysis.  
                These insights aim to assist decision-makers in identifying risks, optimizing portfolios, and leveraging growth opportunities in a dynamic economic environment.  
                """
            )
        elif result_type == "Detailed":
            try:
                stressed_eads = selected_scenario[gdp_growth_var].apply(forecast_ead)
                stressed_lgds = selected_scenario[unemployment_rate_var].apply(forecast_lgd)
                stressed_pds = selected_scenario[interest_rate_var].apply(forecast_pd)
                expected_losses = stressed_eads * stressed_lgds * stressed_pds
                results_df = pd.DataFrame({
                    "Scenario": [scenario_name] * len(expected_losses),
                    "GDP Growth": selected_scenario[gdp_growth_var],
                    "Unemployment Rate": selected_scenario[unemployment_rate_var],
                    "Interest Rate": selected_scenario[interest_rate_var],
                    "Stressed EAD": stressed_eads,
                    "Stressed LGD": stressed_lgds,
                    "Stressed PD": stressed_pds,
                    "Expected Loss": expected_losses
                })
                def color_code(val):
                    if val > 100:
                        color = 'red'
                    elif val > 50:
                        color = 'orange'
                    else:
                        color = 'green'
                    return f'background-color: {color}'
                st.subheader("Detailed Stress Test Results")
                st.dataframe(results_df.style.applymap(color_code, subset=['Expected Loss']))
                if st.button("Show Expected Loss Plot"):
                    st.subheader("Expected Loss Visualization")
                    plt.figure(figsize=(10, 6))
                    plt.plot(results_df.index, expected_losses, label='Monthly Expected Loss', marker='o', color='blue')
                    plt.axhline(y=aggregated_expected_loss, color='red', linestyle='--', label=f'Baseline Risk Line: {aggregated_expected_loss:.2f} million')
                    plt.xlabel('Month')
                    plt.ylabel('Expected Loss (in millions)')
                    plt.title('Monthly Expected Loss vs. Average Expected Loss')
                    plt.legend()
                    plt.grid(True)
                    underestimation = results_df[results_df["Expected Loss"] > aggregated_expected_loss]
                    overestimation = results_df[results_df["Expected Loss"] < aggregated_expected_loss]
                    plt.fill_between(results_df.index, expected_losses, aggregated_expected_loss, where=(expected_losses > aggregated_expected_loss), color='red', alpha=0.3, label='Risk Underestimation Zone / High-Risk Zone')
                    plt.fill_between(results_df.index, expected_losses, aggregated_expected_loss, where=(expected_losses < aggregated_expected_loss), color='green', alpha=0.3, label='Risk Overestimation Zone / Low-Risk Zone')
                    plt.text(results_df.index[-1], aggregated_expected_loss + 5, 'Risk Underestimation Zone / High-Risk Zone', color='black', fontsize=10, verticalalignment='bottom', horizontalalignment='right')
                    plt.text(results_df.index[-16], aggregated_expected_loss - 5, 'Risk Overestimation Zone / Low-Risk Zone', color='black', fontsize=10, verticalalignment='top', horizontalalignment='right')
                    st.pyplot(plt)
                    st.write("**Risk Underestimation Zone / High-Risk Zone:**")
                    st.dataframe(underestimation)
                    st.write("""
                        These areas indicate periods where the aggregated/average expected loss underestimates the actual risk faced by the portfolio due to adverse economic conditions.
                        Implication:
                        The portfolio is exposed to heightened risk during these periods, and relying solely on average values may give a false sense of stability. Stress testing based on extreme scenarios is critical in these areas to ensure sufficient capital buffers.
                    """)
                    st.write("**Risk Overestimation Zone / Low-Risk Zone:**")
                    st.dataframe(overestimation)
                    st.write("""
                        These areas represent periods where the aggregated/average expected loss overestimates the actual risk, indicating more favorable economic conditions than the average scenario.
                        Implication:
                        The portfolio is performing better than expected, and there may be an opportunity to adjust strategies to increase returns or reduce overly conservative risk buffers.
                    """)
                st.subheader("Statistical Comparison")
                mean_monthly_loss = expected_losses.mean()
                std_monthly_loss = expected_losses.std()
                max_monthly_loss = expected_losses.max()
                min_monthly_loss = expected_losses.min()
                ratio = aggregated_expected_loss / mean_monthly_loss
                st.write(f"**Mean of Monthly Losses:** {mean_monthly_loss:.2f} million")
                st.write(f"**Standard Deviation of Monthly Losses:** {std_monthly_loss:.2f} million")
                st.write(f"**Max Monthly Loss:** {max_monthly_loss:.2f} million")
                st.write(f"**Min Monthly Loss:** {min_monthly_loss:.2f} million")
                st.write(f"**Ratio (Aggregated Loss / Mean of Monthly Losses):** {ratio:.2f}")
                st.subheader("Insights and Recommendations")
                st.write("### Risk Volatility")
                if std_monthly_loss > mean_monthly_loss:
                    st.write("""
                        The standard deviation of the monthly losses is high, indicating significant variability in the portfolio's risk exposure. 
                        This suggests that simply using average values may underestimate or overestimate risks in certain periods.
                    """)
                else:
                    st.write("""
                        The standard deviation of the monthly losses is moderate, indicating manageable variability in the portfolio's risk exposure.
                    """)
                st.write("### Extreme Scenarios")
                if max_monthly_loss > aggregated_expected_loss:
                    st.write("""
                        The maximum expected loss significantly exceeds the aggregated loss, underscoring the need for scenario-specific stress tests to capture tail risks.
                    """)
                else:
                    st.write("""
                        The maximum expected loss is within the range of the aggregated loss, indicating that the portfolio is relatively stable under extreme conditions.
                    """)
                st.write("### Stability vs. Sensitivity")
                if ratio == 1:
                    st.write("""
                        The ratio between aggregated loss and the average of monthly losses is approximately 1, indicating that the portfolio behaves consistently, and aggregated loss is a good proxy for average risk.
                    """)
                elif ratio > 1:
                    st.write("""
                        The ratio between aggregated loss and the average of monthly losses is greater than 1, indicating that the portfolio is less sensitive to economic fluctuations, and aggregated loss overstates typical monthly losses.
                    """)
                else:
                    st.write("""
                        The ratio between aggregated loss and the average of monthly losses is less than 1, indicating that the portfolio is highly sensitive to specific periods of economic stress, and aggregated loss underestimates potential risks.
                    """)
                st.write("### Strategic Implications")
                st.write("""
                    - If volatility is high, risk managers may need to prepare for periods of elevated risk by increasing capital buffers or adjusting portfolio allocations.
                    - If certain periods (e.g., recessions) consistently show high losses, it may indicate that specific macroeconomic factors drive risk more than others, guiding targeted risk mitigation strategies.
                """)
                high_risk_periods = results_df[results_df["Expected Loss"] > 100]
                moderate_risk_periods = results_df[(results_df["Expected Loss"] > 50) & (results_df["Expected Loss"] <= 100)]
                low_risk_periods = results_df[results_df["Expected Loss"] <= 50]
                if not high_risk_periods.empty:
                    st.write("**High Risk Periods:**")
                    st.dataframe(high_risk_periods)
                    st.write("""
                        Significant expected loss predicted during these periods. Recommendations:
                        - Increase capital reserves to cover potential losses.
                        - Reduce exposure to high-risk sectors.
                        - Enhance credit risk monitoring and management.
                        - Consider hedging strategies to mitigate potential losses.
                    """)
                if not moderate_risk_periods.empty:
                    st.write("**Moderate Risk Periods:**")
                    st.dataframe(moderate_risk_periods)
                    st.write("""
                        Moderate expected loss predicted during these periods. Recommendations:
                        - Monitor credit risk closely and adjust lending strategies.
                        - Diversify the portfolio to spread risk.
                        - Strengthen risk assessment processes for new loans.
                    """)
                if not low_risk_periods.empty:
                    st.write("**Low Risk Periods:**")
                    st.dataframe(low_risk_periods)
                    st.write("""
                        Low expected loss predicted during these periods. Recommendations:
                        - Maintain current strategies and explore growth opportunities.
                        - Consider expanding credit offerings to qualified borrowers.
                        - Continue to monitor economic conditions and adjust strategies as needed.
                    """)
            except Exception as e:
                st.error(f"Error in detailed stress test calculations: {e}")
    else:
        st.warning("No economic scenarios available. Please complete the Scenario Forecast step.")


###############################
####### Stress Testing 2
###############################

elif selected_tab == "Stress Testing 2":
    st.title("💥 Stress Testing 2")
    st.subheader("Bank Assets or Portfolio")
    total_assets = st.number_input("Enter Total Assets (in millions)", value=1000.0, step=0.1, key="total_assets_2")
    ead = st.number_input("Enter Exposure at Default (EAD) (in millions)", value=500.0, step=0.1, key="ead_2")
    
    # Always remind users of the best model from model selection.
    if "valid_models" in st.session_state and st.session_state["valid_models"]:
        valid_models = st.session_state["valid_models"]
        best_model_name = min(valid_models, key=lambda m: valid_models[m].get("RMSE", float("inf")))
        st.write(f"**Recommended Model:** {best_model_name}")
        
        selected_model_name = st.selectbox("Select a Model for Stress Testing", 
                                             options=list(valid_models.keys()), key="st2_model_select")
        
        if "scenario_data" in st.session_state:
            scenario_data = st.session_state["scenario_data"]
            st.subheader("Select Economic Scenario")
            scenario_name = st.selectbox("Select Scenario", options=list(scenario_data.keys()), key="st2_scenario_select")
            selected_scenario = scenario_data[scenario_name]
            st.subheader("Select Variables for Forecasting")
            available_variables = list(selected_scenario.columns)
            gdp_growth_var = st.selectbox("Select Variable for GDP Growth", options=available_variables, key="st2_gdp")
            unemployment_rate_var = st.selectbox("Select Variable for Unemployment Rate", options=available_variables, key="st2_unemp")
            interest_rate_var = st.selectbox("Select Variable for Interest Rate", options=available_variables, key="st2_ir")
            
            # Build input data from the scenario data and ensure all selected features are present.
            input_data = selected_scenario[[gdp_growth_var, unemployment_rate_var, interest_rate_var]].copy()
            for feature in st.session_state["selected_features"]:
                if feature not in input_data.columns:
                    input_data[feature] = st.session_state["data"][feature].mean()
            input_data = input_data[st.session_state["selected_features"]]
            
            # Initialize forecasts variable.
            forecasts = None
            
            # SARIMAX branch
            if selected_model_name == "SARIMAX":
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                model = SARIMAX(st.session_state["data"][st.session_state["target_var"]],
                                exog=st.session_state["data"][st.session_state["selected_features"]],
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12)).fit(disp=False)
                forecasts = model.get_forecast(steps=len(input_data), exog=input_data).predicted_mean

            # OLS branch
            elif selected_model_name == "OLS":
                from statsmodels.api import OLS, add_constant
                X_const = add_constant(st.session_state["data"][st.session_state["selected_features"]])
                model = OLS(st.session_state["data"][st.session_state["target_var"]], X_const).fit()
                input_const = add_constant(input_data, has_constant="add")
                forecasts = model.predict(input_const)

            # Random Forest branch
            elif selected_model_name == "Random Forest":
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(random_state=42)
                model.fit(st.session_state["data"][st.session_state["selected_features"]],
                          st.session_state["data"][st.session_state["target_var"]])
                forecasts = model.predict(input_data)

            # Gradient Boosting branch
            elif selected_model_name == "Gradient Boosting":
                from sklearn.ensemble import GradientBoostingRegressor
                model = GradientBoostingRegressor(random_state=42)
                model.fit(st.session_state["data"][st.session_state["selected_features"]],
                          st.session_state["data"][st.session_state["target_var"]])
                forecasts = model.predict(input_data)

            # XGBoost branch
            elif selected_model_name == "XGBoost":
                import xgboost as xgb
                model = xgb.XGBRegressor(random_state=42)
                model.fit(st.session_state["data"][st.session_state["selected_features"]],
                          st.session_state["data"][st.session_state["target_var"]])
                forecasts = model.predict(input_data)

            # LightGBM branch
            elif selected_model_name == "LightGBM":
                import lightgbm as lgb
                model = lgb.LGBMRegressor(random_state=42)
                model.fit(st.session_state["data"][st.session_state["selected_features"]],
                          st.session_state["data"][st.session_state["target_var"]])
                forecasts = model.predict(input_data)

            # SVR branch
            elif selected_model_name == "SVR":
                from sklearn.svm import SVR
                model = SVR()
                model.fit(st.session_state["data"][st.session_state["selected_features"]],
                          st.session_state["data"][st.session_state["target_var"]])
                forecasts = model.predict(input_data)

            # LASSO branch
            elif selected_model_name == "LASSO":
                from sklearn.linear_model import Lasso
                model = Lasso(alpha=st.session_state.get("lasso_alpha", 0.01), random_state=42)
                model.fit(st.session_state["data"][st.session_state["selected_features"]],
                          st.session_state["data"][st.session_state["target_var"]])
                forecasts = model.predict(input_data)

            # Ridge branch
            elif selected_model_name == "Ridge":
                from sklearn.linear_model import Ridge
                model = Ridge(alpha=st.session_state.get("ridge_alpha", 0.01), random_state=42)
                model.fit(st.session_state["data"][st.session_state["selected_features"]],
                          st.session_state["data"][st.session_state["target_var"]])
                forecasts = model.predict(input_data)

            # Elastic Net branch
            elif selected_model_name == "Elastic Net":
                from sklearn.linear_model import ElasticNet
                model = ElasticNet(alpha=st.session_state.get("elasticnet_alpha", 0.01),
                                   l1_ratio=st.session_state.get("elasticnet_l1_ratio", 0.5),
                                   random_state=42)
                model.fit(st.session_state["data"][st.session_state["selected_features"]],
                          st.session_state["data"][st.session_state["target_var"]])
                forecasts = model.predict(input_data)

            # ECM branch – forecasting using error correction model
            elif selected_model_name == "ECM":
                import statsmodels.api as sm
                data_levels = st.session_state["data"].copy()
                selected_features = st.session_state["selected_features"]
                target = st.session_state["target_var"]
                # Estimate cointegration regression using levels
                X_ecm = sm.add_constant(data_levels[selected_features])
                coint_model = sm.OLS(data_levels[target], X_ecm).fit()
                data_levels["ECT"] = data_levels[target] - X_ecm.dot(coint_model.params)
                # Compute first differences for target and selected features
                data_diff = data_levels.diff().dropna()
                # Include lagged ECT (lag 1)
                data_diff["lagECT"] = data_levels["ECT"].shift(1).loc[data_diff.index]
                # Include one lag of differences for each selected feature
                for feat in selected_features:
                    data_diff["lag_d_" + feat] = data_diff[feat].shift(1)
                data_diff = data_diff.dropna()
                # Fit ECM on all available differenced data
                X_ecm_reg = sm.add_constant(data_diff[["lagECT"] + selected_features + ["lag_d_" + feat for feat in selected_features]])
                y_ecm_reg = data_levels[target].diff().loc[data_diff.index]
                ecm_model = sm.OLS(y_ecm_reg, X_ecm_reg).fit()
                forecasts_list = []
                for i in range(len(input_data)):
                    row = input_data.iloc[i]
                    last_obs = data_levels.iloc[-1]
                    new_row = {}
                    new_row["lagECT"] = last_obs["ECT"]
                    for feat in selected_features:
                        diff_val = row[feat] - last_obs[feat]
                        new_row[feat] = diff_val
                        new_row["lag_d_" + feat] = diff_val
                    new_df = pd.DataFrame([new_row])
                    new_df = sm.add_constant(new_df, has_constant="add")
                    # Reindex to match the ECM training matrix
                    new_df = new_df.reindex(columns=X_ecm_reg.columns, fill_value=0)
                    pred_diff = ecm_model.predict(new_df)[0]
                    forecast = last_obs[target] + pred_diff
                    forecasts_list.append(forecast)
                forecasts = np.array(forecasts_list)

            # VECM branch – forecasting with blending
            elif selected_model_name == "VECM":
                from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
                jres = coint_johansen(st.session_state["data"], det_order=0, k_ar_diff=1)
                crit = jres.cvt[:, 1]
                cointegration_rank = sum(jres.lr1 > crit)
                if cointegration_rank > 0:
                    vecm_blend = st.slider("Blend Weight (VECM)", min_value=0.0, max_value=1.0,
                                            value=0.5, step=0.01, key="vecm_blend_st2")
                    new_data = st.session_state["data"].copy()
                    forecasts_list = []
                    for i in range(len(input_data)):
                        scenario_input = input_data.iloc[i].reindex(new_data.columns).fillna(new_data.mean()).values
                        blended = (1 - vecm_blend) * new_data.iloc[-1].values + vecm_blend * scenario_input
                        new_data.iloc[-1] = blended
                        vecm_model = VECM(new_data, k_ar_diff=1, coint_rank=cointegration_rank)
                        vecm_fit = vecm_model.fit()
                        fc = vecm_fit.predict(steps=1)
                        target_col = st.session_state["target_var"]
                        forecasts_list.append(fc[0, new_data.columns.get_loc(target_col)])
                    forecasts = np.array(forecasts_list)

            # VAR branch
            elif selected_model_name == "VAR":
                from statsmodels.tsa.vector_ar.var_model import VAR
                var_model = VAR(st.session_state["data"])
                var_results = var_model.fit(maxlags=2, ic='aic')
                p = var_results.k_ar
                # Use blending: combine the last observed value with the scenario input.
                historical_last = st.session_state["data"].values[-p:].copy()
                scenario_input = input_data.reindex(columns=st.session_state["data"].columns)
                scenario_input = scenario_input.fillna(pd.DataFrame(st.session_state["data"]).mean()).values[0]
                blend_var = st.slider("Blend Weight (VAR)", min_value=0.0, max_value=1.0,
                                      value=0.5, step=0.01, key="blend_var_scen")
                blended = (1 - blend_var) * historical_last[-1] + blend_var * scenario_input
                historical_last[-1] = blended
                forecast = var_results.forecast(historical_last, steps=len(input_data))
                forecasts = forecast[:, 0]

            # VARX branch
            elif selected_model_name == "VARX":
                import pandas as pd
                data_df = st.session_state["data"]
                target_var = st.session_state["target_var"]
                endog = pd.concat([data_df[target_var], data_df[target_var].shift(1)], axis=1).dropna()
                endog.columns = [target_var, target_var + "_lag1"]
                exog = data_df[st.session_state["selected_features"]].loc[endog.index]
                from statsmodels.tsa.statespace.varmax import VARMAX
                varx_model = VARMAX(endog, exog=exog, order=(2, 0)).fit(disp=False)
                forecasts_list = []
                for i in range(len(input_data)):
                    scenario_exog = input_data.iloc[i:i+1]
                    fc = varx_model.forecast(steps=1, exog=scenario_exog)
                    forecasts_list.append(fc.iloc[0, 0])
                forecasts = np.array(forecasts_list)
            else:
                forecasts = None
            
            if forecasts is not None:
                st.write("Forecasts:", forecasts)
                expected_losses = ead * forecasts
                results_df = pd.DataFrame({
                    "Scenario": [scenario_name] * len(forecasts),
                    "Predicted Loss": forecasts,
                    "Expected Loss": expected_losses
                })
            else:
                results_df = pd.DataFrame()
                
            def color_code(val):
                if val > 100:
                    color = 'red'
                elif val > 50:
                    color = 'orange'
                else:
                    color = 'green'
                return f'background-color: {color}'
            
            st.subheader("Stress Test Results")
            st.dataframe(results_df.style.applymap(color_code, subset=['Expected Loss']))
            st.subheader("Insights and Recommendations")
            high_risk_periods = results_df[results_df["Expected Loss"] > 100]
            moderate_risk_periods = results_df[(results_df["Expected Loss"] > 50) & (results_df["Expected Loss"] <= 100)]
            low_risk_periods = results_df[results_df["Expected Loss"] <= 50]
            if not high_risk_periods.empty:
                st.write("**High Risk Periods:**")
                st.dataframe(high_risk_periods)
                st.write("""
                    Significant expected loss predicted during these periods. Recommendations:
                    - Increase capital reserves to cover potential losses.
                    - Reduce exposure to high-risk sectors.
                    - Enhance credit risk monitoring and management.
                    - Consider hedging strategies to mitigate potential losses.
                """)
            if not moderate_risk_periods.empty:
                st.write("**Moderate Risk Periods:**")
                st.dataframe(moderate_risk_periods)
                st.write("""
                    Moderate expected loss predicted during these periods. Recommendations:
                    - Monitor credit risk closely and adjust lending strategies.
                    - Diversify the portfolio to spread risk.
                    - Strengthen risk assessment processes for new loans.
                """)
            if not low_risk_periods.empty:
                st.write("**Low Risk Periods:**")
                st.dataframe(low_risk_periods)
                st.write("""
                    Low expected loss predicted during these periods. Recommendations:
                    - Maintain current strategies and explore growth opportunities.
                    - Consider expanding credit offerings to qualified borrowers.
                    - Continue to monitor economic conditions and adjust strategies as needed.
                """)
        else:
            st.warning("No economic scenarios available. Please complete the Scenario Forecast step.")
    else:
        st.warning("No valid models available. Please complete the Model Selection step.")






#################################
###### Stress Testing 3
##################################

elif selected_tab == "Stress Testing 3":
    st.title("💥 Stress Testing 3")
    st.subheader("Bank Assets or Portfolio")
    total_assets = st.number_input("Enter Total Assets (in millions)", value=1000.0, step=0.1, key="total_assets_3")
    lgd = st.number_input("Enter Loss Given Default (LGD) (as a percentage)", value=40.0, step=0.1, key="lgd_3")/100
    probability_default = st.number_input("Enter Probability of Default (PD) (as a percentage)", value=5.0, step=0.1, key="pd_3")/100
    if "valid_models" in st.session_state and st.session_state["valid_models"]:
        valid_models = st.session_state["valid_models"]
        filtered_models = {k: v for k, v in valid_models.items() if k not in ["Hybrid", "Ensemble"]}
        if filtered_models:
            best_model_name = min(filtered_models, key=lambda m: filtered_models[m].get("RMSE", float("inf")))
            st.write(f"**Recommended Model:** {best_model_name}")
            selected_model_name = st.selectbox("Select a Model for Stress Testing", 
                                                 options=list(filtered_models.keys()), key="stress_testing3_model")
            
            if "scenario_data" in st.session_state:
                scenario_data = st.session_state["scenario_data"]
                st.subheader("Select Economic Scenario")
                scenario_name = st.selectbox("Select Scenario", options=list(scenario_data.keys()), key="st3_scenario")
                selected_scenario = scenario_data[scenario_name]
                st.subheader("Select Variables for Forecasting")
                available_variables = list(selected_scenario.columns)
                gdp_growth_var = st.selectbox("Select Variable for GDP Growth", options=available_variables, key="st3_gdp")
                unemployment_rate_var = st.selectbox("Select Variable for Unemployment Rate", options=available_variables, key="st3_unemp")
                interest_rate_var = st.selectbox("Select Variable for Interest Rate", options=available_variables, key="st3_ir")
                input_data = selected_scenario[[gdp_growth_var, unemployment_rate_var, interest_rate_var]].copy()
                for feature in st.session_state["selected_features"]:
                    if feature not in input_data.columns:
                        input_data[feature] = st.session_state["data"][feature].mean()
                input_data = input_data[st.session_state["selected_features"]]
                
                forecasts = None
                if selected_model_name == "VAR":
                    from statsmodels.tsa.vector_ar.var_model import VAR
                    var_model = VAR(st.session_state["data"])
                    var_results = var_model.fit(maxlags=2)
                    forecast = var_results.forecast(st.session_state["data"].values[-var_results.k_ar:], steps=len(input_data))
                    forecasts = forecast[:, 0]
                elif selected_model_name in ["VECM", "ECM"]:
                    from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
                    jres = coint_johansen(st.session_state["data"], det_order=0, k_ar_diff=1)
                    crit = jres.cvt[:,1]
                    cointegration_rank = sum(jres.lr1 > crit)
                    if cointegration_rank > 0:
                        vecm_blend = st.slider("Blend Weight (VECM)", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="vecm_blend_st3")
                        new_data = st.session_state["data"].copy()
                        forecasts_list = []
                        for i in range(len(input_data)):
                            # Reindex the Series without the 'columns' keyword.
                            scenario_input = input_data.iloc[i].reindex(new_data.columns)
                            scenario_input = scenario_input.fillna(new_data.mean()).values
                            blended = (1 - vecm_blend) * new_data.iloc[-1].values + vecm_blend * scenario_input
                            new_data.iloc[-1] = blended
                            vecm_model = VECM(new_data, k_ar_diff=1, coint_rank=cointegration_rank)
                            vecm_fit = vecm_model.fit()
                            fc = vecm_fit.predict(steps=1)
                            forecasts_list.append(fc[0,0])
                        forecasts = np.array(forecasts_list)
                    else:
                        forecasts = None
                        st.warning("No cointegration found for VECM/ECM.")
                elif selected_model_name == "SARIMAX":
                    from statsmodels.tsa.statespace.sarimax import SARIMAX
                    model = SARIMAX(
                        st.session_state["data"][st.session_state["target_var"]],
                        exog=st.session_state["data"][st.session_state["selected_features"]],
                        order=(1, 1, 1),
                        seasonal_order=(1, 1, 1, 12)
                    ).fit(disp=False)
                    forecasts = model.get_forecast(steps=len(input_data), exog=input_data).predicted_mean
                elif selected_model_name == "OLS":
                    from statsmodels.api import OLS, add_constant
                    X_const = add_constant(st.session_state["data"][st.session_state["selected_features"]])
                    model = OLS(st.session_state["data"][st.session_state["target_var"]], X_const).fit()
                    input_const = add_constant(input_data, has_constant="add")
                    forecasts = model.predict(input_const)
                # New VARX branch
                elif selected_model_name == "VARX":
                    import pandas as pd
                    data_df = st.session_state["data"]
                    target_var = st.session_state["target_var"]
                    # Create a multivariate endogenous system: target variable and its first lag.
                    endog = pd.concat([data_df[target_var], data_df[target_var].shift(1)], axis=1).dropna()
                    endog.columns = [target_var, target_var + "_lag1"]
                    # Align exogenous variables with the new index.
                    exog = data_df[st.session_state["selected_features"]].loc[endog.index]
                    from statsmodels.tsa.statespace.varmax import VARMAX
                    varx_model = VARMAX(endog, exog=exog, order=(2, 0)).fit(disp=False)
                    forecasts_list = []
                    for i in range(len(input_data)):
                        scenario_exog = input_data.iloc[i:i+1]
                        fc = varx_model.forecast(steps=1, exog=scenario_exog)
                        # Extract the forecast for the target variable (first column)
                        forecasts_list.append(fc.iloc[0, 0])
                    forecasts = np.array(forecasts_list)
                else:
                    forecasts = None
                
                if forecasts is not None:
                    st.write("Forecasts:", forecasts)
                    expected_losses = forecasts * lgd * probability_default
                    results_df = pd.DataFrame({
                        "Scenario": [scenario_name] * len(forecasts),
                        "Predicted Loss": forecasts,
                        "Expected Loss": expected_losses
                    })
                else:
                    results_df = pd.DataFrame()
            else:
                st.warning("No economic scenarios available. Please complete the Scenario Forecast step.")
            
            def color_code(val):
                if val > 100:
                    color = 'red'
                elif val > 50:
                    color = 'orange'
                else:
                    color = 'green'
                return f'background-color: {color}'
            
            st.subheader("Stress Test Results")
            st.dataframe(results_df.style.applymap(color_code, subset=['Expected Loss']))
            st.subheader("Insights and Recommendations")
            high_risk_periods = results_df[results_df["Expected Loss"] > 100]
            moderate_risk_periods = results_df[(results_df["Expected Loss"] > 50) & (results_df["Expected Loss"] <= 100)]
            low_risk_periods = results_df[results_df["Expected Loss"] <= 50]
            if not high_risk_periods.empty:
                st.write("**High Risk Periods:**")
                st.dataframe(high_risk_periods)
                st.write("""
                    Significant expected loss predicted during these periods. Recommendations:
                    - Increase capital reserves to cover potential losses.
                    - Reduce exposure to high-risk sectors.
                    - Enhance credit risk monitoring and management.
                    - Consider hedging strategies to mitigate potential losses.
                """)
            if not moderate_risk_periods.empty:
                st.write("**Moderate Risk Periods:**")
                st.dataframe(moderate_risk_periods)
                st.write("""
                    Moderate expected loss predicted during these periods. Recommendations:
                    - Monitor credit risk closely and adjust lending strategies.
                    - Diversify the portfolio to spread risk.
                    - Strengthen risk assessment processes for new loans.
                """)
            if not low_risk_periods.empty:
                st.write("**Low Risk Periods:**")
                st.dataframe(low_risk_periods)
                st.write("""
                    Low expected loss predicted during these periods. Recommendations:
                    - Maintain current strategies and explore growth opportunities.
                    - Consider expanding credit offerings to qualified borrowers.
                    - Continue to monitor economic conditions and adjust strategies as needed.
                """)
        else:
            st.warning("No valid economic scenarios available. Please complete the Scenario Forecast step.")
    else:
        st.warning("No valid models available. Please complete the Model Selection step.")

