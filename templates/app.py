from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, acf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
import os

app = Flask(__name__)

# Global variables
data = None
stationary_data = None
non_stationary_data = None
target_transformed_name = None
column_mapping = {}
exog_vars = []  # Define exog_vars globally

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Function to make data stationary
def calculate_aic(n, mse, num_params):
    """
    Calculate Akaike Information Criterion (AIC).
    n: Number of observations
    mse: Mean Squared Error
    num_params: Number of parameters in the model
    """
    return n * np.log(mse) + 2 * num_params


# Function to detect seasonality
def is_seasonal(data, threshold=0.2, seasonal_period=12):
    acf_values = acf(data, fft=True, nlags=seasonal_period * 2)
    seasonal_acf = acf_values[seasonal_period::seasonal_period]
    print("ACF Values:", acf_values)  # Debugging output
    print("Seasonal ACF Values:", seasonal_acf)  # Debugging output
    is_seasonal = any(abs(value) > threshold for value in seasonal_acf)
    return is_seasonal, acf_values

def make_stationary(data, target=None, significance_level=0.05):
    stationary_data = pd.DataFrame(index=data.index)
    stationarity_results = []
    target_transformed_name = target if target else None
    column_mapping = {}

    for column in data.columns:
        series = data[column]
        diff_count = 0
        while adfuller(series.dropna())[1] >= significance_level:
            series = series.diff().dropna()
            diff_count += 1
        column_name = f"{column}" if diff_count == 0 else f"{column}_diff{diff_count}"

        stationary_data[column_name] = series
        column_mapping[column] = column_name

        if column == target:
            target_transformed_name = column_name

        stationarity_results.append({
            'Variable': column,
            'ADF Statistic': adfuller(series.dropna())[0],
            'p-value': adfuller(series.dropna())[1],
            'Stationary': adfuller(series.dropna())[1] < significance_level,
            'Transformations': 'None' if diff_count == 0 else f"Differenced {diff_count} times",
            'Transformed Variable': column_name
        })

    stationary_data.dropna(inplace=True)
    return pd.DataFrame(stationarity_results), stationary_data, target_transformed_name, column_mapping

# Home route to upload data
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files['file']
        if file and file.filename.endswith('.xlsx'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            global data, non_stationary_data
            data = pd.read_excel(filepath)
            non_stationary_data = data.copy()

            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                data.set_index('date', inplace=True)

            return render_template("target_selection.html", columns=data.columns, data_preview=data.head().to_html())
    
    return render_template("index.html")

# Route to check stationarity
@app.route("/stationarity", methods=["GET", "POST"])
def stationarity():
    global data, target, stationary_data, target_transformed_name, exog_vars, column_mapping
    target = request.form.get("target")
    exog_vars = request.form.getlist("exog_vars")

    # Perform stationarity and seasonality checks
    stationarity_results, stationary_data, target_transformed_name, column_mapping = make_stationary(data, target)
    seasonal, _ = is_seasonal(data[target])  # Assuming `is_seasonal` checks seasonality
    seasonality_message = "Seasonality detected. SARIMAX might be a better model." if seasonal else "No significant seasonality detected."

    return render_template(
        "stationarity.html", 
        stationarity_results=stationarity_results.to_html(), 
        stationary_data_preview=stationary_data.head().to_html(),
        seasonality_message=seasonality_message
    )


# Route for model selection with options for stationary and non-stationary data
@app.route("/model_selection", methods=["POST"])
def model_selection():
    global stationary_data, non_stationary_data, target_transformed_name, exog_vars, column_mapping, target

    # Get data type (stationary or non-stationary) from user selection
    data_type = request.form.get("data_type")

    # Check if target or exog_vars are missing
    if data_type == "stationary":
        if target_transformed_name is None:
            return "Error: Target variable not defined or selected in stationary data.", 400
        if stationary_data is None or stationary_data.empty:
            return "Error: Stationary data not found or is empty. Ensure data is processed.", 400
    else:  # Non-stationary data
        if target is None:
            return "Error: Target variable not defined or selected in non-stationary data.", 400
        if non_stationary_data is None or non_stationary_data.empty:
            return "Error: Non-stationary data not found or is empty. Ensure data is uploaded.", 400

    # Select the correct dataset and variables
    if data_type == "stationary":
        data_used = stationary_data
        target_column = target_transformed_name
        transformed_exog_vars = [column_mapping[var] for var in exog_vars if var in column_mapping]
        data_type_message = "Using Stationary Data for Model Selection"
    else:
        data_used = non_stationary_data
        target_column = target
        transformed_exog_vars = exog_vars
        data_type_message = "Using Non-Stationary Data for Model Selection"

    # Handle missing data or errors in the dataset
    if data_used is None or data_used.empty:
        return "Error: Data not found or is empty. Ensure data is uploaded and processed.", 400
    if target_column not in data_used.columns:
        return f"Error: Target column '{target_column}' not found in the dataset.", 400

    # Train-test split
    train_size = int(len(data_used) * 0.7)
    y_train = data_used[target_column][:train_size]
    y_test = data_used[target_column][train_size:]
    X_train = data_used[transformed_exog_vars][:train_size] if transformed_exog_vars else None
    X_test = data_used[transformed_exog_vars][train_size:] if transformed_exog_vars else None


    sarima_metrics, ols_metrics, var_metrics, vecm_metrics = {}, {}, {}, {}

    # Perform model evaluation (SARIMAX, OLS, VAR, VECM)
    try:
        sarima_model = SARIMAX(y_train, exog=X_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
        sarima_forecast = sarima_model.get_forecast(steps=len(y_test), exog=X_test).predicted_mean
        sarima_metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_test, sarima_forecast)),
            'MAPE': mean_absolute_percentage_error(y_test, sarima_forecast),
            'AIC': sarima_model.aic
        }
    except Exception as e:
        sarima_metrics['Error'] = f"SARIMA Error: {str(e)}"

    try:
        if X_train is not None:
            ols_model = LinearRegression().fit(X_train, y_train)
            ols_forecast = ols_model.predict(X_test)
            ols_rmse = np.sqrt(mean_squared_error(y_test, ols_forecast))
            ols_mape = mean_absolute_percentage_error(y_test, ols_forecast)
            ols_aic = len(y_test) * np.log(((y_test - ols_forecast) ** 2).mean()) + 2 * (X_train.shape[1] + 1)
            ols_metrics = {
                'RMSE': ols_rmse, 
                'MAPE': ols_mape, 
                'AIC': ols_aic
            }
    except Exception as e:
        ols_metrics['Error'] = f"OLS Error: {str(e)}"

    try:
        if len(transformed_exog_vars) >= 2:
            var_data = data_used[transformed_exog_vars + [target_column]].iloc[:train_size]
            var_model = VAR(var_data)
            var_results = var_model.fit(maxlags=2)
            var_forecast = var_results.forecast(y=var_data.values[-var_results.k_ar:], steps=len(y_test))
            var_metrics = {
                'RMSE': np.sqrt(mean_squared_error(y_test, [f[0] for f in var_forecast])),
                'MAPE': mean_absolute_percentage_error(y_test, [f[0] for f in var_forecast]),
                'AIC': var_results.aic
            }
    except Exception as e:
        var_metrics['Error'] = f"VAR Error: {str(e)}"

    try:
        if len(transformed_exog_vars) >= 2:  # VECM requires at least two time series
            vecm_data = data_used[transformed_exog_vars + [target_column]].iloc[:train_size]
            johansen_test = coint_johansen(vecm_data, det_order=0, k_ar_diff=1)
            coint_rank = np.sum(johansen_test.lr1 > johansen_test.cvt[:, 1])  # Cointegration rank at 5% significance

            if coint_rank > 0:
                vecm_model = VECM(vecm_data, coint_rank=coint_rank, k_ar_diff=1)
                vecm_results = vecm_model.fit()
                vecm_forecast = vecm_results.predict(steps=len(y_test))
                vecm_forecast_target = vecm_forecast[:, 0]  # Assuming the target variable is the first column
                vecm_rmse = np.sqrt(mean_squared_error(y_test, vecm_forecast_target))
                vecm_mape = mean_absolute_percentage_error(y_test, vecm_forecast_target)
                num_params = vecm_results.alpha.shape[0] * vecm_results.beta.shape[0]
                vecm_aic = calculate_aic(len(y_test), vecm_rmse**2, num_params)
                vecm_metrics = {
                    'RMSE': vecm_rmse,
                    'MAPE': vecm_mape,
                    'AIC': vecm_aic,
                    'Cointegration Rank': coint_rank
                }
            else:
                vecm_metrics = {'Error': 'No cointegration found in data.'}
        else:
            vecm_metrics = {'Error': 'Insufficient variables for VECM. At least two time series are required.'}
    except Exception as e:
        vecm_metrics['Error'] = f"VECM Error: {str(e)}"

    # Evaluate models and determine the best one
    models = {
        "SARIMA": sarima_metrics,
        "OLS": ols_metrics,
        "VAR": var_metrics,
        "VECM": vecm_metrics
    }

    best_model = None
    lowest_aic = float("inf")
    reason = ""

    for model_name, metrics in models.items():
        if metrics.get("AIC") and metrics["AIC"] < lowest_aic:
            best_model = model_name
            lowest_aic = metrics["AIC"]

    if best_model:
        if best_model == "SARIMA":
            reason = "SARIMA was selected because it achieved the lowest AIC, which is critical for balancing accuracy and model simplicity. Additionally, it handles seasonality effectively, which was detected in the data."
        elif best_model == "OLS":
            reason = "OLS was chosen as it provides a simple yet effective model with competitive RMSE and AIC values compared to other models."
        elif best_model == "VAR":
            reason = "VAR was selected due to its strong ability to model multivariate time series relationships with a relatively low AIC."
        elif best_model == "VECM":
            reason = "VECM was chosen because it best captures the cointegration relationships between variables, as evidenced by the lowest AIC and cointegration rank."

    return render_template(
        "model_selection.html",
        sarima_metrics=sarima_metrics,
        ols_metrics=ols_metrics,
        var_metrics=var_metrics,
        vecm_metrics=vecm_metrics,
        best_model=best_model,
        reason=reason,
        data_type_message=data_type_message
    )

if __name__ == "__main__":
    app.run(debug=True)
