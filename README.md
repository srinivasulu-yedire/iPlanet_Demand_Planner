# iPlanet Demand Planner

iPlanet Demand Planner is a Streamlit demand forecasting and inventory planning app for iPlanet retail sales data. It supports role-based login, an admin settings workspace, model retraining from uploaded sales workbooks, product-level forecasts, safety stock/reorder guidance, holiday context, and CSV exports.

## What The App Does

- Authenticates users from `data/users.json`
- Supports `Admin` and `Store Operator` roles
- Shows Store Operators only the prediction workflow
- Lets Admin users access Prediction, Model Training, and Settings
- Trains demand models from `data/sales_data.xlsx`
- Builds weekly product-store-region demand features from transaction data
- Adds lag, rolling, seasonal, discount, EMI/cashback, funded scheme, and Indian holiday features
- Selects the best model from XGBoost, Random Forest, and HistGradientBoosting variants
- Saves model and encoder artifacts under `models/`
- Runs product-level forecasts with safety stock, reorder point, and procurement guidance
- Exports individual product forecasts and bulk store forecasts as CSV files

## Project Structure

```text
iPlanet_stock_prediction_app/
|-- README.md
|-- requirements.txt
|-- test_genai.py
|-- data/
|   |-- sales_data.xlsx
|   |-- inventory_master_data.xlsx
|   |-- users.json
|   `-- training_uploads/
|-- logs/
|   `-- app.log
|-- models/
|   |-- master_model.pkl
|   |-- model_metrics.pkl
|   |-- le_region.pkl
|   |-- le_store.pkl
|   |-- le_prod.pkl
|   |-- le_product_store.pkl
|   |-- le_month_product.pkl
|   `-- _training_latest/
`-- src/
    |-- streamlit_app.py
    |-- training_pipeline.py
    |-- report_engine.py
    |-- data_engine.py
    |-- external_features.py
    |-- genai_layer.py
    `-- utils/
        |-- __init__.py
        `-- logger.py
```

## Main Files

| File | Purpose |
| --- | --- |
| `src/streamlit_app.py` | Main Streamlit app, login flow, role-based workspaces, prediction controls, training UI, Settings/User Management UI, and dashboard rendering. |
| `src/training_pipeline.py` | Validates sales data, engineers features, runs the model tournament, evaluates models, and writes model artifacts. |
| `src/report_engine.py` | Anchors forecasts to recent historical sales and creates individual and bulk CSV reports. |
| `src/data_engine.py` | Utility functions for data preparation. |
| `src/external_features.py` | Adds Indian holiday features and optional external feature files when available. |
| `src/genai_layer.py` | Placeholder for future GenAI functions. It is currently empty, so GenAI imports fall back gracefully. |
| `test_genai.py` | Smoke test script intended for future GenAI integration. |
| `src/utils/logger.py` | Simple timestamped console logger used by training. |

## Login And Roles

User records are stored in `data/users.json`.

Default admin login:

- User: `admin`
- Password: `admin123`

Roles:

- `Admin`: can access Prediction, Model Training, and Settings.
- `Store Operator`: can access only the Prediction screen. The workspace selector is hidden for this role because there is only one available workspace.

The login screen includes a short transition screen with the message `Taking you to iPlanet software hub...`.

## Settings / User Management

Admin users can open `Settings` from the top workspace selector.

The Settings screen supports:

- Creating users
- Assigning `Admin` or `Store Operator`
- Enforcing unique usernames
- Showing users in a table-style summary
- Changing a user's password from the row-level password reset icon
- Tracking `Last Password Change`

Passwords are stored as SHA-256 hashes in `data/users.json`.

## Data Inputs

Required:

- `data/sales_data.xlsx`

Required training columns:

- `RegionName`
- `StoreName`
- `ProductName`
- `BillDate`
- `Quantity`
- `MRP`
- `ProductLevelDiscAmount`

Optional columns used when present:

- `True Demand`
- `Bank Funded NoCost EMI Applied With BankName`
- `Bank Funded Cashback Coverted To Percentage With BankName`
- `Additional Apple Or Distributor Funded Scheme In Percentage`

Also present:

- `data/inventory_master_data.xlsx`

Optional external feature files supported by `src/external_features.py`:

- `data/economic_indicators.xlsx`
- `data/weather_data.xlsx`
- `data/competitor_data.xlsx`
- `data/digital_marketing.xlsx`
- `data/supply_chain.xlsx`
- `data/bank_offers.xlsx`
- `data/custom_discounts.xlsx`
- `data/custom_promotions.xlsx`

The app works without these optional files. Missing optional files are skipped.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the app:

```powershell
streamlit run src\streamlit_app.py
```

Run commands from the project root so relative paths like `data/sales_data.xlsx` and `models/master_model.pkl` resolve correctly.

## Prediction Workflow

After login, use the sidebar Prediction Controls:

- Region
- Store
- Product
- Forecast Start Date (Historical)
- Planning Horizon: 7, 30, or 60 days
- Supplier Lead Time: 7, 30, or 60 days
- Growth Margin percentage

Click `Run Intelligence Report` to generate the dashboard.

The dashboard includes:

- Historical sales trend
- Demand forecast chart
- Demand total
- Safety stock
- Reorder point
- Daily velocity
- Procurement recommendation
- Technical scorecard
- Holiday calendar
- Individual product CSV download
- Bulk store CSV download
- GenAI tab placeholder, disabled until `src/genai_layer.py` is implemented

## Train Or Refresh The Model

There are two supported training flows.

### Train From The Streamlit UI

Login as an Admin and switch the top workspace selector to `Training`.

The training workspace lets you:

- Upload a sales workbook
- Validate required columns
- Review row, duplicate, date gap, and anomaly summaries
- Choose a retraining strategy:
  - Full retrain
  - Rolling window using the latest 2-3 years
- Click `Train Model`
- Stage generated artifacts in `models/_training_latest/`
- Promote staged artifacts to `models/` only after training succeeds
- Replace the active `data/sales_data.xlsx` with the uploaded training file
- Clear Streamlit caches so predictions use the refreshed model

Uploaded training files are retained under `data/training_uploads/` with timestamps.

### Train From The Command Line

Run from the project root:

```powershell
python src\training_pipeline.py
```

The pipeline:

1. Loads `data/sales_data.xlsx`
2. Validates required columns
3. Aggregates transactions into weekly product-store-region demand
4. Builds enhanced time-series, seasonal, holiday, pricing, and promotion features
5. Trains a model tournament
6. Selects the lowest-MAE model
7. Saves the winning model, label encoders, and metrics to `models/`

Current saved metrics in `models/model_metrics.pkl`:

- Algorithm: `Hist Gradient Boosting Enhanced`
- Forecast Accuracy: `82.17%`
- MAE: `0.5005`
- RMSE: `2.18`
- sMAPE: `83.45%`
- Features Used: `48`

## Model Artifacts

The active prediction workflow expects these files in `models/`:

- `master_model.pkl`
- `model_metrics.pkl`
- `le_region.pkl`
- `le_store.pkl`
- `le_prod.pkl`
- `le_product_store.pkl`
- `le_month_product.pkl`

Training first writes new artifacts to `models/_training_latest/` and promotes them only after all required artifacts exist.

## Feature Engineering

The training pipeline builds features such as:

- Encoded region, store, product, product-store, and month-product IDs
- Week, month, year, quarter, month-start, and month-end flags
- Cyclical week/month sine and cosine features
- Weekly lags for 1, 2, 3, 4, 8, and 12 weeks
- Rolling means and standard deviations for 2, 4, 8, and 12 weeks
- EWMA, month-over-month change, trend, and z-score features
- True demand, EMI usage, cashback percentage, and funded scheme percentage
- Indian holiday indicators and distance-to-holiday features
- Discount percentage, discount intensity, and effective price

## Inventory Logic

The dashboard calculates:

- Total demand over the selected horizon
- Average daily velocity
- Safety stock using a 95% service-level approximation
- Reorder point based on demand during lead time plus safety stock
- Procurement quantity with the selected growth margin applied

## GenAI Status

The app and report engine contain hooks for future GenAI-powered explanations and recommendations, but `src/genai_layer.py` is currently empty.

Until those functions are implemented:

- The app falls back gracefully.
- The AI insights area may show disabled or limited availability messaging.
- `OPENAI_API_KEY` alone is not enough to enable GenAI features.

Expected future functions include:

- `initialize_genai`
- `explain_forecast_accuracy`
- `analyze_forecast_trends`
- `generate_inventory_recommendations`
- `create_executive_summary`
- `detect_anomalies`
- `get_forecast_insights`
- `format_business_alert`

## Notes

- Streamlit's default cache spinner is disabled for metadata loading so the app uses its own progress messaging.
- The top-right Streamlit toolbar/menu is hidden by app CSS for a cleaner operator experience.
- The checked-in `venv/` directory is not required if you create your own virtual environment.
- `logs/app.log` exists, but active training logs are primarily emitted to the console/status UI.
- Some Python labels contain mojibake characters from earlier encoding issues; this README uses plain ASCII for readability.
