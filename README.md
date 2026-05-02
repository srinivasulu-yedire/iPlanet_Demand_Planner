# iPlanet Demand Planner

iPlanet Demand Planner is a Streamlit-based demand forecasting and inventory planning app for iPlanet retail sales data. The project trains an AutoML-style model tournament on historical sales, saves the winning model artifacts, and uses them in an interactive dashboard for product-level forecasts, safety stock, reorder points, holiday context, and downloadable reports.

## What This App Does

- Trains demand models from `data/sales_data.xlsx`
- Builds weekly product-store-region demand features from transaction data
- Adds lag, rolling, seasonal, discount, EMI/cashback, and Indian holiday features
- Selects the best model from XGBoost, Random Forest, and HistGradientBoosting variants
- Saves model and encoder artifacts under `models/`
- Runs a Streamlit dashboard for historical trend review and inventory planning
- Provides a separate Streamlit model-training workspace for uploading a new `sales_data.xlsx`
- Exports individual product forecasts and bulk store forecasts as CSV files

## Project Structure

```text
iPlanet_stock_prediction_app/
|-- README.md
|-- requirements.txt
|-- test_genai.py
|-- data/
|   |-- sales_data.xlsx
|   `-- inventory_master_data.xlsx
|-- logs/
|   `-- app.log
|-- models/
|   |-- master_model.pkl
|   |-- model_metrics.pkl
|   |-- le_region.pkl
|   |-- le_store.pkl
|   |-- le_prod.pkl
|   |-- le_product_store.pkl
|   `-- le_month_product.pkl
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
| `src/streamlit_app.py` | Streamlit UI for selecting region, store, product, forecast date, horizon, lead time, and growth margin. |
| `src/training_pipeline.py` | Loads sales data, validates required columns, engineers features, trains multiple models, evaluates them, and writes model artifacts. |
| `src/report_engine.py` | Anchors forecasts to recent historical sales and creates individual or bulk CSV reports. |
| `src/data_engine.py` | Utility functions for dropdown data and weekly inference data preparation. |
| `src/external_features.py` | Adds Indian holiday features and optional external feature files if present. |
| `src/genai_layer.py` | Placeholder for GenAI functions. It is currently empty, so AI insight imports fall back gracefully. |
| `test_genai.py` | Smoke test script intended for future GenAI integration. |
| `src/utils/logger.py` | Simple timestamped console logger used by the training pipeline. |

## Data Inputs

Required:

- `data/sales_data.xlsx`

Currently used columns include:

- `RegionName`
- `StoreName`
- `ProductName`
- `BillDate`
- `Quantity`
- `MRP`
- `ProductLevelDiscAmount`
- `True Demand`
- `Bank Funded NoCost EMI Applied With BankName`
- `Bank Funded Cashback Coverted To Percentage With BankName`
- `Additional Apple Or Distributor Funded Scheme In Percentage`

Also present:

- `data/inventory_master_data.xlsx`, with product metadata columns such as `AlternateProductCodes`, `ProductName`, `BusinessSegmentName`, `Class`, `SubClass`, and `ProductCategoryName`.

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

## Train Or Refresh The Model

There are two supported training flows.

### Train From The Streamlit UI

Start the app:

```powershell
streamlit run src\streamlit_app.py
```

In the sidebar, switch `Workspace` to `Model Training`.

The training workspace lets you:

- Upload a workbook named `sales_data.xlsx`
- Validate the required training columns
- Click `Train Model`
- Train the model tournament on the uploaded file
- Stage generated `.pkl` files in `models/_training_latest/`
- Promote the staged artifacts to `models/` only after training succeeds
- Replace the active `data/sales_data.xlsx` with the uploaded training file
- Clear Streamlit caches so the prediction dashboard can load the new model

After training completes, switch `Workspace` back to `Prediction Dashboard` and run predictions with the newly active model artifacts.

Uploaded training files are retained under `data/training_uploads/` with timestamps.

### Train From The Command Line

Run the training pipeline from the project root:

```powershell
python src\training_pipeline.py
```

This pipeline:

1. Loads `data/sales_data.xlsx`
2. Aggregates transactions into weekly product-store-region demand
3. Builds enhanced time-series, seasonal, holiday, pricing, and promotion features
4. Trains an AutoML-style model tournament
5. Selects the lowest-MAE model
6. Saves the winning model and label encoders to `models/`

Current saved metrics in `models/model_metrics.pkl`:

- Algorithm: `Hist Gradient Boosting Enhanced`
- Forecast Accuracy: `82.17%`
- MAE: `0.5005`
- RMSE: `2.18`
- sMAPE: `83.45%`
- Features Used: `48`

## Run The Dashboard

Start Streamlit from the project root:

```powershell
streamlit run src\streamlit_app.py
```

The sidebar lets you choose:

- Region
- Store
- Product
- Historical forecast start date
- Planning horizon: 7, 30, or 60 days
- Supplier lead time: 7, 30, or 60 days
- Growth margin percentage

Click `Run Intelligence Report` to generate the dashboard.

## Dashboard Views

The dashboard includes four tabs:

- `Historical Trend`: monthly historical sales trend for the selected product and store.
- `Demand Forecast & Strategy`: weekly forecast chart, demand total, safety stock, reorder point, daily velocity, procurement recommendation, technical scorecard, and holiday calendar.
- `Reports`: downloadable individual product forecast CSV and bulk store forecast CSV.
- `AI Insights`: reserved for GenAI-powered explanations and recommendations when `src/genai_layer.py` is implemented.

## Forecasting Workflow

```text
data/sales_data.xlsx
        |
        v
src/training_pipeline.py
        |
        v
models/master_model.pkl + label encoders + metrics
        |
        v
src/streamlit_app.py
        |
        +--> src/report_engine.py
        +--> src/external_features.py
        |
        v
Streamlit dashboard + CSV exports
```

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

`src/streamlit_app.py`, `src/report_engine.py`, and `test_genai.py` are wired for a future GenAI layer, but `src/genai_layer.py` is currently empty. Because of that:

- The dashboard will show limited AI insight availability.
- `test_genai.py` is expected to fail the GenAI import checks until the functions are implemented.
- Setting `OPENAI_API_KEY` alone is not enough until `src/genai_layer.py` contains the required functions.

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

- Run commands from the project root so relative paths like `data/sales_data.xlsx` and `models/master_model.pkl` resolve correctly.
- The checked-in `venv/` directory is not needed if you create your own virtual environment.
- `logs/app.log` exists, but the active logger currently prints timestamped messages to the console.
- Some app labels in the Python files contain mojibake characters from encoding issues; this README uses plain ASCII for readability.
