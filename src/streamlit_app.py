import streamlit as st
import pandas as pd
import joblib
import os
import datetime
import shutil
import numpy as np
import io
import altair as alt

try:
    from training_pipeline import MODEL_ARTIFACTS, validate_sales_data_columns, run_master_tournament
    training_pipeline_available = True
except ImportError as e:
    training_pipeline_available = False
    training_pipeline_error = e

# Import the dedicated report module
try:
    import report_engine as re
except ImportError:
    st.error("Missing 'report_engine.py' in the src folder.")

# Import holiday feature helper
try:
    from external_features import add_external_features, get_state_holiday_list
    holiday_features_available = True
except ImportError:
    holiday_features_available = False
    st.warning("Holiday feature module not available. Predictions will use default holiday behavior.")

# Import GenAI layer
try:
    from genai_layer import (
        initialize_genai, explain_forecast_accuracy, analyze_forecast_trends,
        generate_inventory_recommendations, create_executive_summary,
        detect_anomalies, get_forecast_insights, format_business_alert
    )
    genai_available = True
except ImportError:
    genai_available = False
    genai_import_error = "GenAI layer not available. AI insights will be limited."

# --- PAGE CONFIG ---
st.set_page_config(page_title="iPlanet Demand Planner | Bangalore", layout="wide", page_icon="🍎")

# --- APP SHELL STYLING ---
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }
    .app-brand {
        border: 1px solid rgba(49, 51, 63, 0.16);
        border-radius: 10px;
        padding: 0.9rem 0.95rem;
        margin-bottom: 0.9rem;
        background: linear-gradient(135deg, #f8fafc 0%, #eef5f2 100%);
    }
    .app-brand-title {
        font-size: 1.02rem;
        font-weight: 750;
        color: #1f2937;
        line-height: 1.25;
    }
    .app-brand-subtitle {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 0.25rem;
    }
    .workspace-help {
        border-left: 3px solid #2f855a;
        padding: 0.45rem 0.65rem;
        margin: 0.55rem 0 1rem 0;
        background: #f7fbf8;
        color: #334155;
        font-size: 0.82rem;
    }
    .section-kicker {
        color: #64748b;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        margin-bottom: 0.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- TRAINING WORKSPACE ---
DATA_PATH = 'data/sales_data.xlsx'
MODEL_DIR = 'models'
TRAINING_UPLOAD_DIR = os.path.join('data', 'training_uploads')
STAGED_MODEL_DIR = os.path.join(MODEL_DIR, '_training_latest')


def render_workspace_selector():
    st.sidebar.markdown(
        """
        <div class="app-brand">
            <div class="app-brand-title">iPlanet Demand Planner</div>
            <div class="app-brand-subtitle">Forecasting, inventory planning, and model refresh</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.sidebar.markdown('<div class="section-kicker">Workspace</div>', unsafe_allow_html=True)
    options = ["Prediction Dashboard", "Model Training"]
    labels = {
        "Prediction Dashboard": "Prediction",
        "Model Training": "Training"
    }
    if hasattr(st.sidebar, "segmented_control"):
        selected_workspace = st.sidebar.segmented_control(
            "Workspace",
            options,
            default="Prediction Dashboard",
            format_func=lambda option: labels[option],
            label_visibility="collapsed"
        )
    else:
        selected_workspace = st.sidebar.radio(
            "Workspace",
            options,
            format_func=lambda option: labels[option],
            label_visibility="collapsed"
        )

    help_text = {
        "Prediction Dashboard": "Use the active trained model to forecast product demand and export reports.",
        "Model Training": "Upload fresh sales data, train a new model, and promote it after a successful run."
    }
    st.sidebar.markdown(
        f'<div class="workspace-help">{help_text[selected_workspace]}</div>',
        unsafe_allow_html=True
    )
    return selected_workspace


def render_training_workspace():
    st.title("Model Training")
    st.caption("Upload fresh sales data, run the model tournament, and promote the generated artifacts only after training succeeds.")

    if not training_pipeline_available:
        st.error(f"Training pipeline could not be imported: {training_pipeline_error}")
        return

    step_a, step_b, step_c = st.columns(3)
    with step_a:
        with st.container(border=True):
            st.markdown("**1. Upload**")
            st.write("Choose the latest `sales_data.xlsx` workbook.")
    with step_b:
        with st.container(border=True):
            st.markdown("**2. Validate**")
            st.write("Required sales columns are checked before training.")
    with step_c:
        with st.container(border=True):
            st.markdown("**3. Promote**")
            st.write("New `.pkl` files become active only after success.")

    current_metrics_path = os.path.join(MODEL_DIR, 'model_metrics.pkl')
    model_col, upload_col = st.columns([1, 1.25], vertical_alignment="top")
    with model_col:
        with st.container(border=True):
            st.subheader("Current Active Model")
            if os.path.exists(current_metrics_path):
                try:
                    current_metrics = pd.DataFrame(joblib.load(current_metrics_path).items(), columns=["Metric", "Value"])
                    current_metrics['Value'] = current_metrics['Value'].astype(str)
                    st.dataframe(current_metrics, use_container_width=True, hide_index=True)
                except Exception as e:
                    st.warning(f"Could not read current model metrics: {e}")
            else:
                st.info("No active model metrics found yet.")

    with upload_col:
        upload_panel = st.container(border=True)
        upload_panel.subheader("Train From Uploaded Sales Data")
        uploaded_file = upload_panel.file_uploader(
            "Upload input file named sales_data.xlsx",
            type=["xlsx"],
            help="The workbook must include the required sales columns used by the training pipeline."
        )

    if uploaded_file is None:
        upload_panel.info("Choose a sales workbook to enable training.")
        return

    if uploaded_file.name.lower() != "sales_data.xlsx":
        upload_panel.warning("Recommended filename is `sales_data.xlsx`. The uploaded file will still be validated before training.")

    try:
        preview_columns = pd.read_excel(uploaded_file, nrows=0, engine='openpyxl').columns.tolist()
        uploaded_file.seek(0)
        required_columns = [
            'RegionName', 'StoreName', 'ProductName', 'BillDate',
            'Quantity', 'MRP', 'ProductLevelDiscAmount'
        ]
        missing_columns = [column for column in required_columns if column not in preview_columns]
        if missing_columns:
            upload_panel.error(f"Cannot train yet. Missing required columns: {', '.join(missing_columns)}")
            return
        upload_panel.success(f"Validated workbook columns. Found {len(preview_columns)} columns.")
    except Exception as e:
        upload_panel.error(f"Could not read uploaded workbook: {e}")
        return

    if upload_panel.button("Train Model", type="primary"):
        os.makedirs(TRAINING_UPLOAD_DIR, exist_ok=True)
        os.makedirs(STAGED_MODEL_DIR, exist_ok=True)
        for artifact in MODEL_ARTIFACTS:
            staged_artifact_path = os.path.join(STAGED_MODEL_DIR, artifact)
            if os.path.exists(staged_artifact_path):
                os.remove(staged_artifact_path)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        upload_path = os.path.join(TRAINING_UPLOAD_DIR, f"sales_data_{timestamp}.xlsx")

        with open(upload_path, "wb") as output_file:
            output_file.write(uploaded_file.getbuffer())

        training_logs = []

        def ui_log(message):
            training_logs.append(message)

        try:
            with upload_panel.status("Training model tournament. This can take a few minutes...", expanded=True) as training_status:
                missing_columns, _ = validate_sales_data_columns(upload_path)
                if missing_columns:
                    raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

                metrics = run_master_tournament(
                    data_path=upload_path,
                    model_dir=STAGED_MODEL_DIR,
                    log_func=ui_log
                )
                if metrics is None:
                    raise RuntimeError("Training did not return model metrics.")

                missing_artifacts = [
                    artifact for artifact in MODEL_ARTIFACTS
                    if not os.path.exists(os.path.join(STAGED_MODEL_DIR, artifact))
                ]
                if missing_artifacts:
                    raise RuntimeError(f"Training completed but artifacts are missing: {', '.join(missing_artifacts)}")

                training_status.update(label="Training finished. Promoting model artifacts...", state="running")
                os.makedirs(MODEL_DIR, exist_ok=True)
                for artifact in MODEL_ARTIFACTS:
                    shutil.copyfile(
                        os.path.join(STAGED_MODEL_DIR, artifact),
                        os.path.join(MODEL_DIR, artifact)
                    )
                shutil.copyfile(upload_path, DATA_PATH)

                st.cache_data.clear()
                st.cache_resource.clear()
                training_status.update(label="Training completed. New model artifacts are active.", state="complete", expanded=False)

            upload_panel.success("Training completed. New model artifacts are now active for prediction.")
            upload_panel.subheader("New Model Metrics")
            upload_panel.table(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]))
            with upload_panel.expander("Training log"):
                st.text("\n".join(training_logs))
        except Exception as e:
            upload_panel.error(f"Training failed: {e}")
            if training_logs:
                with upload_panel.expander("Training log"):
                    st.text("\n".join(training_logs))


workspace = render_workspace_selector()
if not genai_available:
    st.sidebar.caption("AI insights are disabled until the GenAI layer is implemented.")

if workspace == "Model Training":
    render_training_workspace()
    st.stop()

# --- ASSET LOADING ---
@st.cache_resource
def load_assets():
    """Loads the AI brain and the technical scorecard generated during training."""
    try:
        model = joblib.load('models/master_model.pkl')
        le_reg = joblib.load('models/le_region.pkl')
        le_store = joblib.load('models/le_store.pkl')
        le_prod = joblib.load('models/le_prod.pkl')
        
        # Try to load enhanced encoders, use defaults if not available
        try:
            le_combo = joblib.load('models/le_product_store.pkl')
        except:
            le_combo = None
            
        try:
            le_month_prod = joblib.load('models/le_month_product.pkl')
        except:
            le_month_prod = None
            
        metrics = joblib.load('models/model_metrics.pkl')
        return model, le_reg, le_store, le_prod, le_combo, le_month_prod, metrics
    except:
        return None, None, None, None, None, None, None

# --- SESSION STATE INITIALIZATION ---
# This prevents the app from refreshing to the landing page after every click.
if 'report_generated' not in st.session_state:
    st.session_state.report_generated = False

# --- GENAI INITIALIZATION ---
if genai_available:
    try:
        # Try to initialize GenAI (will work if OPENAI_API_KEY is set)
        initialize_genai(provider="openai")
        genai_initialized = True
    except Exception as e:
        st.sidebar.warning(f"GenAI not initialized: {e}")
        genai_initialized = False
else:
    genai_initialized = False

# --- SIDEBAR: PREDICTION CONTROLS ---
st.sidebar.markdown('<div class="section-kicker">Prediction Controls</div>', unsafe_allow_html=True)
DATA_PATH = 'data/sales_data.xlsx'

if os.path.exists(DATA_PATH):
    @st.cache_data
    def get_meta(path):
        """Loads necessary columns for historical trending and AutoML grounding."""
        base_cols = ['RegionName', 'StoreName', 'ProductName', 'BillDate', 'Quantity']
        optional_cols = [
            'True Demand',
            'Bank Funded NoCost EMI Applied With BankName',
            'Bank Funded Cashback Coverted To Percentage With BankName',
            'Additional Apple Or Distributor Funded Scheme In Percentage'
        ]
        available_columns = pd.read_excel(path, nrows=0, engine='openpyxl').columns.tolist()
        usecols = base_cols + [c for c in optional_cols if c in available_columns]
        df = pd.read_excel(path, usecols=usecols, engine='openpyxl')
        return df, [c for c in optional_cols if c in df.columns]
    
    df_meta, available_extra_features = get_meta(DATA_PATH)
    sel_reg = st.sidebar.selectbox("Region", sorted(df_meta['RegionName'].unique()))
    sel_store = st.sidebar.selectbox("Store", sorted(df_meta[df_meta['RegionName'] == sel_reg]['StoreName'].unique()))
    sel_prod = st.sidebar.selectbox("Product", sorted(df_meta[df_meta['StoreName'] == sel_store]['ProductName'].unique()))
    
    st.sidebar.divider()
    # Add historical forecast date selector
    max_date = pd.to_datetime(df_meta['BillDate']).max().date()
    min_date = pd.to_datetime(df_meta['BillDate']).min().date()
    forecast_start_date = st.sidebar.date_input(
        "Forecast Start Date (Historical)",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
        help="Select a historical date to forecast from. Data up to this date will be used for training."
    )
    st.sidebar.divider()
    horizon = st.sidebar.select_slider("Planning Horizon (Days)", options=[7, 30, 60], value=30)
    lead_time = st.sidebar.select_slider("Supplier Lead Time (Days)", options=[7, 30, 60], value=7)
    growth_margin = st.sidebar.slider("Growth Margin (%)", 0, 50, 15, 5)
else:
    st.error("Critical: 'data/sales_data.xlsx' not found.")
    st.stop()

# --- TRIGGER REPORT GENERATION ---
if st.sidebar.button("🚀 Run Intelligence Report"):
    st.session_state.report_generated = True

# --- MAIN DASHBOARD AREA ---
if st.session_state.report_generated:
    model, le_reg, le_store, le_prod, le_combo, le_month_prod, metrics = load_assets()
    
    if model and metrics:
        try:
            # 1. FETCH HISTORICAL STARTING POINTS (using cutoff date)
            last_qty, rolling_qty, last_true_demand, last_emi_ratio, last_cashback_pct, last_add_pct = re.get_last_known_metrics(
                df_meta, sel_store, sel_prod, forecast_start_date
            )
            num_weeks = (horizon // 7) + (1 if horizon % 7 > 0 else 0)
            
            # 2. CONSTRUCT ENHANCED FEATURE DATAFRAME (matching training pipeline)
            # Create base features
            base_features = {
                'RegionID': [le_reg.transform([sel_reg])[0]]*num_weeks,
                'StoreID': [le_store.transform([sel_store])[0]]*num_weeks,
                'ProductID': [le_prod.transform([sel_prod])[0]]*num_weeks,
                'RegionName': [sel_reg] * num_weeks,
                'BillDate': [(forecast_start_date + datetime.timedelta(weeks=i)) for i in range(num_weeks)],
                'Week': [(forecast_start_date + datetime.timedelta(weeks=i)).isocalendar()[1] for i in range(num_weeks)],
                'Month': [(forecast_start_date + datetime.timedelta(weeks=i)).month for i in range(num_weeks)],
                'Year': [(forecast_start_date + datetime.timedelta(weeks=i)).year for i in range(num_weeks)],
                'DiscountPct': [0.05]*num_weeks,
                'Lag_1_Week': [last_qty]*num_weeks,
                'Lag_2_Week': [last_qty]*num_weeks,
                'Lag_3_Week': [last_qty]*num_weeks,
                'Lag_4_Week': [last_qty]*num_weeks,
                'Lag_8_Week': [last_qty]*num_weeks,
                'Lag_12_Week': [last_qty]*num_weeks,
                'Rolling_Mean_2': [rolling_qty]*num_weeks,
                'Rolling_Mean_4': [rolling_qty]*num_weeks,
                'Rolling_Mean_8': [rolling_qty]*num_weeks,
                'Rolling_Mean_12': [rolling_qty]*num_weeks,
                'Rolling_Std_2': [rolling_qty * 0.15]*num_weeks,
                'Rolling_Std_4': [rolling_qty * 0.15]*num_weeks,
                'Rolling_Std_8': [rolling_qty * 0.15]*num_weeks,
                'Rolling_Std_12': [rolling_qty * 0.15]*num_weeks,
                'True_Demand': [last_true_demand]*num_weeks,
                'TrueDemand_Ratio': [last_true_demand / (last_qty + 1e-5)]*num_weeks,
                'Bank_Emi_Flag': [last_emi_ratio]*num_weeks,
                'Bank_Cashback_Pct': [last_cashback_pct]*num_weeks,
                'Additional_Funded_Scheme_Pct': [last_add_pct]*num_weeks
            }
            
            # Create future_df with base features
            future_df = pd.DataFrame(base_features)
            future_df['BillDate'] = pd.to_datetime(future_df['BillDate'])
            
            # Add enhanced features to match training pipeline
            
            # Temporal features
            future_df['Quarter'] = future_df['Month'].apply(lambda x: (x-1)//3 + 1)
            future_df['Is_Month_Start'] = [(forecast_start_date + datetime.timedelta(weeks=i)).day <= 7 for i in range(num_weeks)]
            future_df['Is_Month_End'] = [(forecast_start_date + datetime.timedelta(weeks=i)).day >= 25 for i in range(num_weeks)]
            future_df['Week_Sin'] = np.sin(2 * np.pi * future_df['Week'] / 52.0)
            future_df['Week_Cos'] = np.cos(2 * np.pi * future_df['Week'] / 52.0)
            future_df['Month_Sin'] = np.sin(2 * np.pi * future_df['Month'] / 12.0)
            future_df['Month_Cos'] = np.cos(2 * np.pi * future_df['Month'] / 12.0)

            # Add holiday features from external feature module
            if holiday_features_available:
                try:
                    future_df = add_external_features(future_df, region_column='RegionName')
                except Exception as e:
                    st.warning(f"Holiday feature generation failed: {e}")
            else:
                st.warning("Holiday features are unavailable because the module could not be imported.")
            
            # Lag features (use available historical data or defaults)
            for lag in [2, 3, 4, 8, 12]:
                future_df[f'Lag_{lag}_Week'] = [last_qty] * num_weeks  # Default to last known value
            
            # Rolling statistics (use available data or defaults)
            for window in [2, 4, 8, 12]:
                future_df[f'Rolling_Mean_{window}'] = [rolling_qty] * num_weeks
                future_df[f'Rolling_Std_{window}'] = [rolling_qty * 0.15] * num_weeks  # Estimate std as 15% of mean
            
            # Trend indicators
            future_df['EWMA_4'] = [rolling_qty] * num_weeks
            future_df['MoM_Change'] = [0.0] * num_weeks  # Default to no change
            future_df['Trend_4Week'] = [0.0] * num_weeks  # Default to no trend
            future_df['Quantity_ZScore'] = [0.0] * num_weeks  # Default to mean
            
            # Price & promotion features
            future_df['Discount_Intensity'] = pd.cut(future_df['DiscountPct'], bins=[0, 0.05, 0.15, 0.3, 1.0], labels=[0, 1, 2, 3]).fillna(0).astype(int)
            future_df['Effective_Price'] = [100.0] * num_weeks  # Default MRP
            future_df['Bank_Cashback_Intensity'] = pd.cut(future_df['Bank_Cashback_Pct'], bins=[-1, 0, 2, 5, 10, 100], labels=[0, 1, 2, 3, 4]).fillna(0).astype(int)
            future_df['Additional_Funded_Scheme_Intensity'] = pd.cut(future_df['Additional_Funded_Scheme_Pct'], bins=[-1, 0, 2, 5, 10, 100], labels=[0, 1, 2, 3, 4]).fillna(0).astype(int)
            
            # Interaction features (using proper encoders if available)
            future_df['Product_Store_Combo'] = future_df['ProductID'].astype(str) + '_' + future_df['StoreID'].astype(str)
            if le_combo is not None:
                try:
                    future_df['Product_Store_ID'] = le_combo.transform(future_df['Product_Store_Combo'])
                except ValueError:
                    # Handle unseen labels by using a default value
                    future_df['Product_Store_ID'] = [0] * num_weeks  # Default for unseen combinations
                    st.warning("Some product-store combinations not seen during training, using defaults")
            else:
                future_df['Product_Store_ID'] = [0] * num_weeks  # Fallback for older models
            
            future_df['Month_Product'] = future_df['Month'].astype(str) + '_' + future_df['ProductID'].astype(str)
            if le_month_prod is not None:
                try:
                    future_df['Month_Product_ID'] = le_month_prod.transform(future_df['Month_Product'])
                except ValueError:
                    # Handle unseen labels by using a default value
                    future_df['Month_Product_ID'] = [0] * num_weeks  # Default for unseen combinations
                    st.warning("Some month-product combinations not seen during training, using defaults")
            else:
                future_df['Month_Product_ID'] = [0] * num_weeks  # Fallback for older models
            
            # NOW detect which features the model was trained with (after all features are created)
            try:
                # Try to get feature names from the model if available (for sklearn models)
                if hasattr(model, 'feature_names_in_'):
                    model_features = list(model.feature_names_in_)
                    st.info(f"Model expects {len(model_features)} features")
                else:
                    # Fallback: use all available features that match expected pattern
                    expected_features = [
                        'RegionID', 'StoreID', 'ProductID', 'Product_Store_ID', 'Month_Product_ID',
                        'Week', 'Month', 'Year', 'Quarter', 'Is_Month_Start', 'Is_Month_End',
                        'Week_Sin', 'Week_Cos', 'Month_Sin', 'Month_Cos',
                        'Lag_1_Week', 'Lag_2_Week', 'Lag_3_Week', 'Lag_4_Week', 'Lag_8_Week', 'Lag_12_Week',
                        'Rolling_Mean_2', 'Rolling_Mean_4', 'Rolling_Mean_8', 'Rolling_Mean_12',
                        'Rolling_Std_2', 'Rolling_Std_4', 'Rolling_Std_8', 'Rolling_Std_12',
                        'EWMA_4', 'MoM_Change', 'Trend_4Week', 'Quantity_ZScore',
                        'True_Demand', 'TrueDemand_Ratio', 'Bank_Emi_Flag', 'Bank_Cashback_Pct',
                        'Bank_Cashback_Intensity', 'Additional_Funded_Scheme_Pct', 'Additional_Funded_Scheme_Intensity',
                        'DiscountPct', 'Discount_Intensity', 'Effective_Price'
                    ]
                    model_features = [f for f in expected_features if f in future_df.columns]
                    st.info(f"Using {len(model_features)} available features")
            except Exception as e:
                # Ultimate fallback to basic features that should always exist
                model_features = ['RegionID', 'StoreID', 'ProductID', 'Week', 'Month', 'Year', 'DiscountPct', 'Lag_1_Week', 'Rolling_Mean_4']
                st.warning(f"Feature detection failed ({e}), using basic features")
            
            # Reorder columns to match model expectations
            missing_features = [f for f in model_features if f not in future_df.columns]
            if missing_features:
                for feature in missing_features:
                    future_df[feature] = 0
                st.warning(f"Filled missing model features with defaults: {missing_features}")

            future_df = future_df[model_features]
            
            # Debug: Check for missing features
            missing_features = [f for f in model_features if f not in future_df.columns]
            if missing_features:
                st.error(f"Missing features after reordering: {missing_features}")
                st.stop()

            # 3. EXECUTE PREDICTION
            preds = [max(0, p) for p in model.predict(future_df)]

            # Holiday list for selected state and forecast window
            holiday_df = pd.DataFrame()
            if holiday_features_available:
                try:
                    years = sorted({forecast_start_date.year, (forecast_start_date + datetime.timedelta(days=horizon-1)).year})
                    holiday_df = get_state_holiday_list(sel_reg, years=years)
                    end_date = forecast_start_date + datetime.timedelta(days=horizon-1)
                    holiday_df['Date'] = pd.to_datetime(holiday_df['Date'])
                    holiday_df = holiday_df[(holiday_df['Date'] >= pd.Timestamp(forecast_start_date)) & (holiday_df['Date'] <= pd.Timestamp(end_date))]
                    holiday_df['Date'] = holiday_df['Date'].dt.date
                except Exception as e:
                    st.warning(f"Unable to build holiday list: {e}")
            total_qty = int(round(sum(preds)))
            labels = [(forecast_start_date + datetime.timedelta(weeks=i)).strftime('%b %d') for i in range(num_weeks)]

            # --- TABBED LAYOUT ---
            st.title(f"📊 {sel_prod} Analysis: {sel_store}")
            st.caption(f"Forecast starts from {forecast_start_date.strftime('%B %d, %Y')}.")
            
            tab1, tab2, tab3, tab4 = st.tabs([
                "📈 Historical Trend", 
                "🔮 Demand Forecast & Strategy", 
                "📋 Reports", 
                "🤖 AI Insights"
            ])

            # TAB 1: HISTORICAL TREND
            with tab1:
                trend_panel = st.container(border=True)
                trend_panel.subheader("3-Year Historical Sales Trend")
                full_hist = df_meta[(df_meta['StoreName'] == sel_store) & (df_meta['ProductName'] == sel_prod)].copy()
                if not full_hist.empty:
                    full_hist['BillDate'] = pd.to_datetime(full_hist['BillDate'])
                    # resample uses 'ME' for modern Pandas compatibility
                    trend_data = full_hist.set_index('BillDate').sort_index().resample('ME')['Quantity'].sum()
                    trend_panel.line_chart(trend_data, color="#3498db")
                    trend_panel.write(f"*Actual sales recorded from {trend_data.index.min().date()} to {trend_data.index.max().date()}.*")
                else:
                    trend_panel.warning("No historical data available for this selection.")

            # TAB 2: FORECAST & STRATEGY
            with tab2:
                avg_daily = total_qty / horizon
                sigma = np.std(preds) if len(preds) > 1 else (total_qty * 0.15)
                safety_stock = round(1.65 * sigma * np.sqrt(lead_time))
                reorder_point = round((avg_daily * lead_time) + safety_stock)
                final_proc = int(round(total_qty * (1 + (growth_margin/100))))

                with st.container(border=True):
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric(f"{horizon}-Day Demand", f"{total_qty} Units")
                    k2.metric("Safety Stock", f"{safety_stock} Units")
                    k3.metric("Reorder Point", f"{reorder_point}")
                    k4.metric("Daily Velocity", f"{round(avg_daily, 2)} / Day")
                
                l_col, r_col = st.columns([1.55, 1], vertical_alignment="top")
                with l_col:
                    forecast_panel = st.container(border=True)
                    forecast_panel.subheader("AI Demand Projection")
                    forecast_chart = pd.DataFrame({'Week': labels, 'Units': preds})
                    forecast_chart['Week'] = forecast_chart['Week'].astype(str)
                    chart = alt.Chart(forecast_chart).encode(
                        x=alt.X('Week:N', title='Week', sort=None),
                        y=alt.Y('Units:Q', title='Forecast Units'),
                        tooltip=[
                            alt.Tooltip('Week:N', title='Week'),
                            alt.Tooltip('Units:Q', title='Forecast Units', format=',.0f')
                        ]
                    )
                    forecast_viz = (
                        chart.mark_area(color="#2ecc71", opacity=0.14)
                        + chart.mark_line(color="#2ecc71", size=3, interpolate='monotone')
                        + chart.mark_point(color="#117a65", size=120, filled=True)
                    ).configure_axis(labelAngle=-45)
                    forecast_panel.altair_chart(forecast_viz, use_container_width=True)
                    forecast_panel.caption(f"Trend projection from {labels[0]} to {labels[-1]}.")
                with r_col:
                    strategy_panel = st.container(border=True)
                    strategy_panel.subheader("Inventory Strategy")
                    strategy_panel.info(f"Planning horizon velocity: **{round(np.mean(preds), 1)} units/week**.")
                    strategy_panel.info(f"Lead time buffer: **{safety_stock} units**.")
                    strategy_panel.info(f"Restock trigger: **{reorder_point} units**.")
                    strategy_panel.success(f"Procure **{final_proc} units** with {growth_margin}% growth margin.")
                    # Add AI insights if available
                    if genai_initialized:
                        strategy_panel.markdown("---")
                        strategy_panel.subheader("AI-Powered Insights")
                        with st.spinner("Generating AI insights..."):
                            quick_insights = get_forecast_insights(preds, sel_prod)
                            strategy_panel.info(quick_insights)

                score_col, holiday_col = st.columns(2, vertical_alignment="top")
                with score_col:
                    score_panel = st.container(border=True)
                    score_panel.subheader(f"Technical Scorecard: {metrics['Algorithm']}")
                    m_df = pd.DataFrame(metrics.items(), columns=["Parameter", "Value"])
                    m_df['Value'] = m_df['Value'].astype(str) # Prevents Arrow TypeError
                    score_panel.dataframe(m_df, use_container_width=True, hide_index=True)

                with holiday_col:
                    holiday_panel = st.container(border=True)
                    holiday_panel.subheader("Holiday Calendar")
                    if not holiday_df.empty:
                        holiday_panel.write(f"Showing holidays for **{sel_reg}** from **{forecast_start_date}** to **{end_date}**.")
                        holiday_panel.dataframe(holiday_df.reset_index(drop=True), use_container_width=True, hide_index=True)

                        total_holidays = len(holiday_df)
                        upcoming_holiday = holiday_df.iloc[0] if total_holidays > 0 else None
                        if upcoming_holiday is not None:
                            holiday_name = upcoming_holiday['Holiday'] or 'Unnamed holiday'
                            holiday_date = upcoming_holiday['Date']
                            holiday_panel.markdown("**Holiday Impact Summary**")
                            holiday_panel.write(
                                f"- **{total_holidays} holiday(s)** occur during the forecast window.\n"
                                f"- The earliest upcoming holiday is **{holiday_name}** on **{holiday_date}**.\n"
                                f"- Holiday weeks often shift buying behavior, so inventory and promotions should be checked for these dates."
                            )
                        else:
                            holiday_panel.markdown("**Holiday Impact Summary**")
                            holiday_panel.write("- Holiday dates were identified but the list could not be summarized.")
                    elif holiday_features_available:
                        holiday_panel.info("No holidays found in the forecast window for this region.")
                    else:
                        holiday_panel.info("Holiday support is unavailable for this session.")

            # TAB 3: REPORTS
            with tab3:
                st.subheader("Export Center")
                col_a, col_b = st.columns(2)
                with col_a:
                    single_report_panel = st.container(border=True)
                    single_report_panel.subheader("Individual Product Report")
                    csv_bytes = re.generate_single_forecast_csv(sel_reg, sel_store, sel_prod, labels, preds, forecast_start_date)
                    single_report_panel.download_button("Download Detailed CSV", csv_bytes, f"Forecast_{sel_prod}.csv", "text/csv")
                
                with col_b:
                    master_report_panel = st.container(border=True)
                    master_report_panel.subheader("Master Store Report")
                    if master_report_panel.button("Generate Bulk Analysis"):
                        with st.spinner("Processing store-wide intelligence..."):
                            bulk_df = re.generate_bulk_report(df_meta, model, (le_reg, le_store, le_prod, le_combo, le_month_prod), sel_reg, sel_store, forecast_start_date)
                            master_report_panel.download_button("Download Master Bulk CSV", bulk_df.to_csv(index=False).encode('utf-8'), f"Bulk_{sel_store}.csv", "text/csv")

                            # Add AI-powered executive summary if GenAI is available
                            if genai_initialized and not bulk_df.empty:
                                master_report_panel.subheader("AI Executive Summary")
                                with st.spinner("Generating executive summary..."):
                                    executive_summary = re.generate_executive_summary_report(
                                        bulk_df, metrics,
                                        business_context=f"Store-wide forecast analysis for {sel_store} in {sel_reg}"
                                    )
                                    master_report_panel.write(executive_summary)

            # TAB 4: AI INSIGHTS
            with tab4:
                st.subheader("🤖 GenAI Strategic Briefing")

                if not genai_initialized:
                    st.warning("⚠️ GenAI not available. Please set OPENAI_API_KEY environment variable to enable AI insights.")
                    st.info(f"The {metrics['Algorithm']} model currently suggests a stability in demand for {sel_prod} based on historical patterns in {sel_reg}.")
                else:
                    # Get historical data for analysis
                    hist_data = df_meta[(df_meta['StoreName'] == sel_store) & (df_meta['ProductName'] == sel_prod)].copy()
                    hist_avg = hist_data['Quantity'].mean() if not hist_data.empty else 0

                    # AI-Powered Analysis Tabs
                    insight_tab1, insight_tab2, insight_tab3, insight_tab4 = st.tabs([
                        "📊 Model Performance",
                        "🔍 Trend Analysis",
                        "📦 Inventory Strategy",
                        "🚨 Alerts & Anomalies"
                    ])

                    with insight_tab1:
                        st.subheader("Model Accuracy Explanation")
                        with st.spinner("Analyzing model performance..."):
                            accuracy_explanation = explain_forecast_accuracy(
                                metrics,
                                business_context=f"Retail demand forecasting for {sel_prod} in {sel_store}, {sel_reg}"
                            )
                            st.write(accuracy_explanation)

                    with insight_tab2:
                        st.subheader("Forecast Trend Analysis")
                        with st.spinner("Analyzing forecast trends..."):
                            # Get external factors (placeholder - can be enhanced later)
                            external_factors = {
                                "season": "Based on historical patterns",
                                "promotion": f"Assumed {future_df['DiscountPct'].iloc[0]*100}% discount",
                                "competitor_activity": "Not available",
                                "economic_indicators": "Not available"
                            }

                            trend_analysis = analyze_forecast_trends(
                                preds, sel_prod, hist_avg, external_factors
                            )
                            st.write(trend_analysis)

                    with insight_tab3:
                        st.subheader("Smart Inventory Recommendations")
                        current_stock = st.number_input(
                            "Current Stock Level",
                            min_value=0,
                            value=int(reorder_point * 0.8),  # Estimate current stock
                            help="Enter your current inventory level for personalized recommendations"
                        )

                        with st.spinner("Generating inventory strategy..."):
                            inventory_recommendations = generate_inventory_recommendations(
                                preds, current_stock, sel_prod, lead_time
                            )

                            if "error" not in inventory_recommendations:
                                col1, col2 = st.columns(2)

                                with col1:
                                    st.metric("Recommended Reorder Quantity", inventory_recommendations['recommended_reorder_quantity'])
                                    st.metric("Safety Stock Level", inventory_recommendations['safety_stock_level'])
                                    st.metric("Risk Level", inventory_recommendations['risk_level'])

                                with col2:
                                    st.metric("Reorder Point", inventory_recommendations['reorder_point'])
                                    if inventory_recommendations['projected_stockout_week']:
                                        st.metric("Projected Stockout", f"Week {inventory_recommendations['projected_stockout_week']}")

                                st.subheader("AI Strategy Insights")
                                st.write(inventory_recommendations['ai_insights'])
                            else:
                                st.error(inventory_recommendations['error'])

                    with insight_tab4:
                        st.subheader("Business Alerts & Anomaly Detection")

                        # Business alerts
                        alert = format_business_alert(preds, current_stock, sel_prod)
                        if alert:
                            st.error(alert)
                        else:
                            st.success("✅ No critical inventory alerts at current stock levels")

                        # Anomaly detection
                        if not hist_data.empty:
                            with st.spinner("Detecting forecast anomalies..."):
                                anomaly_analysis = detect_anomalies(
                                    preds, hist_data, sel_prod, threshold=2.0
                                )
                                st.subheader("Anomaly Analysis")
                                st.write(anomaly_analysis)
                        else:
                            st.info("Insufficient historical data for anomaly detection")

                    # Quick Insights Summary
                    st.divider()
                    st.subheader("🎯 Quick AI Insights")
                    with st.spinner("Generating summary insights..."):
                        quick_insights = get_forecast_insights(preds, sel_prod)
                        st.info(quick_insights)

        except Exception as e:
            st.error(f"⚠️ Calculation Error: {e}")
            st.session_state.report_generated = False
    else:
        st.error("Model assets missing. Please run training_pipeline.py.")
        st.session_state.report_generated = False
else:
    # --- LANDING STATE ---
    st.info("👈 Select your parameters and click 'Run Intelligence Report' to begin.")
st.image("https://images.unsplash.com/photo-1510878939963-10214bb20ec8?auto=format&fit=crop&q=80&w=1200", 
             caption="iPlanet Demand Planner | Bangalore", width=1200)
