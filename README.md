# SupplyWise AI — AI Supply Chain Optimization Project

A beginner-friendly college project that demonstrates:
- Inventory monitoring
- Demand forecasting baseline
- Stockout risk estimates
- Reorder-point and suggested-order calculations
- Supplier ranking by cost, speed, reliability, packaging score, and distance
- Downloadable reorder plan and simulated purchase-order preview

**Important:** this is a prototype with synthetic sample data. It does not connect to real suppliers or place real orders.

## Recommended tools
- **Google Antigravity** or **VS Code**: edit the project files.
- **Python 3.10+**: run the app locally.
- **Streamlit**: web dashboard framework.
- **Kaggle**: optional source for a real historical demand dataset later.
- **GitHub**: optional version control and portfolio sharing.
- **Streamlit Community Cloud**: optional deployment after testing.

You do not need Kaggle to run this starter project because sample CSV files are included.

## Run on Windows
1. Install Python 3.10 or newer from the official Python website and enable **Add Python to PATH** during setup.
2. Extract the ZIP file.
3. Open the extracted `ai_supply_chain_optimizer` folder in Antigravity or VS Code.
4. Open a terminal in that folder and run:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

5. Your browser should open the dashboard. If not, copy the local address shown in the terminal (usually `http://localhost:8501`) into your browser.

If `python` is not recognized, try `py` instead:
```bash
py -m venv .venv
```

## Project files
- `app.py` — complete Streamlit dashboard, forecasting logic, supplier scoring, CSV downloads, and purchase-order demo.
- `ml_utils.py` — Random Forest demand model training, chronological train/test split evaluation, and multi-day autoregressive forecasting.
- `test_reorder_math.py` — unit tests for baseline reorder math formulas.
- `test_ml_forecast.py` — unit tests for data preprocessing, Random Forest training/evaluation, and reorder math integration.
- `requirements.txt` — Python packages.
- `data/sales_history.csv` — synthetic 90-day daily sales data.
- `data/inventory.csv` — current inventory, supplier lead time, cost, and stock-cover settings.
- `data/suppliers.csv` — synthetic supplier information and sustainability proxies.

## CSV formats for your own data
### sales_history.csv
Required columns: `date`, `product_id`, `product_name`, `units_sold`

### inventory.csv
Required columns: `product_id`, `product_name`, `stock_on_hand`, `lead_time_days`, `unit_cost`, `supplier_category`, `target_cover_days`, `safety_stock_days`

Keep the column names as shown when uploading your own CSV files in the sidebar.

## How the calculations work
- Average daily demand is estimated from the most recent 14 days and lightly adjusted for a linear trend over the latest 45 days.
- Reorder point = forecast daily demand × (supplier lead time + safety-stock days).
- Suggested quantity = max(0, forecast daily demand × (lead time + target cover days + safety-stock days) − current stock).
- Supplier ranking blends relative cost, delivery speed, reliability, packaging score, and distance. A slider controls sustainability priority.

This is an educational baseline, not a validated forecasting system. For a stronger research project, compare it against a moving-average baseline and a model such as Random Forest/XGBoost, use a chronological train/test split, and report MAE or RMSE.

## Kaggle: optional next step
Search Kaggle for datasets using terms such as **retail demand forecasting**, **store item demand**, or **supply chain inventory**. Inspect the dataset's license, column definitions, date coverage, and missing values before using it. Real Kaggle datasets may not match this project's CSV columns; clean and map them before uploading.

## Suggested college-project title
**SupplyWise AI: Demand Forecasting, Smart Reordering, and Sustainable Supplier Optimization**

## Antigravity prompt you can use
Open this folder in Antigravity and paste:

"Review this Python Streamlit project. Explain the code to a beginner, check for bugs, run the app if possible, and improve it without removing its existing features. Keep purchase orders simulated only; do not connect to real suppliers or send real orders. Add clear comments and basic tests for the reorder-point and suggested-quantity calculations."

## Suggested future improvements
1. Add forecast backtesting and MAE/RMSE evaluation.
2. Compare moving average, linear trend, and Random Forest/XGBoost.
3. Add minimum order quantity and supplier capacity constraints.
4. Add a product-level history page and inventory CSV export.
5. Add login, database persistence, audit logging, and approval workflows before considering real procurement.
