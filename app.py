import io
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import ml_utils

DATA_DIR = APP_DIR

st.set_page_config(page_title="SupplyWise AI", page_icon="📦", layout="wide")

st.title("📦 SupplyWise AI")
st.caption("AI-assisted inventory forecasting, smart reordering, and sustainable supplier selection")
st.info("Demo project: sample data is synthetic. Purchase orders are simulated only; this app does not contact suppliers or spend money.")

@st.cache_data
def load_data():
    sales = pd.read_csv(DATA_DIR / "sales history.csv", parse_dates=["date"])
    inventory = pd.read_csv(DATA_DIR / "inventory.csv")
    suppliers = pd.read_csv(DATA_DIR / "suppliers.csv")
    return sales, inventory, suppliers

sales, inventory, suppliers = load_data()

@st.cache_resource
def train_model(sales_df):
    """Train and cache the Random Forest demand forecasting model."""
    try:
        model, metrics, evaluation, prepared = ml_utils.train_evaluate(sales_df)
        return model, metrics, evaluation, prepared, None
    except Exception as exc:
        return None, None, None, None, str(exc)

def forecast_daily_demand(history, forecast_days=14):
    """Simple, explainable trend forecast using recent weighted average + linear trend."""
    if history is None or history.empty:
        return 0.0
    y = history.sort_values("date")["units_sold"].astype(float).tail(45).to_numpy()
    if len(y) == 0:
        return 0.0
    recent = float(np.mean(y[-14:])) if len(y) >= 14 else float(np.mean(y))
    if len(y) >= 14:
        x = np.arange(len(y), dtype=float)
        slope = float(np.polyfit(x, y, 1)[0])
        trend_adjustment = np.clip(slope * 7, -0.20 * max(recent, 1), 0.20 * max(recent, 1))
        predicted = 0.75 * recent + 0.25 * max(0.0, recent + trend_adjustment)
    else:
        predicted = recent
    return max(0.0, predicted)

def supplier_recommendations(category, base_unit_cost, quantity, lead_time_days, sustainability_weight):
    options = suppliers[suppliers["product_category"] == category].copy()
    if options.empty:
        return options
    options["estimated_unit_cost"] = options["unit_cost_factor"] * base_unit_cost
    options["estimated_order_cost"] = options["estimated_unit_cost"] * quantity
    options["lead_time_days"] = options["lead_time_days"].astype(float)
    # Normalize metrics to 0..1 where 1 is better.
    cost_score = 1 - (options["unit_cost_factor"] - options["unit_cost_factor"].min()) / max(0.01, options["unit_cost_factor"].max() - options["unit_cost_factor"].min())
    speed_score = 1 - (options["lead_time_days"] - options["lead_time_days"].min()) / max(1, options["lead_time_days"].max() - options["lead_time_days"].min())
    reliability_score = options["reliability_pct"] / 100
    plastic_score = 1 - (options["single_use_plastic_score"] - 1) / 4
    co2_score = 1 - (options["distance_km"] - options["distance_km"].min()) / max(1, options["distance_km"].max() - options["distance_km"].min())
    options["supplier_score"] = (
        0.30 * cost_score + 0.20 * speed_score + 0.20 * reliability_score
        + sustainability_weight * (0.18 * plastic_score + 0.12 * co2_score)
    ) / (1 + sustainability_weight * 0.30)
    return options.sort_values("supplier_score", ascending=False)

# Sidebar controls
st.sidebar.header("⚙️ Planning settings")
forecasting_engine = st.sidebar.selectbox(
    "Forecasting engine",
    ["Random Forest (ML)", "Heuristic Baseline"],
    index=0,
    help="Select whether inventory planning uses the trained Random Forest model or the heuristic baseline."
)
forecast_horizon = st.sidebar.slider("Forecast horizon (days)", 7, 30, 14)
sustainability_weight = st.sidebar.slider("Sustainability priority", 0.0, 2.0, 1.0, 0.1)
st.sidebar.caption("Higher sustainability priority gives more weight to packaging and distance.")

# Optional uploads
st.sidebar.header("📁 Optional CSV uploads")
uploaded_sales = st.sidebar.file_uploader("Replace sales history CSV", type=["csv"])
uploaded_inventory = st.sidebar.file_uploader("Replace inventory CSV", type=["csv"])

if uploaded_sales is not None:
    try:
        candidate = pd.read_csv(uploaded_sales, parse_dates=["date"])
        required = {"date", "product_id", "product_name", "units_sold"}
        if required.issubset(candidate.columns):
            sales = candidate.copy()
        else:
            st.sidebar.error(f"Sales CSV must contain: {', '.join(sorted(required))}")
    except Exception as exc:
        st.sidebar.error(f"Could not read sales CSV: {exc}")

if uploaded_inventory is not None:
    try:
        candidate = pd.read_csv(uploaded_inventory)
        required = {"product_id", "product_name", "stock_on_hand", "lead_time_days", "unit_cost", "supplier_category", "target_cover_days", "safety_stock_days"}
        if required.issubset(candidate.columns):
            inventory = candidate.copy()
        else:
            st.sidebar.error(f"Inventory CSV must contain: {', '.join(sorted(required))}")
    except Exception as exc:
        st.sidebar.error(f"Could not read inventory CSV: {exc}")

# Train the Random Forest model
rf_model, rf_metrics, rf_evaluation, rf_prepared, rf_error = train_model(sales)

# Build forecast and reorder table
product_forecasts = {}
rows = []
for _, item in inventory.iterrows():
    p_id = item["product_id"]
    hist = sales[sales["product_id"] == p_id] if "product_id" in sales.columns else pd.DataFrame()
    
    # Generate forecast using selected engine
    daily = None
    if forecasting_engine == "Random Forest (ML)" and rf_model is not None and rf_prepared is not None:
        try:
            f_df = ml_utils.forecast_product(rf_model, rf_prepared, p_id, forecast_horizon)
            product_forecasts[p_id] = f_df
            daily = float(f_df["forecast_units"].mean())
        except Exception:
            daily = None

    if daily is None:
        daily = forecast_daily_demand(hist, forecast_horizon)

    stock = float(item["stock_on_hand"])
    lead = float(item["lead_time_days"])
    safety_days = float(item["safety_stock_days"])
    cover_days = float(item["target_cover_days"])
    reorder_point = int(np.ceil(daily * (lead + safety_days)))
    suggested_qty = int(max(0, np.ceil(daily * (lead + cover_days + safety_days) - stock)))
    days_left = stock / daily if daily > 0 else float("inf")
    rows.append({
        "product_id": p_id,
        "product_name": item["product_name"],
        "stock_on_hand": int(stock),
        "forecast_units_per_day": round(daily, 2),
        "estimated_days_until_stockout": round(days_left, 1) if np.isfinite(days_left) else None,
        "lead_time_days": int(lead),
        "reorder_point": reorder_point,
        "suggested_order_qty": suggested_qty,
        "unit_cost": float(item["unit_cost"]),
        "supplier_category": item["supplier_category"],
        "estimated_order_value": round(suggested_qty * float(item["unit_cost"]), 2),
        "status": "REORDER NOW" if stock <= reorder_point else ("PLAN SOON" if days_left <= lead + safety_days + 5 else "STOCK OK")
    })
plan = pd.DataFrame(rows)

# KPI cards
total_skus = len(plan)
urgent = int((plan["status"] == "REORDER NOW").sum())
inventory_value = float((inventory["stock_on_hand"] * inventory["unit_cost"]).sum()) if not inventory.empty else 0.0
potential_order_value = float(plan["estimated_order_value"].sum()) if not plan.empty else 0.0
c1, c2, c3, c4 = st.columns(4)
c1.metric("Products monitored", total_skus)
c2.metric("Need reorder now", urgent)
c3.metric("Current inventory value", f"₹{inventory_value:,.0f}")
c4.metric("Suggested order value", f"₹{potential_order_value:,.0f}")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔮 Demand forecast", "🌱 Supplier optimizer", "🧾 Purchase order demo"])

with tab1:
    st.subheader("Inventory health")
    st.caption(f"Reorder recommendations currently driven by: **{forecasting_engine}**")
    
    # Product filter for Overview while preserving all-products default
    overview_options = ["All products (Fleet overview)"] + [
        f"{r['product_id']} — {r['product_name']}" for _, r in inventory.iterrows()
    ]
    overview_choice = st.selectbox(
        "Filter overview by product (or view all)",
        options=overview_options,
        index=0,
        key="overview_product_filter",
        help="View the complete inventory overview or focus on a single product's stock health."
    )

    if overview_choice == "All products (Fleet overview)":
        st.dataframe(plan, use_container_width=True, hide_index=True)
        chart = px.bar(plan, x="product_name", y="stock_on_hand", color="status", title="Current stock by product")
        chart.update_layout(xaxis_title="", yaxis_title="Units in stock")
        st.plotly_chart(chart, use_container_width=True)
        csv_bytes = plan.to_csv(index=False).encode("utf-8")
        st.download_button("Download complete reorder plan (CSV)", csv_bytes, "reorder_plan.csv", "text/csv")
    else:
        chosen_pid = overview_choice.split(" — ")[0]
        single_plan = plan[plan["product_id"] == chosen_pid]
        if not single_plan.empty:
            p_row = single_plan.iloc[0]
            st.dataframe(single_plan, use_container_width=True, hide_index=True)
            
            # Focused single-product KPI summary
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Current Stock", f"{p_row['stock_on_hand']} units", delta=p_row['status'], delta_color="normal" if p_row['status'] == "STOCK OK" else "inverse")
            k2.metric("Days Until Stockout", f"{p_row['estimated_days_until_stockout']} days" if p_row['estimated_days_until_stockout'] is not None else "No Stockout Risk")
            k3.metric("Reorder Point", f"{p_row['reorder_point']} units")
            k4.metric("Suggested Order Qty", f"{p_row['suggested_order_qty']} units", f"₹{p_row['estimated_order_value']:,.0f} value")
            
            # Stock vs Reorder Point chart for this product
            stock_comparison = pd.DataFrame([
                {"Metric": "Stock on Hand", "Units": p_row["stock_on_hand"]},
                {"Metric": "Reorder Point", "Units": p_row["reorder_point"]},
                {"Metric": "Suggested Order", "Units": p_row["suggested_order_qty"]}
            ])
            comp_fig = px.bar(stock_comparison, x="Metric", y="Units", color="Metric", title=f"{p_row['product_name']} ({chosen_pid}): Inventory Levels vs Reorder Threshold")
            comp_fig.update_layout(showlegend=False)
            st.plotly_chart(comp_fig, use_container_width=True)
            
            st.info(f"💡 For detailed historical trends, model validation, and AI demand projections for **{p_row['product_name']}**, switch to the **Demand forecast** tab.")
        else:
            st.warning("Selected product details not found in inventory plan.")

with tab2:
    st.subheader("Interactive Product Demand Forecasting & Model Evaluation")
    
    # Check if inventory has products
    if inventory.empty or "product_id" not in inventory.columns:
        st.warning("No inventory products available to forecast.")
    else:
        # Dynamic Product Selection Dropdown using product_id and product_name
        product_ids = inventory["product_id"].tolist()
        product_name_map = dict(zip(inventory["product_id"], inventory["product_name"]))
        
        selected_pid = st.selectbox(
            "Select product to analyze",
            options=product_ids,
            format_func=lambda pid: f"{pid} — {product_name_map.get(pid, 'Unknown Product')}",
            key="demand_forecast_product_select",
            help="Choose a product to update its historical sales, model validation metrics, actual vs predicted chart, and future demand forecast."
        )

        # Lookup selected product details
        selected_inv_rows = inventory[inventory["product_id"] == selected_pid]
        selected_plan_rows = plan[plan["product_id"] == selected_pid]
        
        if selected_inv_rows.empty:
            st.error(f"Product ID '{selected_pid}' not found in inventory records.")
        else:
            item = selected_inv_rows.iloc[0]
            plan_item = selected_plan_rows.iloc[0] if not selected_plan_rows.empty else None
            chosen_name = item["product_name"]
            
            # 1. Product Inventory & Reorder Summary Cards
            st.markdown(f"### 📦 Inventory & Reorder Status: **{chosen_name}** (`{selected_pid}`)")
            if plan_item is not None:
                r1, r2, r3, r4 = st.columns(4)
                stock_delta = plan_item["status"]
                delta_color = "normal" if stock_delta == "STOCK OK" else "inverse"
                r1.metric("Current Stock on Hand", f"{plan_item['stock_on_hand']:,} units", delta=stock_delta, delta_color=delta_color)
                days_disp = f"{plan_item['estimated_days_until_stockout']:.1f} days" if plan_item['estimated_days_until_stockout'] is not None else "No Stockout Risk"
                r2.metric("Estimated Days to Stockout", days_disp)
                r3.metric("Reorder Point", f"{plan_item['reorder_point']:,} units", help="When stock falls to or below this point, order immediately")
                r4.metric("Recommended Order Qty", f"{plan_item['suggested_order_qty']:,} units", f"₹{plan_item['estimated_order_value']:,.2f} est. cost")
                
                st.caption(
                    f"**Lead time:** {int(item['lead_time_days'])} days · "
                    f"**Safety stock target:** {int(item['safety_stock_days'])} days · "
                    f"**Cover target:** {int(item['target_cover_days'])} days · "
                    f"**Unit cost:** ₹{float(item['unit_cost']):.2f} · "
                    f"**Supplier category:** {item['supplier_category']}"
                )
            st.divider()

            # 2. Product-specific & Fleet Evaluation Metrics
            st.markdown(f"### 📈 Model Evaluation for {chosen_name} (`{selected_pid}`)")
            
            if rf_error:
                st.warning(f"Could not train Random Forest model: {rf_error}. Using baseline forecasting.")
            elif rf_evaluation is not None and not rf_evaluation.empty:
                prod_eval = rf_evaluation[rf_evaluation["product_id"] == selected_pid].dropna(subset=["units_sold", "predicted_units"])
                
                if not prod_eval.empty and len(prod_eval) > 0:
                    p_rf_mae = float(mean_absolute_error(prod_eval["units_sold"], prod_eval["predicted_units"]))
                    p_rf_rmse = float(np.sqrt(mean_squared_error(prod_eval["units_sold"], prod_eval["predicted_units"])))
                    p_base_mae = float(mean_absolute_error(prod_eval["units_sold"], prod_eval["baseline_units"]))
                    p_base_rmse = float(np.sqrt(mean_squared_error(prod_eval["units_sold"], prod_eval["baseline_units"])))
                    
                    mae_improvement = ((p_base_mae - p_rf_mae) / p_base_mae) * 100 if p_base_mae > 0 else 0
                    rmse_improvement = ((p_base_rmse - p_rf_rmse) / p_base_rmse) * 100 if p_base_rmse > 0 else 0
                    
                    e1, e2, e3, e4 = st.columns(4)
                    e1.metric(f"Random Forest MAE ({selected_pid})", f"{p_rf_mae:.2f}", delta=f"{mae_improvement:+.1f}% vs baseline", delta_color="inverse")
                    e2.metric(f"Random Forest RMSE ({selected_pid})", f"{p_rf_rmse:.2f}", delta=f"{rmse_improvement:+.1f}% vs baseline", delta_color="inverse")
                    e3.metric(f"Baseline MAE ({selected_pid})", f"{p_base_mae:.2f}")
                    e4.metric(f"Baseline RMSE ({selected_pid})", f"{p_base_rmse:.2f}")
                    
                    st.caption(f"Calculated on the chronological 20% holdout test set ({len(prod_eval)} days) for {chosen_name}.")
                else:
                    st.info(f"No holdout test evaluation records found for product {selected_pid}.")

                if rf_metrics is not None:
                    with st.expander("🌐 View overall fleet model metrics (All 5 Products combined)"):
                        f1, f2, f3, f4 = st.columns(4)
                        all_rf_mae = rf_metrics["Random Forest MAE"]
                        all_rf_rmse = rf_metrics["Random Forest RMSE"]
                        all_base_mae = rf_metrics["Baseline MAE"]
                        all_base_rmse = rf_metrics["Baseline RMSE"]
                        f_mae_diff = ((all_base_mae - all_rf_mae) / all_base_mae) * 100 if all_base_mae > 0 else 0
                        f_rmse_diff = ((all_base_rmse - all_rf_rmse) / all_base_rmse) * 100 if all_base_rmse > 0 else 0
                        f1.metric("Overall Fleet RF MAE", f"{all_rf_mae:.2f}", delta=f"{f_mae_diff:+.1f}% vs baseline", delta_color="inverse")
                        f2.metric("Overall Fleet RF RMSE", f"{all_rf_rmse:.2f}", delta=f"{f_rmse_diff:+.1f}% vs baseline", delta_color="inverse")
                        f3.metric("Overall Baseline MAE", f"{all_base_mae:.2f}")
                        f4.metric("Overall Baseline RMSE", f"{all_base_rmse:.2f}")
            st.divider()

            # Retrieve product's own sales history
            history = sales[sales["product_id"] == selected_pid].sort_values("date") if "product_id" in sales.columns else pd.DataFrame()

            # 3. Actual vs Predicted Sales Chart
            st.markdown(f"#### 🔍 Actual vs. Predicted Demand: **{chosen_name}** (Holdout Test Period)")
            if rf_evaluation is not None and not rf_evaluation.empty:
                prod_eval = rf_evaluation[rf_evaluation["product_id"] == selected_pid].sort_values("date")
                if not prod_eval.empty:
                    fig_eval = go.Figure()
                    fig_eval.add_trace(go.Scatter(
                        x=prod_eval["date"],
                        y=prod_eval["units_sold"],
                        mode="lines+markers",
                        name="Actual Sales (Test Set)",
                        line=dict(color="#1f77b4", width=2.5)
                    ))
                    fig_eval.add_trace(go.Scatter(
                        x=prod_eval["date"],
                        y=prod_eval["predicted_units"],
                        mode="lines+markers",
                        name="Random Forest Predicted",
                        line=dict(color="#2ca02c", width=2.5, dash="dash")
                    ))
                    fig_eval.add_trace(go.Scatter(
                        x=prod_eval["date"],
                        y=prod_eval["baseline_units"],
                        mode="lines",
                        name="7-day Rolling Baseline",
                        line=dict(color="#ff7f0e", width=1.5, dash="dot")
                    ))
                    fig_eval.update_layout(
                        title=f"{chosen_name} ({selected_pid}): Model Validation on Test Split",
                        xaxis_title="Date",
                        yaxis_title="Units Sold",
                        hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_eval, use_container_width=True)
                else:
                    st.info(f"No holdout validation data available for {chosen_name}.")
            else:
                st.info("Evaluation data is unavailable.")

            # 4. Product Historical Sales Chart
            st.markdown(f"#### 📊 Historical Daily Sales Trend: **{chosen_name}**")
            if history.empty:
                st.warning(f"No historical sales history found for product {selected_pid}.")
            else:
                hist_plot = history.copy()
                hist_plot["7_day_ma"] = hist_plot["units_sold"].rolling(7).mean()
                hist_plot["14_day_ma"] = hist_plot["units_sold"].rolling(14).mean()
                
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Scatter(
                    x=hist_plot["date"],
                    y=hist_plot["units_sold"],
                    mode="lines",
                    name="Daily Units Sold",
                    line=dict(color="#3366cc", width=1.5)
                ))
                fig_hist.add_trace(go.Scatter(
                    x=hist_plot["date"],
                    y=hist_plot["7_day_ma"],
                    mode="lines",
                    name="7-Day Moving Average",
                    line=dict(color="#ff9900", width=2)
                ))
                fig_hist.add_trace(go.Scatter(
                    x=hist_plot["date"],
                    y=hist_plot["14_day_ma"],
                    mode="lines",
                    name="14-Day Moving Average",
                    line=dict(color="#109618", width=2, dash="dash")
                ))
                fig_hist.update_layout(
                    title=f"{chosen_name} ({selected_pid}): Full Historical Sales Timeline ({len(hist_plot)} days recorded)",
                    xaxis_title="Date",
                    yaxis_title="Units Sold",
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            # 5. Future Demand Forecast Chart
            st.markdown(f"#### 🔮 Future Demand Forecast: **{chosen_name}** ({forecast_horizon} Days Ahead)")
            
            # Fetch / generate future forecast for this specific product
            if selected_pid in product_forecasts:
                future_rf = product_forecasts[selected_pid].copy()
            elif rf_model is not None and rf_prepared is not None:
                try:
                    future_rf = ml_utils.forecast_product(rf_model, rf_prepared, selected_pid, forecast_horizon)
                except Exception:
                    future_rf = None
            else:
                future_rf = None

            if history.empty:
                st.warning(f"Cannot generate future forecast without sales history for {selected_pid}.")
            else:
                recent_history = history.tail(45).copy()
                daily_baseline = forecast_daily_demand(history, forecast_horizon)
                future_dates = pd.date_range(start=history["date"].max() + pd.Timedelta(days=1), periods=forecast_horizon, freq="D")

                fig_future = go.Figure()
                fig_future.add_trace(go.Scatter(
                    x=recent_history["date"],
                    y=recent_history["units_sold"],
                    mode="lines",
                    name="Recent Historical Sales (Last 45 Days)",
                    line=dict(color="#1f77b4", width=2)
                ))
                fig_future.add_trace(go.Scatter(
                    x=future_dates,
                    y=[daily_baseline] * forecast_horizon,
                    mode="lines",
                    name="Baseline Flat Forecast",
                    line=dict(color="#ff7f0e", width=2, dash="dot")
                ))

                if future_rf is not None:
                    fig_future.add_trace(go.Scatter(
                        x=future_rf["date"],
                        y=future_rf["forecast_units"],
                        mode="lines+markers",
                        name="Random Forest Projected Demand",
                        line=dict(color="#2ca02c", width=2.5)
                    ))
                    avg_rf_daily = float(future_rf["forecast_units"].mean())
                    total_rf = float(future_rf["forecast_units"].sum())
                    st.success(
                        f"**{chosen_name} Demand Projection ({forecasting_engine}):** "
                        f"Average demand **{avg_rf_daily:.1f} units/day** · "
                        f"Total demand **{total_rf:.0f} units** across the next {forecast_horizon} days "
                        f"(vs. Heuristic Baseline: **{daily_baseline:.1f} units/day**)."
                    )
                else:
                    st.info(f"Estimated demand: **{daily_baseline:.1f} units/day** · **{daily_baseline * forecast_horizon:.0f} units** over the next {forecast_horizon} days.")

                fig_future.update_layout(
                    title=f"{chosen_name} ({selected_pid}): Historical Actuals and {forecast_horizon}-Day Future Projection",
                    xaxis_title="Date",
                    yaxis_title="Units",
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_future, use_container_width=True)

                if future_rf is not None:
                    with st.expander(f"📋 View day-by-day Random Forest forecast table for {chosen_name}"):
                        st.dataframe(future_rf, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Compare suppliers")
    product_choice = st.selectbox("Product category", sorted(inventory["supplier_category"].unique()), key="supplier_category")
    matching = inventory[inventory["supplier_category"] == product_choice]
    selected_product = st.selectbox("Product to estimate order for", matching["product_name"].tolist(), key="supplier_product")
    product_row = inventory[inventory["product_name"] == selected_product].iloc[0]
    product_plan = plan[plan["product_name"] == selected_product].iloc[0]
    quantity = st.number_input("Order quantity", min_value=1, max_value=100000, value=max(1, int(product_plan["suggested_order_qty"])), key="supplier_quantity")
    ranked = supplier_recommendations(product_choice, float(product_row["unit_cost"]), int(quantity), float(product_row["lead_time_days"]), sustainability_weight)
    show = ranked[["supplier_name","estimated_unit_cost","estimated_order_cost","lead_time_days","distance_km","single_use_plastic_score","reliability_pct","supplier_score"]].copy()
    show = show.rename(columns={
        "supplier_name":"Supplier", "estimated_unit_cost":"Est. unit cost (₹)", "estimated_order_cost":"Est. order cost (₹)",
        "lead_time_days":"Lead time (days)", "distance_km":"Distance (km)",
        "single_use_plastic_score":"Plastic score (1=best)", "reliability_pct":"Reliability (%)", "supplier_score":"Weighted score"
    })
    st.dataframe(show.round(3), use_container_width=True, hide_index=True)
    if not ranked.empty:
        best = ranked.iloc[0]
        st.success(f"Recommended supplier: **{best['supplier_name']}** — estimated order cost ₹{best['estimated_order_cost']:,.2f}, lead time {int(best['lead_time_days'])} days, distance {int(best['distance_km'])} km.")
    st.caption("Supplier scores are illustrative and depend on the selected weights and the synthetic supplier data. Verify supplier claims before making real purchasing decisions.")

with tab4:
    st.subheader("Simulated purchase order")
    st.write("Select a product to generate a preview. This does not send an order to any real supplier.")
    po_product = st.selectbox("Product", plan["product_name"].tolist(), key="po_product")
    po_row = plan[plan["product_name"] == po_product].iloc[0]
    po_qty = st.number_input("Quantity to order", min_value=1, max_value=100000, value=max(1, int(po_row["suggested_order_qty"])), key="po_qty")
    po_category = po_row["supplier_category"]
    po_ranked = supplier_recommendations(po_category, float(po_row["unit_cost"]), int(po_qty), float(po_row["lead_time_days"]), sustainability_weight)
    supplier_name = st.selectbox("Supplier", po_ranked["supplier_name"].tolist(), key="po_supplier")
    if st.button("Generate purchase order preview", type="primary", key="btn_generate_po"):
        selected_supplier = po_ranked[po_ranked["supplier_name"] == supplier_name].iloc[0]
        po = pd.DataFrame([{
            "purchase_order_id": f"DEMO-PO-{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}",
            "product_id": po_row["product_id"],
            "product_name": po_product,
            "quantity": int(po_qty),
            "supplier": supplier_name,
            "estimated_unit_cost": round(float(selected_supplier["estimated_unit_cost"]), 2),
            "estimated_total": round(float(selected_supplier["estimated_unit_cost"]) * int(po_qty), 2),
            "estimated_lead_time_days": int(selected_supplier["lead_time_days"]),
            "status": "DRAFT - NOT SENT"
        }])
        st.session_state["latest_po"] = po
    if "latest_po" in st.session_state:
        st.dataframe(st.session_state["latest_po"], use_container_width=True, hide_index=True)
        po_csv = st.session_state["latest_po"].to_csv(index=False).encode("utf-8")
        st.download_button("Download purchase order preview (CSV)", po_csv, "purchase_order_demo.csv", "text/csv")
        st.warning("For safety, real automatic ordering should require authenticated supplier integrations, approval limits, audit logs, and human approval.")

with st.expander("📚 Project methodology and limitations"):
    st.markdown("""
    **Demand forecasting:** A Random Forest Regressor trains on historical daily sales using temporal and lag features (day of week, day of year, lag 1, lag 7, 7-day rolling mean, 14-day rolling mean). The model is evaluated on a chronological 80/20 train/test split and measured against a 7-day rolling baseline using MAE and RMSE. Future multi-day demand is predicted autoregressively.

    **Product selection & switching:** Selecting an individual SKU dynamically updates its stock status, stockout timeline, reorder point, product-specific MAE/RMSE test accuracy, historical daily trend, holdout validation curve, and multi-day future forecast using that product's specific historical sales series.

    **Reorder point** = forecast average daily demand × (supplier lead time + safety-stock days).

    **Suggested order quantity** = max(0, forecast average daily demand × (lead time + target stock-cover days + safety-stock days) − current stock).

    **Supplier ranking** blends relative price, delivery speed, reliability, packaging score, and distance. The sustainability slider changes the relative importance of environmental factors.

    **Limitations:** sample data is synthetic; CO₂ is a rough proxy based on distance and a supplied factor; supplier data is not verified; orders are only simulated. Before real use, add promotions/seasonality data, minimum order quantities, supplier contract validation, authentication, and secure procurement APIs.
    """)
