import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta
import json
import os
from streamlit.runtime.scriptrunner import RerunData, RerunException
from streamlit.components.v1 import html
import base64


def generate_leads(
    current_customers,
    current_month,
    outbound_lead_volume,
    paid_ads_lead_volume,
    seo_lead_volume,
    outbound_growth_rate,
    paid_ads_growth_rate,
    seo_growth_rate,
    referral_probability,
    referral_leads_per_customer,
):
    leads = {}

    # Outbound leads
    leads["outbound"] = (
        outbound_lead_volume * (1 + outbound_growth_rate) ** current_month
    )

    # Paid ads leads
    leads["paid_ads"] = (
        paid_ads_lead_volume * (1 + paid_ads_growth_rate) ** current_month
    )

    # SEO leads
    leads["seo"] = seo_lead_volume * (1 + seo_growth_rate) ** current_month

    # Referral leads (now treated as demos)
    leads["referral_demo"] = (
        current_customers * referral_probability * referral_leads_per_customer
    )

    return pd.Series(leads)


def run_pipeline(params, leads):
    months = params["projection_months"]
    pipeline_stages = ["leads", "demo", "trial", "closed_won"]
    lead_sources = ["outbound", "paid_ads", "seo"]

    # Initialize pipeline_by_source
    pipeline_by_source = pd.DataFrame(
        index=range(months),
        columns=[
            f"{stage}_{source}" for stage in pipeline_stages for source in lead_sources
        ]
        + ["demo_referral", "trial_referral", "closed_won_referral"],
    )

    # Fill in leads for each source
    for source in lead_sources:
        pipeline_by_source[f"leads_{source}"] = leads[source].fillna(0)

    # Fill in referral demos
    pipeline_by_source["demo_referral"] = leads["referral_demo"].fillna(0)

    # Process each stage of the pipeline
    for i, stage in enumerate(pipeline_stages[1:]):
        prev_stage = pipeline_stages[i]
        conversion_rate = params[f"{prev_stage}_to_{stage.lower()}_rate"]

        for source in lead_sources:
            pipeline_by_source[f"{stage}_{source}"] = (
                pipeline_by_source[f"{prev_stage}_{source}"] * conversion_rate
            ).fillna(0)

        # Handle referrals separately
        if stage == "trial":
            referral_boost = params.get(
                "referral_to_trial_boost", 1.0
            )  # Default to 1.0 if not provided
            pipeline_by_source["trial_referral"] = (
                pipeline_by_source["demo_referral"]
                * params["demo_to_trial_rate"]
                * referral_boost
            )
        elif stage == "closed_won":
            pipeline_by_source["closed_won_referral"] = (
                pipeline_by_source["trial_referral"]
                * params["trial_to_closed_won_rate"]
            )

    # Adjust for sales cycle time
    sales_cycle_time = params["sales_cycle_time"]
    for stage in pipeline_stages[1:]:  # Apply to all stages after 'leads'
        for source in lead_sources + ["referral"]:
            if f"{stage}_{source}" in pipeline_by_source.columns:
                pipeline_by_source[f"{stage}_{source}"] = (
                    pipeline_by_source[f"{stage}_{source}"]
                    .shift(sales_cycle_time)
                    .fillna(0)
                )

    # Compute overall pipeline results
    pipeline_results = pd.DataFrame(index=range(months), columns=pipeline_stages)
    for stage in pipeline_stages:
        if stage == "leads":
            pipeline_results[stage] = pipeline_by_source[
                [f"{stage}_{source}" for source in lead_sources]
            ].sum(axis=1)
        else:
            pipeline_results[stage] = pipeline_by_source[
                [f"{stage}_{source}" for source in lead_sources + ["referral"]]
            ].sum(axis=1)

    return pipeline_results, pipeline_by_source


def run_model(params, lead_params, gmv_params):
    months = params["projection_months"]
    start_date = params["start_date"]

    dates = [start_date + timedelta(days=30 * i) for i in range(months)]

    mrr = [params["start_mrr"]]
    customers = [params["start_mrr"] / params["monthly_sales_per_user"]]
    customers_by_source = pd.DataFrame(
        index=range(months), columns=["outbound", "paid_ads", "seo", "referral"]
    )
    customers_by_source.iloc[0] = [customers[0], 0, 0, 0]

    new_customers_by_source = pd.DataFrame(
        index=range(months), columns=["outbound", "paid_ads", "seo", "referral"]
    )
    new_customers_by_source.iloc[0] = [0, 0, 0, 0]

    leads = pd.DataFrame(
        index=range(months), columns=["outbound", "paid_ads", "seo", "referral_demo"]
    )

    gmv_based_revenue = [0]

    for i in range(months):
        new_leads = generate_leads(
            current_customers=customers[-1],
            current_month=i,
            outbound_lead_volume=lead_params["outbound_lead_volume"],
            paid_ads_lead_volume=lead_params["paid_ads_lead_volume"],
            seo_lead_volume=lead_params["seo_lead_volume"],
            outbound_growth_rate=lead_params["outbound_growth_rate"],
            paid_ads_growth_rate=lead_params["paid_ads_growth_rate"],
            seo_growth_rate=lead_params["seo_growth_rate"],
            referral_probability=lead_params["referral_probability"],
            referral_leads_per_customer=lead_params["referral_leads_per_customer"],
        )

        # Set referral_demo to 0 for the first month
        if i == 0:
            new_leads["referral_demo"] = 0

        leads.loc[i] = new_leads

        pipeline_results, pipeline_by_source = run_pipeline(params, leads.loc[:i])

        new_customers = int(pipeline_results.loc[i, "closed_won"])
        churned_customers = int(customers[-1] * params["churn_rate"]) if i > 0 else 0
        current_customers = max(0, customers[-1] + new_customers - churned_customers)
        customers.append(current_customers)

        current_mrr = current_customers * params["monthly_sales_per_user"]
        mrr.append(current_mrr)

        # Calculate GMV-based revenue
        total_users = current_customers * gmv_params["users_per_customer"]
        total_gmv = total_users * gmv_params["gmv_per_user"]
        gmv_revenue = (
            total_gmv * gmv_params["feature_penetration"] * gmv_params["take_rate"]
        )
        gmv_based_revenue.append(gmv_revenue)

        for source in ["outbound", "paid_ads", "seo", "referral"]:
            new_customers_source = int(
                pipeline_by_source.loc[i, f"closed_won_{source}"]
            )
            new_customers_by_source.loc[i, source] = new_customers_source

            churned_customers_source = (
                int(customers_by_source.loc[i - 1, source] * params["churn_rate"])
                if i > 0
                else 0
            )
            customers_by_source.loc[i, source] = max(
                0,
                (customers_by_source.loc[i - 1, source] if i > 0 else 0)
                + new_customers_source
                - churned_customers_source,
            )

    # Prepare results DataFrame
    results = pd.DataFrame(
        {
            "Date": dates,
            "MRR": mrr[1:],  # Remove the initial MRR value
            "Customers": customers[1:],  # Remove the initial customer count
            "ARR": [m * 12 for m in mrr[1:]],
            "GMV-based Revenue": gmv_based_revenue[1:],
            "Total Revenue": [m + g for m, g in zip(mrr[1:], gmv_based_revenue[1:])],
            "New customers": pipeline_results["closed_won"].astype(int),
            "New customers from Outbound": pipeline_by_source[
                "closed_won_outbound"
            ].astype(int),
            "New customers from Paid_Ads": pipeline_by_source[
                "closed_won_paid_ads"
            ].astype(int),
            "New customers from SEO": pipeline_by_source["closed_won_seo"].astype(int),
            "New customers from Referral": pipeline_by_source[
                "closed_won_referral"
            ].astype(int),
            "Referral demos": leads["referral_demo"].astype(
                int
            ),  # Add this line to see referral demos
        }
    )

    return results, customers_by_source, new_customers_by_source


def find_1m_arr_date(results):
    target_arr = 1000000  # €1M ARR
    for index, row in results.iterrows():
        if row["ARR"] >= target_arr:
            return row["Date"]
    return None  # Return None if €1M ARR is not reached within the projection period


# New function to save scenarios to a file
def save_scenarios_to_file():
    scenarios_to_save = {
        name: {
            "params": {
                k: str(v) if isinstance(v, date) else v
                for k, v in scenario["params"].items()
            },
            "lead_params": scenario["lead_params"],
            "gmv_params": scenario["gmv_params"],
        }
        for name, scenario in st.session_state.scenarios.items()
    }
    with open("scenarios.json", "w") as f:
        json.dump(scenarios_to_save, f)


# New function to load scenarios from a file
def load_scenarios_from_file():
    if os.path.exists("scenarios.json"):
        with open("scenarios.json", "r") as f:
            scenarios = json.load(f)
        for name, scenario in scenarios.items():
            if "start_date" in scenario["params"]:
                scenario["params"]["start_date"] = date.fromisoformat(
                    scenario["params"]["start_date"]
                )
        return scenarios
    return {}


# Modified function to save a scenario
def save_scenario(name):
    model_params = {
        "start_date": st.session_state.start_date,
        "start_mrr": st.session_state.start_mrr,
        "monthly_sales_per_user": st.session_state.monthly_sales_per_user,
        "churn_rate": st.session_state.churn_rate,
        "projection_months": st.session_state.projection_months,
        "leads_to_demo_rate": st.session_state.leads_to_demo_rate,
        "demo_to_trial_rate": st.session_state.demo_to_trial_rate,
        "trial_to_closed_won_rate": st.session_state.trial_to_closed_won_rate,
        "referral_to_trial_boost": st.session_state.referral_to_trial_boost,
        "sales_cycle_time": st.session_state.sales_cycle_time,
    }

    lead_params = {
        "projection_months": st.session_state.projection_months,
        "outbound_lead_volume": st.session_state.outbound_lead_volume,
        "paid_ads_lead_volume": st.session_state.paid_ads_lead_volume,
        "seo_lead_volume": st.session_state.seo_lead_volume,
        "outbound_growth_rate": st.session_state.outbound_growth_rate,
        "paid_ads_growth_rate": st.session_state.paid_ads_growth_rate,
        "seo_growth_rate": st.session_state.seo_growth_rate,
        "referral_probability": st.session_state.referral_probability,
        "referral_leads_per_customer": st.session_state.referral_leads_per_customer,
    }

    gmv_params = {
        "users_per_customer": st.session_state.users_per_customer,
        "gmv_per_user": st.session_state.gmv_per_user,
        "feature_penetration": st.session_state.feature_penetration,
        "take_rate": st.session_state.take_rate,
    }

    st.session_state.scenarios[name] = {
        "params": model_params,
        "lead_params": lead_params,
        "gmv_params": gmv_params,
    }
    save_scenarios_to_file()
    st.success(f"Scenario '{name}' saved successfully!")


# Modified function to load a scenario
def load_scenario(name):
    scenario = st.session_state.scenarios.get(name)
    if scenario:
        return {
            "params": scenario["params"],
            "lead_params": scenario["lead_params"],
            "gmv_params": scenario["gmv_params"],
        }
    return None


# Initialize session state for scenarios if not exists
if "scenarios" not in st.session_state:
    st.session_state.scenarios = load_scenarios_from_file()

if "current_scenario" not in st.session_state:
    st.session_state.current_scenario = "Default"

if "last_loaded_scenario" not in st.session_state:
    st.session_state.last_loaded_scenario = ""


# Main app
st.title("Startup growth simulator")

# Create tabs for different pages
tab1, tab2 = st.tabs(["Main Model", "Scenario Comparison"])

with tab1:
    # Sidebar for scenario management
    st.sidebar.header("Scenario Management")

    # Create a new scenario or select an existing one
    scenario_action = st.sidebar.radio(
        "Choose action", ["Create New Scenario", "Load Existing Scenario"]
    )

    if scenario_action == "Create New Scenario":
        new_scenario_name = st.sidebar.text_input("New Scenario Name")
        create_scenario_button = st.sidebar.button("Create New Scenario")

        if create_scenario_button and new_scenario_name:
            # Initialize a new scenario with current parameters
            save_scenario(new_scenario_name)
            st.sidebar.success(f"New scenario '{new_scenario_name}' created!")
            st.session_state.current_scenario = new_scenario_name

    elif scenario_action == "Load Existing Scenario":
        load_scenario_select = st.sidebar.selectbox(
            "Select Scenario to Load", [""] + list(st.session_state.scenarios.keys())
        )

        if (
            load_scenario_select
            and load_scenario_select != st.session_state.last_loaded_scenario
        ):
            loaded_scenario = load_scenario(load_scenario_select)
            if loaded_scenario:
                # Update session state with loaded scenario values
                for key, value in loaded_scenario["params"].items():
                    if key in st.session_state:
                        if key.endswith("_rate") and key != "take_rate":
                            st.session_state[key] = value
                        else:
                            st.session_state[key] = value
                for key, value in loaded_scenario["lead_params"].items():
                    if key in st.session_state:
                        if key.endswith("_rate") or key == "referral_probability":
                            st.session_state[key] = value
                        else:
                            st.session_state[key] = value
                for key, value in loaded_scenario["gmv_params"].items():
                    if key in st.session_state:
                        if key in ["feature_penetration", "take_rate"]:
                            st.session_state[key] = value
                        else:
                            st.session_state[key] = value
                st.session_state.current_scenario = load_scenario_select
                st.session_state.last_loaded_scenario = load_scenario_select
                st.sidebar.success(f"Scenario '{load_scenario_select}' loaded!")

                # Use RerunException to trigger a rerun
                raise RerunException(RerunData())

    # Display current scenario and save option
    st.sidebar.write(f"Current Scenario: {st.session_state.current_scenario}")
    save_changes_button = st.sidebar.button("Save Changes to Current Scenario")

    if save_changes_button:
        save_scenario(st.session_state.current_scenario)
        st.sidebar.success(
            f"Changes saved to scenario '{st.session_state.current_scenario}'!"
        )

    # Sidebar for input parameters
    st.sidebar.header("Model Parameters")
    st.session_state.start_date = st.sidebar.date_input(
        "Start Date", value=st.session_state.get("start_date", date.today())
    )
    st.session_state.start_mrr = st.sidebar.number_input(
        "Starting MRR", value=st.session_state.get("start_mrr", 21000.0), step=1000.0
    )
    st.session_state.monthly_sales_per_user = st.sidebar.number_input(
        "Monthly Sales per User",
        value=st.session_state.get("monthly_sales_per_user", 250.0),
        step=10.0,
    )
    st.session_state.churn_rate = (
        st.sidebar.slider(
            "Monthly Churn Rate (%)",
            0.0,
            20.0,
            float(
                st.session_state.get("churn_rate", 0.03) * 100
            ),  # Multiply by 100 here
            0.1,
        )
        / 100  # Divide by 100 to store as decimal
    )
    st.session_state.projection_months = st.sidebar.slider(
        "Projection Months", 1, 24, st.session_state.get("projection_months", 6)
    )

    st.sidebar.header("Lead Generation Parameters")
    st.session_state.outbound_lead_volume = st.sidebar.number_input(
        "Monthly Outbound Lead Volume",
        value=st.session_state.get("outbound_lead_volume", 400),
        step=50,
    )
    st.session_state.paid_ads_lead_volume = st.sidebar.number_input(
        "Monthly Paid Ads Lead Volume",
        value=st.session_state.get("paid_ads_lead_volume", 50),
        step=50,
    )
    st.session_state.seo_lead_volume = st.sidebar.number_input(
        "Monthly SEO Lead Volume",
        value=st.session_state.get("seo_lead_volume", 20),
        step=50,
    )

    st.session_state.outbound_growth_rate = (
        st.sidebar.slider(
            "Outbound Lead Growth Rate (%)",
            0,
            20,
            int(st.session_state.get("outbound_growth_rate", 0.02) * 100),
            step=5,
        )
        / 100
    )
    st.session_state.paid_ads_growth_rate = (
        st.sidebar.slider(
            "Paid Ads Lead Growth Rate (%)",
            0,
            20,
            int(st.session_state.get("paid_ads_growth_rate", 0.02) * 100),
            step=5,
        )
        / 100
    )
    st.session_state.seo_growth_rate = (
        st.sidebar.slider(
            "SEO Lead Growth Rate (%)",
            0,
            20,
            int(st.session_state.get("seo_growth_rate", 0.02) * 100),
            step=5,
        )
        / 100
    )

    # New referral parameters
    st.session_state.referral_probability = (
        st.sidebar.slider(
            "Referral Probability (%)",
            0,
            20,
            int(st.session_state.get("referral_probability", 0.1) * 100),
            step=1,
        )
        / 100
    )
    st.session_state.referral_leads_per_customer = st.sidebar.slider(
        "Referral leads per Customer",
        0.0,
        5.0,
        float(st.session_state.get("referral_leads_per_customer", 1.0)),
    )

    st.sidebar.header("Sales Pipeline Parameters")
    st.session_state.leads_to_demo_rate = (
        st.sidebar.slider(
            "Leads to Demo Conversion (%)",
            0,
            100,
            int(st.session_state.get("leads_to_demo_rate", 0.2) * 100),
            step=5,
        )
        / 100
    )
    st.session_state.demo_to_trial_rate = (
        st.sidebar.slider(
            "Demo to Trial Conversion (%)",
            0,
            100,
            int(st.session_state.get("demo_to_trial_rate", 0.75) * 100),
            step=5,
        )
        / 100
    )
    st.session_state.trial_to_closed_won_rate = (
        st.sidebar.slider(
            "Trial to Closed Won Conversion (%)",
            0,
            100,
            int(st.session_state.get("trial_to_closed_won_rate", 0.85) * 100),
            step=5,
        )
        / 100
    )
    st.session_state.referral_to_trial_boost = st.sidebar.slider(
        "Referral to Trial Conversion Boost",
        1.0,
        2.0,
        float(st.session_state.get("referral_to_trial_boost", 1.2)),
        step=0.1,
    )
    st.session_state.sales_cycle_time = st.sidebar.slider(
        "Sales Cycle Time (Months)", 0, 12, st.session_state.get("sales_cycle_time", 1)
    )

    st.sidebar.header("GMV-based Revenue Parameters")
    st.session_state.users_per_customer = st.sidebar.number_input(
        "Users per Customer",
        value=st.session_state.get("users_per_customer", 3),
        step=1,
    )
    st.session_state.gmv_per_user = st.sidebar.number_input(
        "GMV per User (€)",
        value=st.session_state.get("gmv_per_user", 300000),
        step=10000,
    )
    st.session_state.feature_penetration = (
        st.sidebar.slider(
            "Feature Penetration (%)",
            0,
            100,
            int(st.session_state.get("feature_penetration", 0.1) * 100),
            step=5,
        )
        / 100
    )
    st.session_state.take_rate = (
        st.sidebar.slider(
            "Take Rate (%)",
            0.0,
            10.0,
            float(st.session_state.get("take_rate", 0.025) * 100),
            step=0.5,
        )
        / 100
    )

    # After collecting all input parameters, define model_params

    # Update model_params
    model_params = {
        "start_date": st.session_state.start_date,
        "start_mrr": st.session_state.start_mrr,
        "monthly_sales_per_user": st.session_state.monthly_sales_per_user,
        "churn_rate": st.session_state.churn_rate,
        "projection_months": st.session_state.projection_months,
        "leads_to_demo_rate": st.session_state.leads_to_demo_rate,
        "demo_to_trial_rate": st.session_state.demo_to_trial_rate,
        "trial_to_closed_won_rate": st.session_state.trial_to_closed_won_rate,
        "referral_to_trial_boost": st.session_state.referral_to_trial_boost,
        "sales_cycle_time": st.session_state.sales_cycle_time,
    }

    # Define lead_params and gmv_params here as well
    lead_params = {
        "projection_months": st.session_state.projection_months,
        "outbound_lead_volume": st.session_state.outbound_lead_volume,
        "paid_ads_lead_volume": st.session_state.paid_ads_lead_volume,
        "seo_lead_volume": st.session_state.seo_lead_volume,
        "outbound_growth_rate": st.session_state.outbound_growth_rate,
        "paid_ads_growth_rate": st.session_state.paid_ads_growth_rate,
        "seo_growth_rate": st.session_state.seo_growth_rate,
        "referral_probability": st.session_state.referral_probability,
        "referral_leads_per_customer": st.session_state.referral_leads_per_customer,
    }

    gmv_params = {
        "users_per_customer": st.session_state.users_per_customer,
        "gmv_per_user": st.session_state.gmv_per_user,
        "feature_penetration": st.session_state.feature_penetration,
        "take_rate": st.session_state.take_rate,
    }

    # Generate leads
    lead_params = {
        "projection_months": st.session_state.projection_months,
        "outbound_lead_volume": st.session_state.outbound_lead_volume,
        "paid_ads_lead_volume": st.session_state.paid_ads_lead_volume,
        "seo_lead_volume": st.session_state.seo_lead_volume,
        "outbound_growth_rate": st.session_state.outbound_growth_rate,
        "paid_ads_growth_rate": st.session_state.paid_ads_growth_rate,
        "seo_growth_rate": st.session_state.seo_growth_rate,
        "referral_probability": st.session_state.referral_probability,
        "referral_leads_per_customer": st.session_state.referral_leads_per_customer,
    }

    # Use st.cache_data to memoize the run_model function
    @st.cache_data
    def cached_run_model(model_params, lead_params, gmv_params):
        return run_model(model_params, lead_params, gmv_params)

    # Run the model for the current scenario
    results, customers_by_source, new_customers_by_source = cached_run_model(
        model_params, lead_params, gmv_params
    )

    # Display results for the current scenario
    st.subheader(f"Results for Scenario: {st.session_state.current_scenario}")
    st.subheader("Model Results")
    st.dataframe(results)

    # Calculate and display the date when ARR reaches €1M
    arr_1m_date = find_1m_arr_date(results)

    if arr_1m_date:
        months_to_1m = (arr_1m_date - st.session_state.start_date).days // 30
        days_to_1m = (arr_1m_date - st.session_state.start_date).days % 30
        st.success(f"🎉 €1M ARR to be reached on: {arr_1m_date.strftime('%B %d, %Y')}")
        st.success(f"Time to €1M ARR: {months_to_1m} months and {days_to_1m} days")
    else:
        st.warning("€1M ARR not reached within the projection period.")

    # Plot results
    # Customers Chart
    fig_customers = go.Figure()
    fig_customers.add_trace(
        go.Scatter(
            x=results["Date"],
            y=results["Customers"].round().astype(int),
            name="Customers",
        )
    )
    fig_customers.update_layout(
        title="Customers Over Time",
        yaxis_title="Number of Customers",
    )
    st.plotly_chart(fig_customers)

    # MRR Chart
    fig_mrr = go.Figure()
    fig_mrr.add_trace(
        go.Scatter(x=results["Date"], y=results["MRR"].round().astype(int), name="MRR")
    )

    # Add horizontal line for €1M ARR (MRR = €1M / 12)
    fig_mrr.add_hline(
        y=1000000 / 12,
        line_dash="dash",
        line_color="green",
        annotation_text="€1M ARR",
        annotation_position="right",
    )

    fig_mrr.update_layout(
        title="Monthly Recurring Revenue (MRR) Over Time",
        yaxis_title="MRR (€)",
    )
    st.plotly_chart(fig_mrr)

    # Total Revenue Chart (including GMV-based revenue)
    fig_total_revenue = go.Figure()
    fig_total_revenue.add_trace(
        go.Scatter(
            x=results["Date"],
            y=results["Total Revenue"].round().astype(int),
            name="Total Revenue",
        )
    )
    fig_total_revenue.update_layout(
        title="Total Revenue Over Time",
        yaxis_title="Total Revenue (€)",
    )
    st.plotly_chart(fig_total_revenue)

    # Revenue Streams Chart
    fig_revenue = go.Figure()
    fig_revenue.add_trace(
        go.Bar(x=results["Date"], y=results["MRR"].round().astype(int), name="MRR")
    )
    fig_revenue.add_trace(
        go.Bar(
            x=results["Date"],
            y=results["GMV-based Revenue"].round().astype(int),
            name="GMV-based Revenue",
        )
    )
    fig_revenue.update_layout(
        title="Revenue Streams Over Time",
        yaxis_title="Revenue (€)",
        barmode="stack",
    )
    st.plotly_chart(fig_revenue)

    # Lead Sources Chart (Monthly)
    st.subheader("Lead Sources Over Time (Monthly)")
    fig_leads = go.Figure()

    # Generate leads for each month
    lead_data = []
    for i, customers in enumerate(results["Customers"]):
        new_leads = generate_leads(
            current_customers=customers,
            current_month=i,
            outbound_lead_volume=lead_params["outbound_lead_volume"],
            paid_ads_lead_volume=lead_params["paid_ads_lead_volume"],
            seo_lead_volume=lead_params["seo_lead_volume"],
            outbound_growth_rate=lead_params["outbound_growth_rate"],
            paid_ads_growth_rate=lead_params["paid_ads_growth_rate"],
            seo_growth_rate=lead_params["seo_growth_rate"],
            referral_probability=lead_params["referral_probability"],
            referral_leads_per_customer=lead_params["referral_leads_per_customer"],
        )
        lead_data.append(new_leads)

    # Convert lead_data to a DataFrame and round to integers
    leads = pd.DataFrame(lead_data).round().astype(int)

    for source in leads.columns:
        fig_leads.add_trace(go.Bar(x=results["Date"], y=leads[source], name=source))
    fig_leads.update_layout(
        title="Lead Sources by Month",
        yaxis_title="Number of Leads",
        barmode="stack",
    )
    st.plotly_chart(fig_leads)

    # New chart: Customers by Channel (Monthly)
    st.subheader("New Customers by Acquisition Channel (Monthly)")
    fig_customers_by_channel = go.Figure()

    # Calculate new customers for each month and round to integers
    new_customers_monthly = (
        customers_by_source.diff()
        .fillna(customers_by_source.iloc[0])
        .round()
        .astype(int)
    )

    for source in new_customers_monthly.columns:
        fig_customers_by_channel.add_trace(
            go.Bar(
                x=results["Date"],
                y=new_customers_monthly[source],
                name=source,
            )
        )

    fig_customers_by_channel.update_layout(
        title="New Customers by Acquisition Channel (Monthly)",
        yaxis_title="Number of New Customers",
        barmode="stack",
    )
    st.plotly_chart(fig_customers_by_channel)

    # Pipeline Funnel Chart
    st.subheader("Sales Pipeline Funnel")
    pipeline_results, pipeline_by_source = run_pipeline(model_params, leads)

    # Calculate average monthly values
    average_pipeline_results = pipeline_results.mean().round().astype(int)

    funnel_data = average_pipeline_results.reset_index()
    funnel_data.columns = ["Stage", "Value"]
    fig_funnel = go.Figure(go.Funnel(y=funnel_data["Stage"], x=funnel_data["Value"]))
    fig_funnel.update_layout(title="Average Monthly Sales Funnel")
    st.plotly_chart(fig_funnel)


with tab2:
    st.header("Scenario Comparison")

    # Select scenarios to compare
    scenarios_to_compare = st.multiselect(
        "Select scenarios to compare", list(st.session_state.scenarios.keys())
    )

    if len(scenarios_to_compare) >= 2:
        # Display scenario parameters side by side
        st.subheader("Scenario Parameters Comparison")

        # Create DataFrames for each parameter category
        params_df = pd.DataFrame()

        for scenario in scenarios_to_compare:
            scenario_data = load_scenario(scenario)

            params_df[scenario] = pd.Series(
                {
                    "Start Date": scenario_data["params"]["start_date"],
                    "Start MRR": f"€{scenario_data['params']['start_mrr']:,.0f}",
                    "Monthly Sales per User": f"€{scenario_data['params']['monthly_sales_per_user']:,.0f}",
                    "Churn Rate": f"{scenario_data['params']['churn_rate']:.2%}",
                    "Projection Months": scenario_data["params"]["projection_months"],
                    "Outbound Lead Volume": scenario_data["lead_params"][
                        "outbound_lead_volume"
                    ],
                    "Paid Ads Lead Volume": scenario_data["lead_params"][
                        "paid_ads_lead_volume"
                    ],
                    "SEO Lead Volume": scenario_data["lead_params"]["seo_lead_volume"],
                    "Outbound Growth Rate": f"{scenario_data['lead_params']['outbound_growth_rate']:.2%}",
                    "Paid Ads Growth Rate": f"{scenario_data['lead_params']['paid_ads_growth_rate']:.2%}",
                    "SEO Growth Rate": f"{scenario_data['lead_params']['seo_growth_rate']:.2%}",
                    "Referral Probability": f"{scenario_data['lead_params']['referral_probability']:.2%}",
                    "Referral Leads per Customer": scenario_data["lead_params"][
                        "referral_leads_per_customer"
                    ],
                    "Leads to Demo Rate": f"{scenario_data['params']['leads_to_demo_rate']:.2%}",
                    "Demo to Trial Rate": f"{scenario_data['params']['demo_to_trial_rate']:.2%}",
                    "Trial to Closed Won Rate": f"{scenario_data['params']['trial_to_closed_won_rate']:.2%}",
                    "Referral to Trial Boost": scenario_data["params"][
                        "referral_to_trial_boost"
                    ],
                    "Sales Cycle Time": f"{scenario_data['params']['sales_cycle_time']} months",
                    "Users per Customer": scenario_data["gmv_params"][
                        "users_per_customer"
                    ],
                    "GMV per User": f"€{scenario_data['gmv_params']['gmv_per_user']:,.0f}",
                    "Feature Penetration": f"{scenario_data['gmv_params']['feature_penetration']:.2%}",
                    "Take Rate": f"{scenario_data['gmv_params']['take_rate']:.2%}",
                }
            )

        # Display parameter comparison table
        st.table(params_df)

        # Run models for selected scenarios
        comparison_results = {}
        for scenario in scenarios_to_compare:
            scenario_data = load_scenario(scenario)
            scenario_params = scenario_data["params"]
            scenario_lead_params = scenario_data["lead_params"]
            scenario_gmv_params = scenario_data["gmv_params"]

            scenario_results, _, _ = cached_run_model(
                scenario_params,
                scenario_lead_params,
                scenario_gmv_params,
            )
            comparison_results[scenario] = scenario_results

        # Create a table for side-by-side comparison of key metrics
        comparison_table = pd.DataFrame()
        for scenario, results in comparison_results.items():
            last_month = results.iloc[-1]
            first_month = results.iloc[0]
            arr_1m_date = find_1m_arr_date(results)

            if arr_1m_date:
                time_to_1m = (arr_1m_date - scenario_params["start_date"]).days
                months_to_1m = time_to_1m // 30
                days_to_1m = time_to_1m % 30
                time_to_1m_str = f"{months_to_1m} months, {days_to_1m} days"
            else:
                arr_1m_date = "Not reached"
                time_to_1m_str = "N/A"

            growth_rate = (last_month["MRR"] / first_month["MRR"]) ** (
                1 / len(results)
            ) - 1

            comparison_table[scenario] = pd.Series(
                {
                    "Final MRR": f"€{last_month['MRR']:,.0f}",
                    "Final ARR": f"€{last_month['ARR']:,.0f}",
                    "Final Customers": f"{last_month['Customers']:,.0f}",
                    "Date €1M ARR Reached": arr_1m_date,
                    "Time to €1M ARR": time_to_1m_str,
                    "Avg Monthly New Customers": f"{results['New customers'].mean():,.1f}",
                    "Monthly Growth Rate": f"{growth_rate:.2%}",
                    "Total Revenue at End of Period": f"€{last_month['Total Revenue']:,.0f}",
                }
            )

        st.subheader("Key Metrics Comparison")
        st.table(comparison_table)

        st.subheader("MRR Comparison")
        fig_mrr_comparison = go.Figure()
        for scenario, data in comparison_results.items():
            fig_mrr_comparison.add_trace(
                go.Scatter(x=data["Date"], y=data["MRR"], name=scenario)
            )
        fig_mrr_comparison.update_layout(
            title="MRR Comparison",
            yaxis_title="MRR (€)",
            yaxis_type="linear",
            yaxis=dict(tickformat="€,.0f"),
        )
        st.plotly_chart(fig_mrr_comparison)

        st.subheader("Customers Comparison")
        fig_customers_comparison = go.Figure()
        for scenario, data in comparison_results.items():
            fig_customers_comparison.add_trace(
                go.Scatter(x=data["Date"], y=data["Customers"], name=scenario)
            )
        fig_customers_comparison.update_layout(
            title="Customers Comparison", yaxis_title="Number of Customers"
        )
        st.plotly_chart(fig_customers_comparison)

        # Add a download button for comparison data
        csv = comparison_table.to_csv(index=True)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="scenario_comparison.csv">Download Comparison CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

    else:
        st.info("Please select at least two scenarios to compare.")
