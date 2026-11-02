"""Explainable diabetes risk screening dashboard (educational prototype)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from generate_dataset import FEATURE_COLUMNS
from lifestyle_simulator import LIFESTYLE_LEVERS, LifestyleSimulator
from planner import PlanBuilder
from predictor import BAND_EXPLANATION, RiskPredictor
from shap_explain import FEATURE_LABELS, RiskExplainer

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
DATA_PATH = ROOT / "data" / "diabetes.csv"

st.set_page_config(
    page_title="Diabetes Risk Advisor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

DISCLAIMER = (
    "Educational research prototype only. This is a risk-screening and teaching tool, "
    "not a medical device, diagnosis, or prescription system. Predictions are model "
    "probabilities from SYNTHETIC fused data. They are not clinical certainty."
)

SAMPLE_PROFILE = {
    "pregnancies": 2,
    "glucose": 148.0,
    "blood_pressure": 82.0,
    "skin_thickness": 32.0,
    "insulin": 180.0,
    "bmi": 34.5,
    "diabetes_pedigree": 0.9,
    "age": 48,
    "sleep_hours": 5.5,
    "resting_hr": 84.0,
    "daily_steps": 4200,
    "active_minutes": 15,
    "hrv_ms": 32.0,
}


def inject_styles() -> None:
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');
          html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
          .block-container { padding-top: 1.15rem; padding-bottom: 2rem; max-width: 1220px; }
          h1, h2, h3 { font-family: 'Fraunces', Georgia, serif !important; letter-spacing: -0.02em; }
          .hero {
            background: linear-gradient(135deg, #0b3d3a 0%, #1a6b63 55%, #2d9b8f 100%);
            color: #f4fffd; border-radius: 18px; padding: 1.45rem 1.7rem; margin-bottom: 1rem;
            box-shadow: 0 12px 40px rgba(11, 61, 58, 0.25);
          }
          .hero h1 { color: #fff !important; margin: 0 0 0.35rem 0; font-size: 1.85rem; }
          .hero p { margin: 0; opacity: 0.92; font-size: 1.0rem; }
          .metric-card {
            background: #f3faf8; border: 1px solid #cfe8e3; border-radius: 14px;
            padding: 1rem 1.1rem; height: 100%;
          }
          .metric-card .label { color: #3d5c57; font-size: 0.85rem; font-weight: 500; }
          .metric-card .value { font-family: Fraunces, serif; font-size: 1.7rem; color: #0b3d3a; margin: 0.2rem 0; }
          .metric-card .sub { color: #5a7a74; font-size: 0.9rem; }
          .plan-item, .rec-item {
            background: #fff; border-left: 4px solid #1a6b63; padding: 0.85rem 1rem;
            margin-bottom: 0.65rem; border-radius: 0 10px 10px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
          }
          .disclaimer { color: #5b6e6a; font-size: 0.82rem; margin-top: 1.2rem; }
          div[data-testid="stTabs"] button { font-weight: 600; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_stack():
    predictor = RiskPredictor()
    explainer = RiskExplainer()
    simulator = LifestyleSimulator(predictor)
    planner = PlanBuilder(predictor, simulator)
    metrics_path = MODEL_DIR / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError("metrics.json is missing. Run: python train.py")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    sample = pd.read_csv(DATA_PATH) if DATA_PATH.exists() else None
    return predictor, explainer, simulator, planner, metrics, sample


def risk_gauge(prob: float, title: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 34, "family": "Fraunces"}},
            title={"text": title, "font": {"size": 14, "color": "#3d5c57"}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": "#0b3d3a"},
                "steps": [
                    {"range": [0, 33], "color": "#d8f3ec"},
                    {"range": [33, 66], "color": "#ffe9c7"},
                    {"range": [66, 100], "color": "#f7b6b6"},
                ],
            },
        )
    )
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def default_patient(sample: pd.DataFrame | None) -> dict:
    if sample is None or sample.empty:
        return dict(SAMPLE_PROFILE)
    at_risk = sample[sample["diabetes"] == 1] if "diabetes" in sample.columns else sample
    row = at_risk.sample(1, random_state=7).iloc[0] if len(at_risk) else sample.iloc[0]
    out = {}
    for col in FEATURE_COLUMNS:
        value = row[col]
        if col in {"pregnancies", "age", "daily_steps", "active_minutes"}:
            out[col] = int(value)
        else:
            out[col] = float(value)
    return out


def collect_inputs(defaults: dict, key_prefix: str) -> dict:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Clinical / lab-style inputs**")
        pregnancies = st.number_input("Pregnancies", 0, 15, int(defaults["pregnancies"]), key=f"{key_prefix}_preg")
        glucose = st.slider("Fasting glucose (mg/dL)", 70, 220, int(defaults["glucose"]), key=f"{key_prefix}_glu")
        blood_pressure = st.slider("Diastolic BP (mmHg)", 40, 120, int(defaults["blood_pressure"]), key=f"{key_prefix}_bp")
        skin_thickness = st.slider("Skin thickness (mm)", 0, 80, int(defaults["skin_thickness"]), key=f"{key_prefix}_skin")
        insulin = st.slider("Insulin (µU/mL)", 0, 600, int(defaults["insulin"]), key=f"{key_prefix}_ins")
        bmi = st.slider("BMI", 18.0, 50.0, float(defaults["bmi"]), 0.1, key=f"{key_prefix}_bmi")
        diabetes_pedigree = st.slider(
            "Family history score (pedigree)", 0.05, 2.5, float(defaults["diabetes_pedigree"]), 0.01, key=f"{key_prefix}_ped"
        )
        age = st.slider("Age", 21, 80, int(defaults["age"]), key=f"{key_prefix}_age")
    with c2:
        st.markdown("**Lifestyle / wearable-style inputs**")
        st.caption("These fields are structured so real wearable data could be plugged in later. Values here are user-entered, not device-verified.")
        sleep_hours = st.slider("Sleep (hours/night)", 3.5, 10.0, float(defaults["sleep_hours"]), 0.1, key=f"{key_prefix}_slp")
        resting_hr = st.slider("Resting HR (bpm)", 48, 110, int(defaults["resting_hr"]), key=f"{key_prefix}_rhr")
        daily_steps = st.slider("Daily steps", 800, 18000, int(defaults["daily_steps"]), 100, key=f"{key_prefix}_stp")
        active_minutes = st.slider("Active minutes/day", 0, 150, int(defaults["active_minutes"]), key=f"{key_prefix}_act")
        hrv_ms = st.slider("HRV (ms)", 15.0, 100.0, float(defaults["hrv_ms"]), 0.5, key=f"{key_prefix}_hrv")

    return {
        "pregnancies": int(pregnancies),
        "glucose": float(glucose),
        "blood_pressure": float(blood_pressure),
        "skin_thickness": float(skin_thickness),
        "insulin": float(insulin),
        "bmi": float(bmi),
        "diabetes_pedigree": float(diabetes_pedigree),
        "age": int(age),
        "sleep_hours": float(sleep_hours),
        "resting_hr": float(resting_hr),
        "daily_steps": int(daily_steps),
        "active_minutes": int(active_minutes),
        "hrv_ms": float(hrv_ms),
    }


def validate_features(features: dict) -> list[str]:
    issues: list[str] = []
    if features["bmi"] < 18.5:
        issues.append("BMI is unusually low for this screening form.")
    if features["glucose"] < 70:
        issues.append("Glucose is below the form minimum.")
    if features["age"] < 21:
        issues.append("Age is below the supported range for this demo model.")
    return issues


def render_dashboard(predictor, planner, explainer, features, risk, band, plan) -> None:
    c1, c2, c3 = st.columns([1.15, 1, 1])
    with c1:
        st.plotly_chart(risk_gauge(risk, "Model diabetes risk probability"), use_container_width=True)
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="label">Risk category (model bins)</div>
              <div class="value">{band}</div>
              <div class="sub">Predicted probability {risk:.1%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(BAND_EXPLANATION)
    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="label">Hypothetical lifestyle scenario</div>
              <div class="value">{plan.projected_risk:.0%}</div>
              <div class="sub">{plan.total_drop_pp:+.1f} pp on the model score → {plan.projected_band}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("A change in the model's score is not a proven medical risk reduction.")

    st.info(plan.summary)

    st.markdown("#### Key factors for this profile")
    try:
        explanation = explainer.explain(features)
        top = explanation.head(4)
        chips = ", ".join(f"{row.label} ({row.direction.split('contributed to a ')[-1]})" for _, row in top.iterrows())
        st.write(chips)
        for line in explainer.narrative(explanation, top_k=3):
            st.write(f"- {line}")
    except Exception as exc:
        st.warning(f"Could not generate an individual explanation: {exc}")


def render_explanation(explainer, features) -> None:
    st.subheader("How the model used these inputs")
    st.caption(
        "SHAP values describe how each feature contributed to this model's prediction. "
        "They do not prove that a feature causes diabetes."
    )
    if not explainer.available():
        st.warning(explainer.error or "SHAP is unavailable for the selected model.")
        return
    try:
        explanation = explainer.explain(features)
        global_imp = explainer.global_importance()
    except Exception as exc:
        st.warning(f"SHAP could not be generated: {exc}")
        return

    left, right = st.columns(2)
    with left:
        st.markdown("**Individual explanation**")
        plot_df = explanation.head(10).copy()
        fig = px.bar(
            plot_df.iloc[::-1],
            x="shap",
            y="label",
            orientation="h",
            color="shap",
            color_continuous_scale=["#1a6b63", "#e8f5f2", "#c0392b"],
            color_continuous_midpoint=0,
            labels={"shap": "Contribution to model output", "label": ""},
        )
        fig.update_layout(height=420, showlegend=False, coloraxis_showscale=False, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        for line in explainer.narrative(explanation, top_k=6):
            st.write(f"- {line}")
    with right:
        st.markdown("**Global explanation (mean |SHAP| on a training sample)**")
        fig_g = px.bar(
            global_imp.iloc[::-1],
            x="mean_abs_shap",
            y="label",
            orientation="h",
            color_discrete_sequence=["#0b3d3a"],
            labels={"mean_abs_shap": "Mean |SHAP|", "label": ""},
        )
        fig_g.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_g, use_container_width=True)
        st.caption(f"Explainer method: {explainer.method}. Global ranks can differ from this single profile.")


def render_whatif(simulator, features, risk) -> None:
    st.subheader("What If I Change My Lifestyle?")
    st.caption(
        "The model is scored again on edited inputs. According to the ML model, a hypothetical "
        "input scenario may produce a different predicted risk. That is not evidence that the "
        "same lifestyle change would change a real person's medical risk by that amount."
    )
    left, right = st.columns(2)
    with left:
        lever = st.selectbox(
            "Lifestyle factor",
            options=list(LIFESTYLE_LEVERS.keys()),
            format_func=lambda k: LIFESTYLE_LEVERS[k]["label"],
        )
        meta = LIFESTYLE_LEVERS[lever]
        new_val = st.slider(
            f"Hypothetical {meta['label']}",
            float(meta["min"]),
            float(meta["max"]),
            float(features[lever]),
            float(meta["step"]),
        )
        one = simulator.simulate_one(features, lever, new_val, baseline_risk=risk)
        st.metric("Current predicted risk", f"{one.baseline_risk:.1%}")
        st.metric(
            "Simulated predicted risk",
            f"{one.new_risk:.1%}",
            delta=f"{(one.new_risk - one.baseline_risk) * 100:+.1f} percentage points",
        )
        st.info(meta["tip"])
        curve = simulator.sensitivity_curve(features, lever)
        fig_c = px.line(
            curve,
            x="value",
            y="risk",
            markers=True,
            labels={"value": meta["label"], "risk": "Predicted probability"},
        )
        fig_c.add_vline(x=float(features[lever]), line_dash="dot", line_color="#888")
        fig_c.add_vline(x=new_val, line_dash="dash", line_color="#1a6b63")
        fig_c.update_layout(height=320, yaxis_tickformat=".0%", margin=dict(t=20))
        st.plotly_chart(fig_c, use_container_width=True)

    with right:
        st.markdown("#### Combined hypothetical scenario")
        combo_changes: dict[str, float] = {}
        for key, m in LIFESTYLE_LEVERS.items():
            use = st.checkbox(f"Edit {m['label']}", value=False, key=f"chk_{key}")
            if use:
                tgt = st.number_input(
                    f"New {m['label']}",
                    float(m["min"]),
                    float(m["max"]),
                    float(features[key]),
                    float(m["step"]),
                    key=f"num_{key}",
                )
                combo_changes[key] = float(tgt)
        if combo_changes:
            base, new_r, _ = simulator.what_if_combo(features, combo_changes)
            st.metric("Current predicted risk", f"{base:.1%}")
            st.metric("Simulated predicted risk", f"{new_r:.1%}")
            st.metric("Change", f"{(new_r - base) * 100:+.1f} percentage points")
            compare = pd.DataFrame({"Scenario": ["Current", "Simulated"], "Risk": [base, new_r]})
            fig_b = px.bar(
                compare,
                x="Scenario",
                y="Risk",
                text=compare["Risk"].map(lambda v: f"{v:.0%}"),
                color="Scenario",
                color_discrete_sequence=["#8aa8a2", "#0b3d3a"],
            )
            fig_b.update_layout(yaxis_tickformat=".0%", showlegend=False, height=300, margin=dict(t=10))
            st.plotly_chart(fig_b, use_container_width=True)
            st.success(
                "According to the ML model, this hypothetical input scenario produces a "
                f"{'lower' if new_r < base else 'higher' if new_r > base else 'similar'} predicted risk."
            )
        else:
            st.warning("Select at least one factor to compare a combined scenario.")


def render_performance(metrics: dict) -> None:
    st.subheader("Selected model (holdout test set)")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    m2.metric("Precision", f"{metrics['precision']:.3f}")
    m3.metric("Recall", f"{metrics['recall']:.3f}")
    m4.metric("F1-score", f"{metrics['f1']:.3f}")
    m5.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    st.info(metrics.get("screening_note", ""))
    st.caption(metrics.get("selection_rule", ""))

    left, right = st.columns(2)
    with left:
        cm = metrics["confusion_matrix"]
        cm_df = pd.DataFrame(cm, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"])
        fig_cm = px.imshow(cm_df, text_auto=True, color_continuous_scale="Greens", aspect="equal")
        fig_cm.update_layout(title="Confusion matrix", height=360, coloraxis_showscale=False)
        st.plotly_chart(fig_cm, use_container_width=True)
    with right:
        roc = metrics["roc_curve"]
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name="Model", line=dict(color="#0b3d3a", width=3)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(color="#888", dash="dash")))
        fig_roc.update_layout(
            title="ROC curve",
            xaxis_title="False positive rate",
            yaxis_title="True positive rate",
            height=360,
            margin=dict(t=40),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("#### Model comparison")
    st.caption("Selection used cross-validation on the training split. The table also shows holdout test scores.")
    comp = pd.DataFrame(metrics["comparison"])
    show_cols = [
        "model",
        "cv_roc_auc",
        "cv_recall",
        "cv_f1",
        "cv_accuracy",
        "test_roc_auc",
        "test_recall",
        "test_precision",
        "test_f1",
        "test_accuracy",
    ]
    present = [c for c in show_cols if c in comp.columns]
    formatted = comp[present].copy()
    for col in present:
        if col != "model":
            formatted[col] = formatted[col].map(lambda v: f"{v:.3f}")
    st.dataframe(formatted, use_container_width=True, hide_index=True)

    fig_cmp = px.bar(
        comp,
        x="model",
        y=["test_roc_auc", "test_recall", "test_f1"],
        barmode="group",
        labels={"value": "Score", "variable": "Metric", "model": ""},
        color_discrete_sequence=["#0b3d3a", "#7EB8A4", "#cfe8e3"],
    )
    fig_cmp.update_layout(yaxis_range=[0, 1.05], height=380, legend_title="")
    st.plotly_chart(fig_cmp, use_container_width=True)
    st.success(f"Selected model: **{metrics['selected_model']}**")


def render_recommendations(plan) -> None:
    st.subheader("Personalized lifestyle suggestions")
    st.caption("Suggestions are conservative, non-diagnostic, and not prescriptions or medication advice.")
    st.write(plan.summary)
    for rec in plan.recommendations:
        st.markdown(f'<div class="rec-item">{rec}</div>', unsafe_allow_html=True)
    if plan.items:
        st.markdown("#### Hypothetical input targets ranked by the model")
        for item in plan.items:
            st.markdown(
                f"""
                <div class="plan-item">
                  <strong>#{item.priority} · {item.title}</strong><br/>
                  {item.action}<br/>
                  <span style="color:#1a6b63"><b>{item.estimated_risk_drop_pp:+.1f} pp</b> change in the model's
                  sequential score after prior steps</span><br/>
                  <em>{item.tip}</em>
                </div>
                """,
                unsafe_allow_html=True,
            )
        export = pd.DataFrame(
            [
                {
                    "priority": i.priority,
                    "factor": i.title,
                    "action": i.action,
                    "current": i.current,
                    "target": i.target,
                    "model_score_change_pp": i.estimated_risk_drop_pp,
                }
                for i in plan.items
            ]
        )
        st.download_button(
            "Download hypothetical plan (CSV)",
            export.to_csv(index=False),
            file_name="diabetes_risk_hypothetical_plan.csv",
            mime="text/csv",
        )


def render_about(metrics: dict, sample: pd.DataFrame | None) -> None:
    profile = metrics.get("data_profile", {})
    st.subheader("About this project")
    st.markdown(
        """
        **Explainable Early Diabetes Risk Prediction Using Clinical and Lifestyle Data
        with Personalized Risk Reduction Recommendations**

        This B.Tech prototype accepts clinical-style and lifestyle-style inputs, scores a
        trained machine-learning model, explains the prediction with SHAP, and lets you
        explore hypothetical input changes. It is a screening/education demo, not care.
        """
    )
    st.markdown("#### Dataset")
    st.write(
        "Training rows are **synthetic**. Clinical column names resemble the Pima Indians "
        "Diabetes table (glucose, BMI, blood pressure, pedigree, …) so the pipeline can later "
        "accept a real clinical extract. Lifestyle/wearable fields (steps, sleep, resting HR, HRV, "
        "active minutes) are simulated. They are not device uploads."
    )
    st.markdown(
        """
        **Why not use Pima alone?** The original Pima Indians Diabetes dataset is small, from a
        specific population, uses 0 as a missing-value sentinel in several columns, has no
        wearable signals, and is not a license to claim clinical performance. This project keeps
        those limitations in mind and labels its current table as synthetic.
        """
    )
    if profile:
        st.json(
            {
                "rows": profile.get("n_rows"),
                "features": profile.get("n_features"),
                "duplicates": profile.get("duplicate_rows"),
                "prevalence": profile.get("positive_prevalence"),
                "missing_values": profile.get("missing_values"),
                "source": profile.get("data_source"),
            }
        )
    st.markdown("#### Limitations")
    st.markdown(
        """
        - Synthetic labels follow a constructed formula, so test scores are optimistic versus messy clinical data.
        - No external validation, fairness audit, or clinical trial.
        - Risk bins are communication tools, not diagnostic cut-offs.
        - SHAP explains the model, not biological causation.
        """
    )
    st.markdown("#### Disclaimer")
    st.warning(DISCLAIMER)
    if sample is not None:
        st.caption(f"Loaded demonstration table: {len(sample):,} rows.")


def render_model_info(metrics: dict) -> None:
    profile = metrics.get("data_profile", {})
    st.sidebar.markdown("### Model information")
    st.sidebar.write(f"**Selected model:** {metrics.get('selected_model', 'unknown')}")
    st.sidebar.write(f"**Training date:** {metrics.get('training_date', 'unknown')}")
    st.sidebar.write(f"**Dataset size:** {profile.get('n_rows', metrics.get('n_train', '—'))} rows")
    st.sidebar.write(f"**Features:** {profile.get('n_features', len(FEATURE_COLUMNS))}")
    st.sidebar.write(
        f"Holdout ROC-AUC {metrics.get('roc_auc', 0):.3f} · recall {metrics.get('recall', 0):.3f}"
    )
    st.sidebar.caption("The number shown in the gauge is a predicted probability, not certainty.")


def main() -> None:
    inject_styles()
    st.markdown(
        """
        <div class="hero">
          <h1>Diabetes Risk Advisor</h1>
          <p>Explainable screening demo: clinical + lifestyle inputs, a trained model,
          SHAP explanations, what-if simulation, and conservative recommendations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        predictor, explainer, simulator, planner, metrics, sample = load_stack()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("In the project folder: `pip install -r requirements.txt` then `python train.py`.")
        return
    except Exception as exc:
        st.error(f"Could not load the trained pipeline: {exc}")
        st.info("Try retraining with `python train.py`.")
        return

    render_model_info(metrics)
    st.sidebar.caption(DISCLAIMER)

    defaults = default_patient(sample)
    if "features" not in st.session_state:
        st.session_state.features = defaults

    tab_dash, tab_input, tab_why, tab_whatif, tab_perf, tab_rec, tab_about = st.tabs(
        [
            "Dashboard",
            "Patient Input",
            "Model Explanation",
            "What-If Simulator",
            "Model Performance",
            "Recommendations",
            "About",
        ]
    )

    with tab_input:
        st.subheader("Patient / demo inputs")
        st.caption("Values stay within slider ranges so the form cannot submit impossible numbers for this demo.")
        drafted = collect_inputs(st.session_state.features, "main")
        if st.button("Estimate risk", type="primary"):
            problems = validate_features(drafted)
            for msg in problems:
                st.warning(msg)
            st.session_state.features = drafted
            st.success("Inputs saved. Open Dashboard to view the model probability.")
        st.caption("A sample higher-risk profile is loaded at start so a viva demo is never blank.")

    features = st.session_state.features
    try:
        risk = predictor.predict_proba(features)
        band = predictor.risk_band(risk)
        plan = planner.build(features)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    with tab_dash:
        render_dashboard(predictor, planner, explainer, features, risk, band, plan)
    with tab_why:
        render_explanation(explainer, features)
    with tab_whatif:
        render_whatif(simulator, features, risk)
    with tab_perf:
        render_performance(metrics)
    with tab_rec:
        render_recommendations(plan)
    with tab_about:
        render_about(metrics, sample)

    labels = ", ".join(FEATURE_LABELS[c] for c in FEATURE_COLUMNS)
    st.caption(f"Features used by the model: {labels}")
    st.markdown(f'<p class="disclaimer">{DISCLAIMER}</p>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
