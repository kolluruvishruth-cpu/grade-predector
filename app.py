from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "student_data.csv"
MODEL_PATH = BASE_DIR / "model.pkl"

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def build_model_bundle(data: pd.DataFrame) -> dict:
    features = [
        "attendance",
        "study_hours",
        "previous_marks",
        "assignments_completed",
        "sleep_hours",
    ]
    target = "final_grade"

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
    )
    model.fit(data[features], data[target])

    bundle = {
        "model": model,
        "features": features,
        "target": target,
        "metrics": {
            "r2_score": "run train_model.py",
            "mean_absolute_error": "run train_model.py",
        },
    }
    joblib.dump(bundle, MODEL_PATH)
    return bundle


@st.cache_resource
def load_model(data: pd.DataFrame):
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return build_model_bundle(data)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Manrope', sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(22, 163, 74, 0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(14, 116, 144, 0.20), transparent 30%),
                linear-gradient(135deg, #f4fbf8 0%, #eef7fb 100%);
        }

        .hero-card, .metric-card {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 20px;
            padding: 1.2rem 1.3rem;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
            backdrop-filter: blur(10px);
        }

        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.35rem;
        }

        .hero-subtitle {
            font-size: 1rem;
            color: #334155;
            margin-bottom: 0;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #14532d 100%);
        }

        [data-testid="stSidebar"] * {
            color: #f8fafc;
        }

        .small-note {
            color: #475569;
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(metrics: dict) -> None:
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-title">Student Performance Prediction</div>
            <p class="hero-subtitle">
                Forecast final grades using attendance, study habits, prior marks, assignment completion, and sleep patterns.
                The current Random Forest model reports R² = <strong>{metrics['r2_score']}</strong>
                and MAE = <strong>{metrics['mean_absolute_error']}</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def prediction_view(data: pd.DataFrame, model_bundle: dict) -> None:
    model = model_bundle["model"]

    st.subheader("Performance Estimator")
    left, right = st.columns([1, 1], gap="large")

    with left:
        attendance = st.slider("Attendance (%)", 50, 100, 85)
        study_hours = st.slider("Study Hours / Day", 0.5, 10.0, 4.0, 0.1)
        previous_marks = st.slider("Previous Marks", 40, 100, 75)
        assignments_completed = st.slider("Assignments Completed", 0, 10, 8)
        sleep_hours = st.slider("Sleep Hours / Day", 4.0, 10.0, 7.0, 0.1)

        user_input = pd.DataFrame(
            [
                {
                    "attendance": attendance,
                    "study_hours": study_hours,
                    "previous_marks": previous_marks,
                    "assignments_completed": assignments_completed,
                    "sleep_hours": sleep_hours,
                }
            ]
        )

        if st.button("Predict Final Grade", use_container_width=True):
            predicted_grade = float(model.predict(user_input)[0])
            st.session_state["predicted_grade"] = round(predicted_grade, 2)
            st.session_state["user_input"] = user_input.iloc[0].to_dict()

    with right:
        predicted_grade = st.session_state.get("predicted_grade")
        if predicted_grade is None:
            st.info("Adjust the inputs and click **Predict Final Grade** to see the result.")
            return

        benchmark = data["final_grade"].mean()
        top_quartile = data["final_grade"].quantile(0.75)

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=predicted_grade,
                number={"suffix": " / 100"},
                title={"text": "Predicted Final Grade"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#15803d"},
                    "steps": [
                        {"range": [0, 60], "color": "#fee2e2"},
                        {"range": [60, 80], "color": "#fef3c7"},
                        {"range": [80, 100], "color": "#dcfce7"},
                    ],
                    "threshold": {
                        "line": {"color": "#0f172a", "width": 4},
                        "thickness": 0.75,
                        "value": predicted_grade,
                    },
                },
            )
        )
        gauge.update_layout(height=310, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(gauge, use_container_width=True)

        comparison_df = pd.DataFrame(
            {
                "Category": ["Your Prediction", "Dataset Average", "Top Quartile"],
                "Grade": [predicted_grade, benchmark, top_quartile],
            }
        )
        comparison_chart = px.bar(
            comparison_df,
            x="Category",
            y="Grade",
            color="Category",
            color_discrete_sequence=["#15803d", "#0891b2", "#f59e0b"],
            text_auto=".2f",
        )
        comparison_chart.update_layout(showlegend=False, height=320)
        st.plotly_chart(comparison_chart, use_container_width=True)

    input_snapshot = st.session_state.get("user_input")
    if input_snapshot:
        st.markdown("#### Input Snapshot")
        snapshot_df = pd.DataFrame(
            {
                "Feature": list(input_snapshot.keys()),
                "Value": list(input_snapshot.values()),
            }
        )
        snapshot_chart = px.line_polar(
            snapshot_df,
            r="Value",
            theta="Feature",
            line_close=True,
            template="plotly_white",
        )
        snapshot_chart.update_traces(fill="toself", line_color="#0f766e")
        snapshot_chart.update_layout(height=420)
        st.plotly_chart(snapshot_chart, use_container_width=True)


def dashboard_view(data: pd.DataFrame) -> None:
    st.subheader("Data Visualization Dashboard")
    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Students in Dataset", len(data))
    metric_2.metric("Average Final Grade", f"{data['final_grade'].mean():.2f}")
    metric_3.metric("Average Attendance", f"{data['attendance'].mean():.2f}%")

    chart_1, chart_2 = st.columns(2)
    with chart_1:
        scatter = px.scatter(
            data,
            x="study_hours",
            y="final_grade",
            size="attendance",
            color="previous_marks",
            trendline="ols",
            color_continuous_scale="Viridis",
            title="Study Hours vs Final Grade",
        )
        scatter.update_layout(height=400)
        st.plotly_chart(scatter, use_container_width=True)

    with chart_2:
        corr = data.corr(numeric_only=True)
        heatmap = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="YlGnBu",
            title="Feature Correlation Heatmap",
        )
        heatmap.update_layout(height=400)
        st.plotly_chart(heatmap, use_container_width=True)

    chart_3, chart_4 = st.columns(2)
    with chart_3:
        histogram = px.histogram(
            data,
            x="final_grade",
            nbins=15,
            color_discrete_sequence=["#0f766e"],
            title="Final Grade Distribution",
        )
        histogram.update_layout(height=380)
        st.plotly_chart(histogram, use_container_width=True)

    with chart_4:
        assignment_chart = px.box(
            data,
            x="assignments_completed",
            y="final_grade",
            color="assignments_completed",
            color_discrete_sequence=px.colors.qualitative.Safe,
            title="Assignments Completed vs Final Grade",
        )
        assignment_chart.update_layout(height=380, showlegend=False)
        st.plotly_chart(assignment_chart, use_container_width=True)

    st.markdown("#### Dataset Preview")
    st.dataframe(data, use_container_width=True)


def insights_view(data: pd.DataFrame) -> None:
    st.subheader("Performance Insights")
    best_attendance = data.nlargest(5, "final_grade")[
        ["attendance", "study_hours", "previous_marks", "final_grade"]
    ]
    st.dataframe(best_attendance, use_container_width=True)

    avg_sleep = data.groupby("sleep_hours", as_index=False)["final_grade"].mean()
    sleep_chart = px.area(
        avg_sleep,
        x="sleep_hours",
        y="final_grade",
        color_discrete_sequence=["#16a34a"],
        title="Average Grade by Sleep Hours",
    )
    sleep_chart.update_layout(height=380)
    st.plotly_chart(sleep_chart, use_container_width=True)

    st.markdown(
        """
        <div class="metric-card small-note">
            Balanced routines tend to perform better in this sample: high attendance, steady study hours,
            strong previous marks, and consistent assignment submission all align with higher final grades.
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_styles()
    data = load_data()
    model_bundle = load_model(data)

    with st.sidebar:
        st.title("Navigation")
        page = st.radio(
            "Go to",
            ["Overview", "Predict Grade", "Dashboard", "Insights"],
        )
        st.markdown("---")
        st.markdown("Built with Streamlit and Random Forest Regression.")

    render_header(model_bundle["metrics"])

    if page == "Overview":
        st.markdown(
            """
            ### Project Overview
            This app predicts student final grades from academic and lifestyle inputs. Use the sidebar to jump to the
            prediction screen, explore the underlying dataset, and review performance insights through interactive charts.
            """
        )
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
                <div class="metric-card">
                    <h4>Model Inputs</h4>
                    <p>attendance, study_hours, previous_marks, assignments_completed, sleep_hours</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                """
                <div class="metric-card">
                    <h4>Prediction Target</h4>
                    <p>final_grade</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.dataframe(data.head(10), use_container_width=True)
    elif page == "Predict Grade":
        prediction_view(data, model_bundle)
    elif page == "Dashboard":
        dashboard_view(data)
    else:
        insights_view(data)


if __name__ == "__main__":
    main()
