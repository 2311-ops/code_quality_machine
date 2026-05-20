"""Streamlit UI for Code Quality Assessor."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ui_utils import (
    MODEL_RESULTS,
    OUTPUT_DIR,
    SAMPLE_SNIPPETS,
    analyze_code,
    get_artifact_paths,
    get_image_paths,
    load_outputs,
)

st.set_page_config(
    page_title="Code Quality Assessor",
    page_icon="CQ",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css():
    st.markdown(
        """
        <style>
          .stApp {
            background:
              radial-gradient(circle at top left, rgba(46, 134, 222, 0.18), transparent 32%),
              radial-gradient(circle at bottom right, rgba(26, 188, 156, 0.14), transparent 34%),
              linear-gradient(180deg, #07111f 0%, #0d1728 100%);
            color: #ecf2ff;
          }
          .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1280px;
          }
          h1, h2, h3, h4 {
            color: #f3f7ff !important;
            letter-spacing: -0.02em;
          }
          .hero {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.88), rgba(11, 24, 39, 0.68));
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 28px;
            padding: 28px;
            box-shadow: 0 24px 60px rgba(0, 0, 0, 0.28);
          }
          .hero h1 {
            font-size: 3rem;
            margin-bottom: 0.35rem;
          }
          .hero p {
            font-size: 1.02rem;
            line-height: 1.6;
            color: #c7d2fe;
          }
          .soft-card {
            background: rgba(15, 23, 42, 0.82);
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 22px;
            padding: 20px;
            box-shadow: 0 16px 36px rgba(0, 0, 0, 0.22);
          }
          .pill {
            display: inline-block;
            padding: 0.4rem 0.8rem;
            border-radius: 999px;
            background: rgba(34, 197, 94, 0.12);
            color: #7ef0a7;
            border: 1px solid rgba(34, 197, 94, 0.3);
            font-size: 0.82rem;
            margin-right: 0.4rem;
          }
          .metric-card {
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.96), rgba(10, 18, 32, 0.86));
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 20px;
            padding: 18px 18px 16px;
            min-height: 118px;
          }
          .metric-title {
            color: #94a3b8;
            font-size: 0.82rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
          }
          .metric-value {
            color: #f8fbff;
            font-size: 1.9rem;
            font-weight: 800;
            line-height: 1.1;
          }
          .metric-subtitle {
            color: #cbd5e1;
            margin-top: 0.45rem;
            font-size: 0.95rem;
          }
          .section-label {
            color: #8ab4ff;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
          }
          .result-pill-good {
            background: rgba(16, 185, 129, 0.16);
            color: #86efac;
            border: 1px solid rgba(16, 185, 129, 0.35);
            padding: 0.6rem 1rem;
            border-radius: 999px;
            display: inline-block;
            font-weight: 700;
          }
          .result-pill-medium {
            background: rgba(245, 158, 11, 0.16);
            color: #fbbf24;
            border: 1px solid rgba(245, 158, 11, 0.35);
            padding: 0.6rem 1rem;
            border-radius: 999px;
            display: inline-block;
            font-weight: 700;
          }
          .result-pill-bad {
            background: rgba(239, 68, 68, 0.16);
            color: #fca5a5;
            border: 1px solid rgba(239, 68, 68, 0.35);
            padding: 0.6rem 1rem;
            border-radius: 999px;
            display: inline-block;
            font-weight: 700;
          }
          .sidebar-title {
            font-size: 1.3rem;
            font-weight: 800;
            color: #f8fbff;
            margin-bottom: 0.25rem;
          }
          .small-note {
            color: #94a3b8;
            font-size: 0.92rem;
          }
          .download-box {
            background: rgba(15, 23, 42, 0.78);
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 16px;
            padding: 16px;
          }
          .stButton > button {
            border-radius: 14px;
            border: 1px solid rgba(125, 211, 252, 0.35);
            background: linear-gradient(135deg, #0ea5e9, #14b8a6);
            color: white;
            font-weight: 700;
            padding: 0.6rem 1rem;
          }
          .stDownloadButton > button {
            border-radius: 12px;
            border: 1px solid rgba(148, 163, 184, 0.25);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def cached_outputs():
    return load_outputs(OUTPUT_DIR)


def badge(label: str) -> str:
    cls = {"good": "result-pill-good", "medium": "result-pill-medium", "bad": "result-pill-bad"}.get(label, "result-pill-medium")
    return f'<span class="{cls}">{label.upper()}</span>'


def metric_card(title: str, value: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-title">{title}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard(data: dict):
    st.markdown(
        """
        <div class="hero">
          <div class="section-label">Code Quality Assessor</div>
          <h1>Quality insights for Python code.</h1>
          <p>
            Explore a polished workflow for code-quality analysis: inspect the collected dataset,
            review feature engineering outputs, analyze uploaded Python files, and browse model
            comparisons from the project artifacts.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Labeled files", f"{data['total_files']}", "From labeled_dataset.csv")
    with c2:
        metric_card("Feature count", f"{data['feature_count']}", "Engineered columns")
    with c3:
        best = data["best_model_metrics"]
        metric_card("Best test F1", f"{best['f1']:.2%}", "Gradient Boosting")
    with c4:
        meta = data["meta"]
        metric_card("Saved model", meta.get("model_name", "CodeQualityCNN"), f"Accuracy {meta.get('accuracy', 0):.2%}")

    st.write("")
    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        with st.container(border=True):
            st.subheader("Project workflow")
            workflow = [
                "1. Collect Python files from GitHub repositories.",
                "2. Label the files using Radon metrics and static analysis.",
                "3. Engineer AST, lint, and TF-IDF features.",
                "4. Train and compare classical and deep models.",
                "5. Present results in a clean dashboard UI.",
            ]
            for step in workflow:
                st.markdown(f"- {step}")

    with right:
        with st.container(border=True):
            st.subheader("Training snapshot")
            st.markdown(
                f"""
                <div class="pill">Random Forest 93.56%</div>
                <div class="pill">Gradient Boosting 100.00%</div>
                <div class="pill">1D CNN 95.54%</div>
                <div class="pill">LSTM 44.55%</div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")
            st.info(
                "The notebook saves the CNN artifact as CodeQualityCNN, while Gradient Boosting has the highest test metrics."
            )

    st.write("")
    st.subheader("Dataset and model screenshots")
    images = get_image_paths(OUTPUT_DIR)
    if images:
        cols = st.columns(2, gap="large")
        for i, img in enumerate(images[:4]):
            with cols[i % 2]:
                st.image(str(img), use_container_width=True)
                st.caption(img.name.replace("_", " ").replace(".png", "").title())


def render_analyzer(data: dict):
    st.markdown(
        """
        <div class="hero">
          <div class="section-label">Analyze Code</div>
          <h1>Paste, upload, or pick a sample.</h1>
          <p>
            Run a guided analysis on Python code and get a quality label, confidence, maintainability
            signals, AST structure, and lint feedback in one place.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    col_left, col_right = st.columns([1.05, 0.95], gap="large")
    with col_left:
        source = st.radio("Input mode", ["Upload file", "Paste code", "Try a sample"], horizontal=True)
        code = ""
        file_name = ""

        if source == "Upload file":
            uploaded = st.file_uploader("Upload a .py file", type=["py"])
            if uploaded is not None:
                file_name = uploaded.name
                code = uploaded.read().decode("utf-8", errors="replace")
        elif source == "Paste code":
            code = st.text_area(
                "Paste Python code",
                height=280,
                placeholder="def hello_world():\n    return 'Hello, Code Quality Assessor!'",
            )
        else:
            sample_name = st.selectbox("Choose a sample", list(SAMPLE_SNIPPETS))
            code = SAMPLE_SNIPPETS[sample_name]
            st.code(code, language="python")

        run = st.button("Run analysis", use_container_width=True)

    with col_right:
        with st.container(border=True):
            st.subheader("What this page returns")
            st.markdown(
                """
                - A demo quality label: `good`, `medium`, or `bad`
                - Confidence and quality score
                - Phase 1 metrics from Radon
                - AST structure signals
                - Pyflakes lint signals
                """,
            )
            st.caption("The UI uses transparent static-analysis scoring so the workflow is useful even before an API is wired in.")

    if run:
        if not code.strip():
            st.error("Please upload or paste some Python code first.")
            return

        result = analyze_code(code)
        st.session_state["analysis_result"] = {
            "file_name": file_name or "Pasted code",
            "code": code,
            "result": result,
        }

    payload = st.session_state.get("analysis_result")
    if payload:
        result = payload["result"]
        st.write("")
        header_left, header_right = st.columns([1.05, 0.95], gap="large")
        with header_left:
            with st.container(border=True):
                st.markdown('<div class="section-label">Analysis result</div>', unsafe_allow_html=True)
                st.subheader(payload["file_name"])
                st.markdown(badge(result.label), unsafe_allow_html=True)
                st.write(
                    f"This file was classified as **{result.label.upper()}** with **{result.confidence:.0%}** confidence."
                )
        with header_right:
            with st.container(border=True):
                st.subheader("Score overview")
                st.progress(result.confidence)
                st.metric("Confidence", f"{result.confidence:.0%}")
                st.metric("Demo quality score", f"{result.quality_score:.1f} / 100")

        a, b, c = st.columns(3)
        with a:
            metric_card("Maintainability Index", f"{result.phase1['mi_score']:.2f}", f"avg CC {result.phase1['avg_cc']:.2f}")
        with b:
            metric_card("Lint issues", str(result.lint["lint_total"]), f"errors {result.lint['lint_errors']}, warnings {result.lint['lint_warnings']}")
        with c:
            metric_card("AST depth", str(result.ast["ast_max_depth"]), f"Functions {result.ast['ast_functions']}, Classes {result.ast['ast_classes']}")

        left, right = st.columns([0.95, 1.05], gap="large")
        with left:
            with st.container(border=True):
                st.subheader("Why this result")
                for item in result.rationale:
                    st.markdown(f"- {item}")

        with right:
            with st.container(border=True):
                st.subheader("Metric table")
                metric_df = pd.DataFrame(
                    [
                        ("mi_score", result.phase1["mi_score"]),
                        ("avg_cc", result.phase1["avg_cc"]),
                        ("max_cc", result.phase1["max_cc"]),
                        ("loc", result.phase1["loc"]),
                        ("lloc", result.phase1["lloc"]),
                        ("sloc", result.phase1["sloc"]),
                        ("comments", result.phase1["comments"]),
                        ("blank", result.phase1["blank"]),
                        ("comment_ratio", result.phase1["comment_ratio"]),
                        ("ast_functions", result.ast["ast_functions"]),
                        ("ast_classes", result.ast["ast_classes"]),
                        ("ast_imports", result.ast["ast_imports"]),
                        ("ast_try_blocks", result.ast["ast_try_blocks"]),
                        ("ast_returns", result.ast["ast_returns"]),
                        ("ast_docstring_ratio", result.ast["ast_docstring_ratio"]),
                        ("lint_errors", result.lint["lint_errors"]),
                        ("lint_warnings", result.lint["lint_warnings"]),
                        ("lint_total", result.lint["lint_total"]),
                    ],
                    columns=["Signal", "Value"],
                )
                st.dataframe(metric_df, use_container_width=True, hide_index=True)

        st.write("")
        with st.container(border=True):
            st.subheader("Code preview")
            st.code(payload["code"], language="python")


def render_insights(data: dict):
    st.markdown(
        """
        <div class="hero">
          <div class="section-label">Insights</div>
          <h1>What the project learned.</h1>
          <p>
            Review the class balance, feature importance, lint patterns, and the model comparison
            that informed the final demo workflow.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    images = get_image_paths(OUTPUT_DIR)
    if images:
        for img in images:
            with st.container(border=True):
                st.image(str(img), use_container_width=True)
                st.caption(img.name)
            st.write("")

    st.subheader("Model comparison")
    model_df = pd.DataFrame(
        [
            {"Model": name, "Accuracy": vals["acc"], "Weighted F1": vals["f1"]}
            for name, vals in MODEL_RESULTS.items()
        ]
    )
    st.dataframe(model_df, use_container_width=True, hide_index=True)


def render_artifacts(data: dict):
    st.markdown(
        """
        <div class="hero">
          <div class="section-label">Artifacts</div>
          <h1>Downloadable project files.</h1>
          <p>
            These are the files the UI is built around: datasets, encoders, scalers, vocabulary,
            and the saved CNN model.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    artifact_paths = get_artifact_paths(OUTPUT_DIR)
    if not artifact_paths:
        st.warning("No artifacts found in the output folder.")
        return

    cols = st.columns(2, gap="large")
    for i, path in enumerate(artifact_paths):
        with cols[i % 2]:
            with st.container(border=True):
                st.markdown(f"**{path.name}**")
                st.caption(f"{path.stat().st_size / 1024:.1f} KB")
                st.download_button(
                    label=f"Download {path.name}",
                    data=path.read_bytes(),
                    file_name=path.name,
                    use_container_width=True,
                )
            st.write("")


def render_about():
    st.markdown(
        """
        <div class="hero">
          <div class="section-label">How it works</div>
          <h1>The project workflow in one view.</h1>
          <p>
            Phase 1 collects and labels code. Phase 2 engineers structural and lexical features.
            Phase 3 compares models and saves the final artifacts used by this UI.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    for title, body in [
        ("Phase 1 - Data Collection", "Scrape GitHub Python files, deduplicate them, compute Radon metrics, and assign good / medium / bad labels."),
        ("Phase 2 - Feature Engineering", "Combine maintainability metrics, AST structure, lint signals, and TF-IDF tokens into a 226-column feature matrix."),
        ("Phase 3 - ML Modeling", "Train Random Forest, Gradient Boosting, CNN, and LSTM models, then save the model outputs and metadata."),
    ]:
        with st.container(border=True):
            st.subheader(title)
            st.write(body)
        st.write("")


def main():
    inject_css()
    data = cached_outputs()

    with st.sidebar:
        st.markdown('<div class="sidebar-title">Code Quality Assessor</div>', unsafe_allow_html=True)
        st.caption("A polished UI workflow for the project")
        nav = st.radio(
            "Navigate",
            ["Dashboard", "Analyze Code", "Insights", "Artifacts", "About"],
            label_visibility="collapsed",
        )
        st.write("")
        st.markdown("### Project stats")
        st.metric("Labeled files", data["total_files"])
        st.metric("Features", data["feature_count"])
        st.metric("Best model", data["best_model_name"])
        st.metric("Saved model", data["meta"].get("model_name", "CodeQualityCNN"))

    if nav == "Dashboard":
        render_dashboard(data)
    elif nav == "Analyze Code":
        render_analyzer(data)
    elif nav == "Insights":
        render_insights(data)
    elif nav == "Artifacts":
        render_artifacts(data)
    else:
        render_about()

    st.write("")
    st.caption(
        "Demo note: the live UI uses transparent static-analysis scoring, while the project notebooks contain the saved trained artifacts and evaluation results."
    )


if __name__ == "__main__":
    main()
