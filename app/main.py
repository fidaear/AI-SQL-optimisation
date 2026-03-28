"""
main.py — Streamlit Web UI for AI-Assisted SQL Query Optimization.
"""
import streamlit as st
from core.orchestrator import run_pipeline
from core.ab_tester import run_ab_test

st.set_page_config(page_title="AI SQL Optimizer", page_icon="🗄️", layout="wide")

st.title("🗄️ AI-Assisted SQL Query Optimization")
st.caption("PostgreSQL • Python • AI • Performance Analysis")

sql_input = st.text_area(
    "📥 Enter your SQL SELECT query:",
    height=200,
    placeholder="SELECT * FROM orders o JOIN customers c ON o.customer_id = c.id WHERE o.order_date >= '2025-01-01';"
)

if st.button("🚀 Analyze & Optimize"):
    if not sql_input.strip():
        st.warning("Please enter a SQL query.")
    else:
        with st.spinner("Analyzing your query..."):
            result = run_pipeline(sql_input)

        if "error" in result:
            st.error(result["error"])
        else:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("⚠️ Detected Issues")
                issues = result.get("issues", [])
                if issues:
                    for issue in issues:
                        st.warning(f"**[{issue['severity']}] {issue['rule']}**\n\n{issue['advice']}")
                else:
                    st.success("No issues detected.")

            with col2:
                st.subheader("💡 Index Suggestions")
                suggestions = result.get("index_suggestions", [])
                if suggestions:
                    for s in suggestions:
                        st.code(s, language="sql")
                else:
                    st.info("No index suggestions.")

            st.subheader("🤖 AI Optimization Suggestion")
            st.markdown(result.get("ai_suggestion", "N/A"))

            with st.expander("📊 Execution Plan Details"):
                plan = result.get("plan", {})
                st.json(plan)
