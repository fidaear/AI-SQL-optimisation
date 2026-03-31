"""
main.py — Streamlit Web UI for AI-Assisted SQL Query Optimization.
"""
import sys, os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ── Charger le .env AVANT tout import du projet ──────────────
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# ── Ajouter la racine du projet au path Python ───────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from core.index_recommender import generate_index_suggestions
from core.rule_engine import analyze

st.set_page_config(page_title="AI SQL Optimizer", page_icon="🗄️", layout="wide")

st.title("🗄️ AI-Assisted SQL Query Optimization")
st.caption("PostgreSQL • Python • AI • Performance Analysis")

# ── Vérification connexion DB ─────────────────────────────────
def test_db_connection():
    try:
        import psycopg2
        conn = psycopg2.connect(
            host     = os.getenv("DB_HOST", "localhost"),
            port     = os.getenv("DB_PORT", "5432"),
            dbname   = os.getenv("DB_NAME", "postgres"),
            user     = os.getenv("DB_USER", "postgres"),
            password = os.getenv("DB_PASSWORD", "")
        )
        conn.close()
        return True, None
    except Exception as e:
        return False, str(e)

# ── Sidebar : statut connexion ────────────────────────────────
with st.sidebar:
    st.subheader("🔌 Connexion PostgreSQL")
    ok, err = test_db_connection()
    if ok:
        st.success("✅ Connecté à PostgreSQL")
        st.caption(f"Host: {os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}")
        st.caption(f"DB: {os.getenv('DB_NAME')} | User: {os.getenv('DB_USER')}")
    else:
        st.error("❌ Non connecté")
        st.caption(err)
        st.info("Vérifie ton fichier .env à la racine du projet.")

    st.divider()
    st.subheader("⚙️ Paramètres")
    run_explain = st.checkbox("Exécuter EXPLAIN ANALYZE", value=ok)

# ── Zone de saisie SQL ────────────────────────────────────────
sql_input = st.text_area(
    "📥 Entre ta requête SQL SELECT :",
    height=180,
    placeholder="""SELECT *
FROM orders o
JOIN customers c ON o.customer_id = c.id
WHERE o.order_date >= '2023-01-01'
  AND c.region = 'EU';"""
)

if st.button("🚀 Analyser & Optimiser", use_container_width=True):
    if not sql_input.strip():
        st.warning("Merci d'entrer une requête SQL.")
    else:
        col1, col2 = st.columns(2)

        # ── Analyse des règles ────────────────────────────────
        with col1:
            st.subheader("⚠️ Problèmes détectés")
            issues = analyze(sql_input)
            if issues:
                for issue in issues:
                    severity_icon = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟡"}.get(issue['severity'], "⚪")
                    st.warning(f"{severity_icon} **[{issue['severity']}] {issue['rule']}**\n\n{issue['advice']}")
            else:
                st.success("✅ Aucun problème détecté.")

        # ── Recommandations d'index ───────────────────────────
        with col2:
            st.subheader("💡 Index recommandés")
            suggestions = generate_index_suggestions(sql_input)
            if suggestions:
                for s in suggestions:
                    st.code(s, language="sql")
                st.info(f"💾 {len(suggestions)} index suggéré(s) pour améliorer les performances.")
            else:
                st.success("✅ Aucun index manquant détecté.")

        # ── EXPLAIN ANALYZE (si connexion OK) ─────────────────
        if run_explain and ok:
            st.subheader("📊 Plan d'exécution (EXPLAIN ANALYZE)")
            try:
                import psycopg2
                conn = psycopg2.connect(
                    host     = os.getenv("DB_HOST", "localhost"),
                    port     = os.getenv("DB_PORT", "5432"),
                    dbname   = os.getenv("DB_NAME", "postgres"),
                    user     = os.getenv("DB_USER", "postgres"),
                    password = os.getenv("DB_PASSWORD", "")
                )
                cur = conn.cursor()
                cur.execute(f"EXPLAIN ANALYZE {sql_input}")
                plan_rows = cur.fetchall()
                conn.close()

                plan_text = "\n".join([row[0] for row in plan_rows])
                st.code(plan_text, language="text")

                # Détection scan type
                if "Seq Scan" in plan_text:
                    st.error("🔴 Sequential Scan détecté — la requête scanne toute la table. Applique les index suggérés !")
                elif "Index Scan" in plan_text:
                    st.success("🟢 Index Scan détecté — la requête utilise déjà un index.")

            except Exception as e:
                st.error(f"Erreur EXPLAIN : {e}")
        elif not ok:
            st.warning("⚠️ EXPLAIN ANALYZE désactivé — pas de connexion PostgreSQL. Vérifie ton .env.")