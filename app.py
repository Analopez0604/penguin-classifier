import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import time
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(
    page_title="Penguin Classifier",
    page_icon="🐧",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@300;400;500&display=swap');

*, html, body { box-sizing: border-box; }

.stApp { background: #05070f; color: #dde3f0; }

section[data-testid="stSidebar"] {
    background: #090d1a !important;
    border-right: 1px solid #1a2035 !important;
}

h1,h2,h3,h4 { font-family: 'Syne', sans-serif !important; }

.hero {
    padding: 3rem 0 1.5rem;
    border-bottom: 1px solid #1a2035;
    margin-bottom: 2rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    line-height: 1.05;
    background: linear-gradient(135deg, #60a5fa, #a78bfa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero-sub {
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    color: #4b5a7a;
    margin-top: 0.6rem;
    font-weight: 300;
    letter-spacing: 0.03em;
}
.badge {
    display: inline-block;
    background: #0f1629;
    border: 1px solid #1e2d4a;
    color: #60a5fa;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    padding: 3px 12px;
    border-radius: 999px;
    margin-right: 6px;
    margin-bottom: 1rem;
    letter-spacing: 0.1em;
}

.panel {
    background: #090d1a;
    border: 1px solid #1a2035;
    border-radius: 18px;
    padding: 1.8rem;
    height: 100%;
    position: relative;
    overflow: hidden;
}
.panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 18px 18px 0 0;
}
.panel-tree::before { background: linear-gradient(90deg, #34d399, #059669); }
.panel-knn::before  { background: linear-gradient(90deg, #60a5fa, #3b82f6); }

.panel-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.panel-title-tree { color: #34d399; }
.panel-title-knn  { color: #60a5fa; }

.panel-desc {
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    color: #4b5a7a;
    margin-bottom: 1.4rem;
    line-height: 1.5;
}

.result-box {
    border-radius: 14px;
    padding: 1.6rem 1.4rem;
    text-align: center;
    margin-top: 1rem;
}
.result-box-tree { background: #061a12; border: 1px solid #34d39944; }
.result-box-knn  { background: #06101a; border: 1px solid #60a5fa44; }

.species-name {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    margin: 0.3rem 0;
}
.species-tree { color: #34d399; }
.species-knn  { color: #60a5fa; }

.confidence-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    color: #4b5a7a;
    text-transform: uppercase;
    margin-top: 0.8rem;
}
.confidence-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    font-weight: 600;
    color: #dde3f0;
}

.metric-row {
    display: flex;
    gap: 0.5rem;
    margin-top: 1rem;
    flex-wrap: wrap;
    justify-content: center;
}
.metric-chip {
    background: #0f1629;
    border: 1px solid #1e2d4a;
    border-radius: 8px;
    padding: 0.35rem 0.8rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #64748b;
}
.metric-chip b { color: #dde3f0; }

.comparison-section {
    background: #090d1a;
    border: 1px solid #1a2035;
    border-radius: 18px;
    padding: 2rem;
    margin-top: 2rem;
}
.comparison-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #dde3f0;
    margin-bottom: 1.5rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid #1a2035;
}

.meter-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    color: #64748b;
    margin-bottom: 0.3rem;
}
.meter-wrap {
    background: #0f1629;
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
    margin-bottom: 0.3rem;
    position: relative;
}
.meter-fill-tree {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #34d399, #059669);
    transition: width 0.8s ease;
}
.meter-fill-knn {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #60a5fa, #3b82f6);
    transition: width 0.8s ease;
}
.meter-vals {
    display: flex;
    justify-content: space-between;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #4b5a7a;
    margin-bottom: 1rem;
}

.winner-banner {
    background: linear-gradient(135deg, #0f1a0f, #0a1f1a);
    border: 1px solid #34d39933;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    text-align: center;
    margin-top: 1.5rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: #64748b;
}
.winner-name {
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: #34d399;
    display: block;
    margin-top: 0.3rem;
}

.sidebar-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4b5a7a;
    margin-top: 1.2rem;
    margin-bottom: 0.4rem;
}
.footer {
    text-align: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #1e2d4a;
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid #0f1629;
}

div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #1d4ed8, #3b82f6);
    color: #fff;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    border: none;
    border-radius: 12px;
    padding: 0.85rem 2rem;
    width: 100%;
    letter-spacing: 0.03em;
    transition: all 0.2s ease;
    margin-top: 0.5rem;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 28px #3b82f644;
}

/* Sliders y selects */
.stSlider [data-baseweb="slider"] { padding: 0; }
div[data-baseweb="select"] > div {
    background: #0f1629 !important;
    border-color: #1e2d4a !important;
}
</style>
""", unsafe_allow_html=True)


# ── Modelo cacheado ───────────────────────────────────────────────────────
@st.cache_resource
def entrenar_modelos():
    df = sns.load_dataset('penguins')
    df_clean = df.copy()
    variables_numericas = ['bill_length_mm','bill_depth_mm','flipper_length_mm','body_mass_g']

    for col in variables_numericas:
        df_clean[col] = df_clean.groupby('species')[col].transform(
            lambda x: x.fillna(x.median())
        )
    df_clean['sex'] = df_clean.groupby('species')['sex'].transform(
        lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else x
    )
    for col in ['sex','island']:
        df_clean[col] = df_clean[col].str.strip().str.capitalize()

    df_enc = pd.get_dummies(df_clean, columns=['island','sex'], drop_first=True)
    X = df_enc.drop('species', axis=1)
    y = df_enc['species']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    tree = DecisionTreeClassifier(max_depth=4, random_state=42)
    tree.fit(X_train_sc, y_train)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_sc, y_train)

    def metricas(model, Xt, yt):
        yp = model.predict(Xt)
        return {
            "Accuracy":  round(accuracy_score(yt, yp),  4),
            "Precision": round(precision_score(yt, yp, average='macro'), 4),
            "Recall":    round(recall_score(yt, yp,    average='macro'), 4),
            "F1-Score":  round(f1_score(yt, yp,        average='macro'), 4),
        }

    met_tree = metricas(tree, X_test_sc, y_test)
    met_knn  = metricas(knn,  X_test_sc, y_test)

    return tree, knn, scaler, met_tree, met_knn

tree_model, knn_model, scaler, met_tree, met_knn = entrenar_modelos()

SPECIES_EMOJI = {"Adelie": "🐧", "Chinstrap": "🐧", "Gentoo": "🐧"}
SPECIES_DESC  = {
    "Adelie":    "Pico corto y robusto · Presente en las 3 islas · Especie más común",
    "Chinstrap": "Banda negra bajo el mentón · Exclusiva de la isla Dream",
    "Gentoo":    "La más grande · Aletas largas · Solo en isla Biscoe",
}

def predecir(model, bill_l, bill_d, flipper, mass, island, sex):
    # Construir todas las columnas en el mismo orden que X_train
    cols_all = list(scaler.feature_names_in_) + ['island_Dream','island_Torgersen','sex_Male']
    cols_num = list(scaler.feature_names_in_)

    row = {
        'bill_length_mm':    bill_l,
        'bill_depth_mm':     bill_d,
        'flipper_length_mm': flipper,
        'body_mass_g':       mass,
        'island_Dream':      1 if island == 'Dream'     else 0,
        'island_Torgersen':  1 if island == 'Torgersen' else 0,
        'sex_Male':          1 if sex == 'Male'         else 0,
    }
    entrada = pd.DataFrame([row])
    num_part = pd.DataFrame(scaler.transform(entrada[cols_num]), columns=cols_num)
    for c in cols_num:
        entrada[c] = num_part[c].values
    pred   = model.predict(entrada)[0]
    proba  = model.predict_proba(entrada)[0]
    clases = model.classes_
    return pred, dict(zip(clases, proba))


# ── SIDEBAR ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="badge">ENTRADAS</div>', unsafe_allow_html=True)
    st.markdown("### Datos del pingüino")

    st.markdown('<div class="sidebar-label">Morfología</div>', unsafe_allow_html=True)
    bill_length  = st.slider("Longitud del pico (mm)",  30.0, 65.0,  44.0, 0.1)
    bill_depth   = st.slider("Profundidad del pico (mm)", 13.0, 22.0, 17.0, 0.1)
    flipper      = st.slider("Longitud de aleta (mm)",  170.0, 235.0, 200.0, 1.0)
    body_mass    = st.slider("Masa corporal (g)",       2500.0, 6500.0, 4000.0, 50.0)

    st.markdown('<div class="sidebar-label">Observación</div>', unsafe_allow_html=True)
    island = st.selectbox("Isla", ["Biscoe", "Dream", "Torgersen"])
    sex    = st.selectbox("Sexo", ["Male", "Female"])

    predict_btn = st.button("🔍 Clasificar")

    st.markdown("---")
    st.markdown('<div class="sidebar-label">Métricas del modelo (test)</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;line-height:2;">
    <span style="color:#34d399">▶ Árbol</span> F1 <b style="color:#dde3f0">{met_tree['F1-Score']}</b>
    &nbsp;Acc <b style="color:#dde3f0">{met_tree['Accuracy']}</b><br>
    <span style="color:#60a5fa">▶ KNN&nbsp;&nbsp;</span> F1 <b style="color:#dde3f0">{met_knn['F1-Score']}</b>
    &nbsp;Acc <b style="color:#dde3f0">{met_knn['Accuracy']}</b>
    </div>
    """, unsafe_allow_html=True)


# ── HERO ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div>
    <span class="badge">CRISP-DM</span>
    <span class="badge">MACHINE LEARNING</span>
    <span class="badge">GRUPO 4 · PASCUAL BRAVO</span>
  </div>
  <p class="hero-title">Penguin Species<br>Classifier</p>
  <p class="hero-sub">Comparación en tiempo real: Árbol de Decisión vs K-Nearest Neighbors · Palmer Penguins Dataset</p>
</div>
""", unsafe_allow_html=True)


# ── RESULTADOS ────────────────────────────────────────────────────────────
if predict_btn:
    t0_tree = time.perf_counter()
    pred_tree, proba_tree = predecir(tree_model, bill_length, bill_depth, flipper, body_mass, island, sex)
    t_tree = (time.perf_counter() - t0_tree) * 1000

    t0_knn = time.perf_counter()
    pred_knn,  proba_knn  = predecir(knn_model,  bill_length, bill_depth, flipper, body_mass, island, sex)
    t_knn  = (time.perf_counter() - t0_knn) * 1000

    col_tree, col_knn = st.columns(2, gap="large")

    # ── Panel Árbol ───────────────────────────────────────────────────────
    with col_tree:
        st.markdown(f"""
        <div class="panel panel-tree">
            <div class="panel-title panel-title-tree">🌿 Árbol de Decisión</div>
            <div class="panel-desc">
                Construye reglas de clasificación mediante divisiones sucesivas.<br>
                max_depth = 4 · random_state = 42
            </div>
            <div class="result-box result-box-tree">
                <div style="font-size:3rem">{SPECIES_EMOJI[pred_tree]}</div>
                <div class="species-name species-tree">{pred_tree}</div>
                <div style="font-size:0.8rem;color:#4b5a7a;margin:0.3rem 0 0.8rem">{SPECIES_DESC[pred_tree]}</div>
                <div class="confidence-label">Confianza</div>
                <div class="confidence-value">{proba_tree[pred_tree]*100:.1f}%</div>
                <div class="metric-row">
                    <span class="metric-chip">F1 <b>{met_tree['F1-Score']}</b></span>
                    <span class="metric-chip">Acc <b>{met_tree['Accuracy']}</b></span>
                    <span class="metric-chip">⏱ {t_tree:.2f}ms</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Probabilidades**")
        for sp, prob in sorted(proba_tree.items(), key=lambda x: -x[1]):
            st.markdown(f"""
            <div class="meter-label">{sp}</div>
            <div class="meter-wrap">
                <div class="meter-fill-tree" style="width:{prob*100:.1f}%"></div>
            </div>
            <div class="meter-vals"><span>{prob*100:.1f}%</span></div>
            """, unsafe_allow_html=True)

    # ── Panel KNN ─────────────────────────────────────────────────────────
    with col_knn:
        st.markdown(f"""
        <div class="panel panel-knn">
            <div class="panel-title panel-title-knn">🔵 K-Nearest Neighbors</div>
            <div class="panel-desc">
                Clasifica según los vecinos más cercanos en el espacio de características.<br>
                k = 5 · distancia Euclidiana
            </div>
            <div class="result-box result-box-knn">
                <div style="font-size:3rem">{SPECIES_EMOJI[pred_knn]}</div>
                <div class="species-name species-knn">{pred_knn}</div>
                <div style="font-size:0.8rem;color:#4b5a7a;margin:0.3rem 0 0.8rem">{SPECIES_DESC[pred_knn]}</div>
                <div class="confidence-label">Confianza</div>
                <div class="confidence-value">{proba_knn[pred_knn]*100:.1f}%</div>
                <div class="metric-row">
                    <span class="metric-chip">F1 <b>{met_knn['F1-Score']}</b></span>
                    <span class="metric-chip">Acc <b>{met_knn['Accuracy']}</b></span>
                    <span class="metric-chip">⏱ {t_knn:.2f}ms</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Probabilidades**")
        for sp, prob in sorted(proba_knn.items(), key=lambda x: -x[1]):
            st.markdown(f"""
            <div class="meter-label">{sp}</div>
            <div class="meter-wrap">
                <div class="meter-fill-knn" style="width:{prob*100:.1f}%"></div>
            </div>
            <div class="meter-vals"><span>{prob*100:.1f}%</span></div>
            """, unsafe_allow_html=True)

    # ── Comparativa de métricas ────────────────────────────────────────────
    metricas_nombres = ["Accuracy", "Precision", "Recall", "F1-Score"]
    st.markdown("""
    <div class="comparison-section">
        <div class="comparison-title">📊 Comparativa de métricas — conjunto de prueba</div>
    """, unsafe_allow_html=True)

    for m in metricas_nombres:
        vt = met_tree[m]
        vk = met_knn[m]
        st.markdown(f"""
        <div style="margin-bottom:1.2rem;">
            <div style="display:flex;justify-content:space-between;font-family:'IBM Plex Mono',monospace;
                        font-size:0.75rem;color:#64748b;margin-bottom:0.4rem;">
                <span>{m}</span>
                <span>
                    <span style="color:#34d399">Árbol {vt}</span>
                    &nbsp;·&nbsp;
                    <span style="color:#60a5fa">KNN {vk}</span>
                </span>
            </div>
            <div class="meter-wrap" style="height:10px;">
                <div class="meter-fill-tree" style="width:{vt*100:.1f}%;opacity:0.7;position:absolute;"></div>
            </div>
            <div class="meter-wrap" style="height:10px;margin-top:4px;">
                <div class="meter-fill-knn"  style="width:{vk*100:.1f}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    mejor = "Árbol de Decisión" if met_tree['F1-Score'] >= met_knn['F1-Score'] else "KNN"
    st.markdown(f"""
    </div>
    <div class="winner-banner">
        Mejor modelo según F1-Score macro
        <span class="winner-name">🏆 {mejor}</span>
        <span style="font-size:0.7rem;display:block;margin-top:0.3rem;color:#334155;">
            Nota: el Árbol alcanza métricas perfectas en el conjunto de prueba,
            lo que puede indicar sobreajuste. KNN ofrece un rendimiento más conservador y generalizable.
        </span>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center;padding:2rem 2rem 1rem;color:#1e2d4a;">
        <div style="font-family:'Syne',sans-serif;font-size:1.1rem;color:#2a3a5a;">
            👈 Ajusta los valores en el panel lateral y haz clic en <b style="color:#60a5fa">Clasificar</b>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── SCATTER PLOT INTERACTIVO — siempre visible ────────────────────────────
st.markdown("""
<div style="margin-top:2.5rem;">
    <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1.3rem;
                color:#dde3f0;margin-bottom:0.3rem;">
        📍 Tu pingüino en el dataset real
    </div>
    <div style="font-family:'Inter',sans-serif;font-size:0.82rem;color:#4b5a7a;margin-bottom:1rem;">
        El punto blanco muestra dónde cae tu pingüino respecto a los 344 registros reales.
        Selecciona los ejes que quieras explorar.
    </div>
</div>
""", unsafe_allow_html=True)

df_plot = sns.load_dataset('penguins').dropna()

col_eje1, col_eje2 = st.columns(2)
opciones = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
labels   = {
    'bill_length_mm':    'Longitud del pico (mm)',
    'bill_depth_mm':     'Profundidad del pico (mm)',
    'flipper_length_mm': 'Longitud de aleta (mm)',
    'body_mass_g':       'Masa corporal (g)',
    'species':           'Especie'
}
with col_eje1:
    eje_x = st.selectbox("Eje X", opciones, index=0, key="ex")
with col_eje2:
    eje_y = st.selectbox("Eje Y", opciones, index=2, key="ey")

COLORES = {"Adelie": "#34d399", "Chinstrap": "#f472b6", "Gentoo": "#60a5fa"}

fig = px.scatter(
    df_plot, x=eje_x, y=eje_y, color="species",
    color_discrete_map=COLORES,
    labels=labels,
    hover_data=["species", "island", "sex"],
    opacity=0.7,
    template="plotly_dark",
)

fig.update_traces(marker=dict(size=9, line=dict(width=0)))

# Punto del usuario
val_x = {
    'bill_length_mm':    bill_length,
    'bill_depth_mm':     bill_depth,
    'flipper_length_mm': flipper,
    'body_mass_g':       body_mass,
}[eje_x]

val_y = {
    'bill_length_mm':    bill_length,
    'bill_depth_mm':     bill_depth,
    'flipper_length_mm': flipper,
    'body_mass_g':       body_mass,
}[eje_y]

fig.add_trace(go.Scatter(
    x=[val_x], y=[val_y],
    mode="markers+text",
    marker=dict(size=18, color="white", symbol="star",
                line=dict(color="#fbbf24", width=2)),
    text=["← Tu pingüino"],
    textposition="middle right",
    textfont=dict(color="white", size=12, family="IBM Plex Mono"),
    name="Tu pingüino",
    hovertemplate=f"<b>Tu pingüino</b><br>{labels[eje_x]}: {val_x}<br>{labels[eje_y]}: {val_y}<extra></extra>"
))

fig.update_layout(
    paper_bgcolor="#090d1a",
    plot_bgcolor="#0d1424",
    font=dict(color="#94a3b8", family="Inter"),
    legend=dict(
        bgcolor="#0f1629",
        bordercolor="#1e2d4a",
        borderwidth=1,
        font=dict(size=12)
    ),
    xaxis=dict(gridcolor="#1e2d4a", title_font=dict(color="#64748b")),
    yaxis=dict(gridcolor="#1e2d4a", title_font=dict(color="#64748b")),
    height=460,
    margin=dict(l=20, r=20, t=20, b=20),
)

st.plotly_chart(fig, use_container_width=True)


st.markdown("""
<div class="footer">
    Proyecto Final · Machine Learning · Institución Universitaria Pascual Bravo · Grupo 4 · 2026<br>
    Árbol de Decisión (max_depth=4) · KNN (k=5) · Palmer Penguins via Seaborn · CRISP-DM
</div>
""", unsafe_allow_html=True)
