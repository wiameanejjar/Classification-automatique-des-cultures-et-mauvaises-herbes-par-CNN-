# ==========================================
# 🌾 APPLICATION STREAMLIT
# Classification automatique des cultures et mauvaises herbes par CNN  - MobileNetV2
# Architecture: Transfer Learning | Dataset: V2 Plant Seedlings
# ==========================================

import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# CONFIGURATION DE LA PAGE
# ==========================================

st.set_page_config(
    page_title="Weed Detection ",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Système de Détection Intelligent des Mauvaises Herbes"
    }
)

# ==========================================
# CSS
# ==========================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    * { font-family: 'Poppins', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-size: 200% 200%;
        animation: gradient 15s ease infinite;
        padding: 2rem; border-radius: 20px;
        text-align: center; margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    @keyframes gradient {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .main-header h1 {
        color: white; font-size: 3rem; font-weight: 700;
        margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .main-header p { color: rgba(255,255,255,0.9); font-size: 1.2rem; margin-top: 0.5rem; }

    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem; border-radius: 15px; color: white;
        margin: 1rem 0; box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .info-card:hover { transform: translateY(-5px); box-shadow: 0 12px 30px rgba(0,0,0,0.25); }
    .info-card h3 { margin: 0 0 0.5rem 0; font-size: 2.5rem; font-weight: 700; }
    .info-card p  { margin: 0; font-size: 1rem; opacity: 0.95; }

    .prediction-box {
        border: 4px solid; padding: 2.5rem; border-radius: 20px;
        text-align: center; margin: 2rem 0;
        box-shadow: 0 10px 40px rgba(0,0,0,0.15);
        animation: fadeIn 0.5s ease-in;
        position: relative; overflow: hidden;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: scale(0.95); }
        to   { opacity: 1; transform: scale(1); }
    }
    .prediction-box::before {
        content: ''; position: absolute;
        top: -50%; left: -50%; width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to   { transform: rotate(360deg); }
    }
    .crop-box {
        border-color: #2E7D32;
        background: linear-gradient(135deg, #C8E6C9 0%, #A5D6A7 100%);
    }
    .weed-box {
        border-color: #C62828;
        background: linear-gradient(135deg, #FFCDD2 0%, #EF9A9A 100%);
    }
    .prediction-content { position: relative; z-index: 1; }
    .prediction-box h1  { font-size: 2.8rem; margin-bottom: 1rem; font-weight: 700; }
    .prediction-box h2  { font-size: 2.2rem; margin: 1rem 0; font-weight: 600; }
    .prediction-box h3  { font-size: 1.8rem; font-weight: 600; }

    .confidence-badge {
        display: inline-block;
        background: rgba(255,255,255,0.9);
        padding: 0.5rem 1.5rem; border-radius: 50px;
        font-size: 1.5rem; font-weight: 700; margin-top: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }

    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; font-size: 1.1rem; font-weight: 600;
        padding: 0.75rem 2rem; border-radius: 50px; border: none;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102,126,234,0.6);
    }

    .success-box {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);
        color: white; padding: 1.5rem; border-radius: 10px;
        margin: 1rem 0; font-weight: 500;
    }
    .warning-box {
        background: linear-gradient(135deg, #f12711 0%, #f5af19 100%);
        color: white; padding: 1.5rem; border-radius: 10px;
        margin: 1rem 0; font-weight: 500;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border-radius: 10px 10px 0 0;
        padding: 0.75rem 1.5rem; font-weight: 600;
    }

    .footer {
        text-align: center; padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border-radius: 15px; margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CONSTANTES
# ==========================================

CLASS_NAMES = [
    "Black-grass", "Charlock", "Cleavers", "Common Chickweed",
    "Common wheat", "Fat Hen", "Loose Silky-bent", "Maize",
    "Scentless Mayweed", "Shepherd's Purse",
    "Small-flowered Cranesbill", "Sugar beet"
]
CROPS = ["Maize", "Sugar beet", "Common wheat"]
WEEDS = [c for c in CLASS_NAMES if c not in CROPS]

MODEL_PATH    = 'mobilenetv2_weed_detection.keras'
METADATA_PATH = 'model_metadata.json'

# ==========================================
# FONCTIONS
# ==========================================

@st.cache_resource
def load_model():
    try:
        return keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f" Erreur chargement modèle : {e}")
        return None

@st.cache_data
def load_metadata():
    try:
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {
            'model_name':        'MobileNetV2 Weed Detection',
            'architecture':      'MobileNetV2 + Transfer Learning',
            'dataset':           'V2 Plant Seedlings',
            'num_classes':       12,
            'accuracy_val':      0.85,
            'accuracy_test':     0.84,
            'error_rate':        0.16,
            'total_params':      2540428,
            'trainable_params':  0,
            'train_samples':     4431,
            'val_samples':       554,
            'test_samples':      554,
            'train_ratio':       0.80,
            'val_ratio':         0.10,
            'test_ratio':        0.10,
            'phase1_epochs':     30,
            'fine_tune_epochs':  15,
            'best_val_accuracy': 0.85,
            'class_names':       CLASS_NAMES,
            'crops':             CROPS,
            'weeds':             WEEDS,
        }

def preprocess_image(image):
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    image = image.convert('RGB').resize((224, 224))
    arr   = preprocess_input(np.array(image, dtype=np.float32))
    return np.expand_dims(arr, 0)

def predict_image(model, image):
    probs     = model.predict(preprocess_image(image), verbose=0)[0]
    idx       = int(np.argmax(probs))
    cls       = CLASS_NAMES[idx]
    is_weed   = cls in WEEDS
    return {
        'class':          cls,
        'category':       "🌿 MAUVAISE HERBE" if is_weed else "🌱 CULTURE",
        'confidence':     float(probs[idx]) * 100,
        'is_weed':        is_weed,
        'probabilities':  {CLASS_NAMES[i]: float(probs[i]) * 100 for i in range(len(CLASS_NAMES))},
    }

# ==========================================
# CHARGEMENT
# ==========================================

model    = load_model()
metadata = load_metadata()
if model is None:
    st.stop()

acc_val  = metadata.get('accuracy_val',  metadata.get('accuracy', 0.85))
acc_test = metadata.get('accuracy_test', acc_val)

# ==========================================
# HEADER
# ==========================================

st.markdown("""
<div class="main-header">
    <h1>🌾 Système de Détection des Mauvaises Herbes</h1>
    <p>Intelligence Artificielle | MobileNetV2 | Transfer Learning | 12 Classes</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=100)
    st.markdown("### 🎯 Navigation")

    page = st.radio("", [
        "🏠 Dashboard",
        "📸 Prédiction",
        "📊 Analyse",
        "⚙️ Modèle",
        "ℹ️ Documentation"
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### 📈 Statistiques")

    c1, c2 = st.columns(2)
    c1.metric("Classes",   metadata['num_classes'])
    c2.metric("Acc. Test", f"{acc_test*100:.1f}%")

    c3, c4 = st.columns(2)
    c3.metric("Train", f"{metadata.get('train_samples', '—')}")
    c4.metric("Val",   f"{metadata.get('val_samples', '—')}")

    st.markdown("---")
    st.markdown("### 🧠 Modèle")
    st.info(f"""
    **Architecture:** MobileNetV2  
    **Type:** Transfer Learning  
    **Paramètres:** {metadata.get('total_params', 0):,}  
    **Dataset:** {metadata.get('dataset', 'V2 Plant Seedlings')}  
    **Split:** 80 / 10 / 10   
    """)

# ==========================================
# PAGE DASHBOARD
# ==========================================

if page == "🏠 Dashboard":

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="info-card"><h3>12</h3><p>Classes Totales</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="info-card"><h3>3</h3><p>Cultures</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="info-card"><h3>9</h3><p>Mauvaises Herbes</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="info-card"><h3>{acc_test*100:.1f}%</h3><p>Accuracy Test</p></div>', unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("##  Objectif du Projet")
        st.markdown(f"""
        Ce système utilise l'**intelligence artificielle** pour identifier automatiquement
        les mauvaises herbes dans les cultures agricoles.

        ###  Avantages
        - **Précision** : Détection de 12 types de plantes — Accuracy Test **{acc_test*100:.1f}%**
        - **Rapidité** : Analyse instantanée en temps réel
        - **Écologie** : Réduction de 30% de l'usage des herbicides
        - **Économie** : Optimisation des coûts de traitement


        """)

    with col2:
        st.markdown("## 📚 Classes")
        with st.expander("🌱 Cultures (3)", expanded=True):
            for c in CROPS:
                st.markdown(f" {c}")
        with st.expander("🌿 Mauvaises Herbes (9)", expanded=False):
            for w in WEEDS:
                st.markdown(f" {w}")

    st.markdown("---")
    st.markdown("## 📊 Visualisations")

    g1, g2 = st.columns(2)

    with g1:
        fig_pie = go.Figure(go.Pie(
            labels=["Cultures (3)", "Mauvaises Herbes (9)"],
            values=[3, 9], hole=0.4,
            marker=dict(colors=['#4CAF50', '#F44336'],
                        line=dict(color='white', width=2)),
            textinfo='percent+label'
        ))
        fig_pie.update_layout(
            title='Répartition Cultures vs Weeds',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with g2:
        fig_split = go.Figure()
        fig_split.add_trace(go.Bar(
            x=["Train (80%)", "Validation (10%)", "Test (10%)"],
            y=[metadata.get('train_samples', 4431),
               metadata.get('val_samples',   554),
               metadata.get('test_samples',  554)],
            marker_color=['#667eea', '#764ba2', '#f093fb'],
            text=[metadata.get('train_samples', 4431),
                  metadata.get('val_samples',   554),
                  metadata.get('test_samples',  554)],
            textposition='outside'
        ))
        fig_split.update_layout(
            title='Répartition du Dataset — Split 80 / 10 / 10',
            yaxis_title="Nombre d'images",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        st.plotly_chart(fig_split, use_container_width=True)

    st.markdown("###  Accuracy : Validation vs Test ")
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Bar(
        x=["Accuracy Validation", "Accuracy Test "],
        y=[acc_val * 100, acc_test * 100],
        marker_color=['#667eea', '#4CAF50'],
        text=[f"{acc_val*100:.2f}%", f"{acc_test*100:.2f}%"],
        textposition='outside',
        width=0.4
    ))
    fig_acc.update_layout(
        yaxis=dict(range=[0, 110], ticksuffix='%'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False, height=320
    )
    st.plotly_chart(fig_acc, use_container_width=True)

# ==========================================
# PAGE PRÉDICTION
# ==========================================

elif page == "📸 Prédiction":

    st.markdown("## 📸 Analysez une Image de Plante")
    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Choisissez une image (PNG, JPG, JPEG)",
            type=['png', 'jpg', 'jpeg'],
            help="Uploadez une photo de plante pour identification"
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Image uploadée", use_column_width=True)

            if st.button("🔍 Analyser l'Image", use_container_width=True):
                with st.spinner(" Analyse en cours..."):
                    result = predict_image(model, image)
                    st.session_state['result']         = result
                    st.session_state['analyzed_image'] = image
                st.success(" Analyse terminée !")

    with col2:
        if 'result' in st.session_state:
            result     = st.session_state['result']
            box_class  = "crop-box" if not result['is_weed'] else "weed-box"
            text_color = "#1B5E20" if not result['is_weed'] else "#B71C1C"

            st.markdown(f"""
            <div class="prediction-box {box_class}">
                <div class="prediction-content">
                    <h1 style="color:{text_color}; margin-bottom:1rem;">{result['category']}</h1>
                    <h2 style="color:#212121; margin:1rem 0;">{result['class']}</h2>
                    <div class="confidence-badge" style="color:{text_color};">
                        Confiance : {result['confidence']:.1f}%
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if result['is_weed']:
                st.markdown("""
                <div class="warning-box">
                    <strong>🚨 Action Recommandée</strong><br>
                    Retirer ou traiter cette mauvaise herbe pour protéger vos cultures
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="success-box">
                    <strong>✔️ Statut</strong><br>
                    Culture saine à protéger et entretenir
                </div>""", unsafe_allow_html=True)

            st.markdown("###  Top 5 Probabilités")
            sorted_probs = sorted(result['probabilities'].items(),
                                  key=lambda x: x[1], reverse=True)[:5]

            for i, (cls, prob) in enumerate(sorted_probs, 1):
                emoji = "🌱" if cls in CROPS else "🌿"
                ca, cb, cc = st.columns([1, 4, 1])
                ca.markdown(f"**{i}.**")
                cb.markdown(f"**{emoji} {cls}**")
                cb.progress(prob / 100)
                cc.metric("", f"{prob:.1f}%", label_visibility="collapsed")

# ==========================================
# PAGE ANALYSE
# ==========================================

elif page == "📊 Analyse":

    st.markdown("## 📑 Analyse Détaillée")

    if 'result' not in st.session_state:
        st.info("📸 Veuillez d'abord analyser une image dans la section **Prédiction**")
    else:
        result = st.session_state['result']

        second = sorted(result['probabilities'].values(), reverse=True)[1]
        margin = result['confidence'] - second

        col1, col2, col3 = st.columns(3)
        col1.metric("Prédiction Principale", f"{result['confidence']:.1f}%")
        col2.metric("2ème Probabilité",      f"{second:.1f}%")
        col3.metric("Marge de Confiance",    f"{margin:.1f}%",
                    delta="Élevée " if margin > 20 else "Faible ⚠️")

        st.markdown("---")
        tab1, tab2 = st.tabs(["📈 Graphique", "📋 Tableau"])

        with tab1:
            df = pd.DataFrame([
                {'Classe': k, 'Probabilité (%)': round(v, 2),
                 'Type': 'Culture' if k in CROPS else 'Weed'}
                for k, v in result['probabilities'].items()
            ]).sort_values('Probabilité (%)', ascending=False)

            fig = px.bar(df, x='Classe', y='Probabilité (%)', color='Type',
                         title='Probabilités — Toutes les Classes',
                         color_discrete_map={'Culture': '#4CAF50', 'Weed': '#F44336'})
            fig.update_layout(xaxis_tickangle=-45,
                               paper_bgcolor='rgba(0,0,0,0)',
                               plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            df_show = df.copy()
            df_show['Classe'] = df_show.apply(
                lambda r: f"🌱 {r['Classe']}" if r['Type'] == 'Culture'
                          else f"🌿 {r['Classe']}", axis=1)
            st.dataframe(
                df_show[['Classe', 'Probabilité (%)', 'Type']]
                    .style.background_gradient(subset=['Probabilité (%)'], cmap='RdYlGn')
                    .format({'Probabilité (%)': '{:.2f}%'}),
                use_container_width=True, hide_index=True
            )

        if 'analyzed_image' in st.session_state:
            with st.expander("🖼️ Voir l'image analysée"):
                st.image(st.session_state['analyzed_image'], width=320,
                         caption=f"{result['class']} — {result['confidence']:.1f}%")

# ==========================================
# PAGE MODÈLE — Architecture + Performances SEULEMENT
# ==========================================

elif page == "⚙️ Modèle":

    st.markdown("## ⚙️ Informations sur le Modèle")

    tab1, tab2 = st.tabs(["🏗️ Architecture", "📈 Performances"])

    # ── Tab Architecture ─────────────────────
    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🧠 Architecture")
            st.info(f"""
            **Nom :** {metadata.get('model_name', 'MobileNetV2 Weed Detection')}  
            **Type :** {metadata.get('architecture', 'MobileNetV2 + Transfer Learning')}  
            **Base :** MobileNetV2 (pré-entraîné ImageNet)  
            **Input Shape :** 224×224×3  
            **Output :** {metadata['num_classes']} classes (Softmax)  
            **Paramètres Totaux :** {metadata.get('total_params', 0):,}  
            **Paramètres Entraînables :** {metadata.get('trainable_params', 0):,}
            """)

        with col2:
            st.markdown("### 📊 Dataset & Split")
            st.success(f"""
            **Dataset :** {metadata.get('dataset', 'V2 Plant Seedlings')}  
            **Split :** 80% Train / 10% Val / 10% Test  
            **Train :** {metadata.get('train_samples', '—')} images  
            **Validation :** {metadata.get('val_samples', '—')} images  
            **Test :** {metadata.get('test_samples', '—')} images  
            **Augmentation :** Rotation, Zoom, Flip, Brightness
            """)

        st.markdown("### 🔧 Architecture Détaillée")
        st.code("""
Input (224×224×3)
        ↓
MobileNetV2 Base — pré-entraîné ImageNet (Phase 1: gelé | Phase 2: fine-tuning)
        ↓
GlobalAveragePooling2D
        ↓
BatchNormalization
        ↓
Dropout (0.5)
        ↓
Dense (256, ReLU)
        ↓
BatchNormalization
        ↓
Dropout (0.4)
        ↓
Dense (128, ReLU)
        ↓
Dropout (0.3)
        ↓
Dense (12, Softmax)  ←  12 classes
        """, language='text')

        st.markdown("###  Optimisations Appliquées")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            -  **Transfer Learning** — Poids pré-entraînés ImageNet
            -  **Data Augmentation** — Rotation, Zoom, Flip, Brightness
            -  **Class Weights** — Équilibrage des classes
            -  **Dropout** — Régularisation (0.3, 0.4, 0.5)
            """)
        with col_b:
            st.markdown("""
            -  **Batch Normalization** — Normalisation des activations
            -  **Early Stopping** — Patience=10
            -  **ReduceLROnPlateau** — Réduction du learning rate
            -  **Test Set Isolé** — Split 80/10/10
            """)

    # ── Tab Performances ─────────────────────
    with tab2:
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Accuracy Validation",
            f"{acc_val*100:.2f}%",
            delta=f"+{(acc_val - 0.5)*100:.1f}% vs baseline"
        )
        col2.metric(
            "Accuracy Test",
            f"{acc_test*100:.2f}%"
        )
        col3.metric(
            "Error Rate",
            f"{metadata.get('error_rate', 1 - acc_test)*100:.2f}%"
        )

        st.markdown("---")

        # Graphique Accuracy Val vs Test
        st.markdown("### 📊 Accuracy : Validation vs Test")
        fig_perf = go.Figure()
        fig_perf.add_trace(go.Bar(
            x=["Validation", "Test"],
            y=[acc_val * 100, acc_test * 100],
            marker_color=['#667eea', '#4CAF50'],
            text=[f"{acc_val*100:.2f}%", f"{acc_test*100:.2f}%"],
            textposition='outside',
            width=0.4
        ))
        fig_perf.update_layout(
            yaxis=dict(range=[0, 110], ticksuffix='%'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False, height=320
        )
        st.plotly_chart(fig_perf, use_container_width=True)

        # Graphique Répartition dataset
        st.markdown("### 📦 Répartition du Dataset")
        fig_split = go.Figure()
        fig_split.add_trace(go.Bar(
            x=["Train (80%)", "Validation (10%)", "Test (10%)"],
            y=[metadata.get('train_samples', 4431),
               metadata.get('val_samples',   554),
               metadata.get('test_samples',  554)],
            marker_color=['#667eea', '#764ba2', '#f093fb'],
            text=[f"{metadata.get('train_samples', 4431):,}",
                  f"{metadata.get('val_samples',   554):,}",
                  f"{metadata.get('test_samples',  554):,}"],
            textposition='outside'
        ))
        fig_split.update_layout(
            yaxis_title="Nombre d'images",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False, height=320
        )
        st.plotly_chart(fig_split, use_container_width=True)

# ==========================================
# PAGE DOCUMENTATION
# ==========================================

elif page == "ℹ️ Documentation":

    st.markdown("## ℹ️ Documentation Complète")
    tabs = st.tabs(["📖 Guide", " Applications", "💡 Conseils"])

    with tabs[0]:
        st.markdown("""
        ### 📖 Guide d'Utilisation

        #### 1️ Préparer votre Image
        - Photo nette de la plante
        - Éclairage naturel recommandé
        - Distance : 20–50 cm
        - Fond simple (terre/sol)

        #### 2️ Analyser
        - Aller dans **📸 Prédiction**
        - Uploader votre image (PNG/JPG)
        - Cliquer sur **Analyser**
        - Résultat en < 2 secondes

        #### 3️ Interpréter
        - 🌱 **Vert** = Culture à protéger
        - 🌿 **Rouge** = Mauvaise herbe à retirer
        - **Confiance** = Niveau de certitude du modèle
        - **Top 5** = Autres possibilités

        #### 4️ Approfondir
        - Page **📊 Analyse** : distribution complète des probabilités
        - Tableau interactif avec toutes les classes
        - Métriques de confiance et marge de décision
        """)

    with tabs[1]:
        st.markdown("""
        ###  Applications Pratiques

        #### 🚜 Agriculture de Précision
        - Traitement ciblé des mauvaises herbes
        - Réduction de 30% des herbicides
        - Cartographie des zones infestées
        - Suivi temporel des invasions

        #### 🤖 Robotique Agricole
        - Robots de désherbage autonomes
        - Drones de surveillance
        - Pulvérisateurs intelligents
        - Systèmes de vision embarqués

        #### 📱 Applications Mobiles
        - Assistant de l'agriculteur
        - Base de données de plantes
        - Historique des traitements
        - Partage avec conseillers

        #### 🌍 Impact Environnemental
        - Réduction de la pollution chimique
        - Préservation de la biodiversité
        - Économie d'eau
        - Agriculture durable
        """)

    with tabs[2]:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ###  Bonnes Pratiques
            - Photo nette et bien cadrée
            - Lumière naturelle du jour
            - Plante au centre de l'image
            - Vue de dessus recommandée
            - Fond simple (terre/sol)
            - Une seule plante principale
            """)
        with col2:
            st.markdown("""
            ### ❌ À Éviter
            - Images floues ou sombres
            - Plusieurs plantes dans le cadre
            - Distance > 1 m
            - Trop près (< 10 cm)
            - Contre-jour
            - Fond complexe
            """)

        st.markdown("""
        ### 🔍 Cas Difficiles
        - Jeunes pousses similaires → Répéter à stade supérieur
        - Plantes entremêlées → Isoler si possible
        - Mauvais éclairage → Flash indirect ou lumière naturelle
        - Confiance < 60% → Prendre 2–3 angles différents
        """)

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")
st.markdown(f"""
<div class="footer">
    <p>
        <strong>Architecture :</strong> MobileNetV2 &nbsp;|&nbsp;
        <strong>Dataset :</strong> V2 Plant Seedlings &nbsp;|&nbsp;
        <strong>Made by :</strong> ANEJJAR Wiame &nbsp;&nbsp;
    </p>
    <p style="margin-top:0.75rem; opacity:0.8;">© 2026 — Agriculture de Précision</p>
</div>
""", unsafe_allow_html=True)
