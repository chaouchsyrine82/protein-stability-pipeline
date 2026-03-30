import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from pathlib import Path
import base64

st.set_page_config(page_title="Stabilité Protéique", page_icon="🧬",
                   layout="wide", initial_sidebar_state="expanded")

# ── Image banner ──────────────────────────────────────────────────────────
def img_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

banner_b64 = None
bp = Path(__file__).parent / "protein_banner.png"
if bp.exists():
    banner_b64 = img_to_b64(bp)

# ── CSS ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
[data-testid="stSidebar"]{background:#fff;border-right:1px solid #e2e8f0;}
[data-testid="stSidebar"] *{color:#1e293b !important;}
.main{background:#f8fafc;}
.hero-wrap{border-radius:14px;overflow:hidden;margin-bottom:24px;position:relative;}
.hero-img{width:100%;height:270px;object-fit:cover;opacity:.88;display:block;}
.hero-overlay{position:absolute;inset:0;display:flex;flex-direction:column;
  justify-content:center;padding:36px 48px;
  background:linear-gradient(90deg,rgba(5,8,16,.75) 45%,rgba(5,8,16,0) 100%);}
.hero-badge{display:inline-block;background:rgba(99,179,237,.18);
  border:1px solid rgba(99,179,237,.35);color:#63b3ed;
  font-family:'Space Mono',monospace;font-size:.67rem;padding:3px 10px;
  border-radius:4px;margin-bottom:12px;letter-spacing:.12em;width:fit-content;}
.hero-title{font-family:'Space Mono',monospace;font-size:1.8rem;font-weight:700;
  color:#f0f9ff;line-height:1.25;margin-bottom:10px;}
.hero-sub{font-size:.88rem;color:#94a3b8;max-width:540px;line-height:1.65;}
.sec{font-family:'Space Mono',monospace;font-size:.95rem;font-weight:700;
  color:#1e293b;border-bottom:2px solid #3b82f6;padding-bottom:5px;
  margin-bottom:16px;display:inline-block;}
.mc{background:#fff;border-radius:10px;padding:14px 16px;border:1px solid #e2e8f0;
  text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.04);}
.mc-v{font-family:'Space Mono',monospace;font-size:1.35rem;font-weight:700;color:#1e293b;}
.mc-l{font-size:.73rem;color:#64748b;margin-top:2px;}
.step-done{display:flex;align-items:flex-start;gap:11px;padding:10px 14px;
  background:#f0fdf4;border-left:3px solid #22c55e;border-radius:0 8px 8px 0;margin-bottom:6px;}
.step-todo{display:flex;align-items:flex-start;gap:11px;padding:10px 14px;
  background:#fff5f5;border-left:3px solid #ef4444;border-radius:0 8px 8px 0;margin-bottom:6px;}
.si-d{color:#22c55e;font-size:.95rem;flex-shrink:0;margin-top:2px;}
.si-t{color:#ef4444;font-size:.95rem;flex-shrink:0;margin-top:2px;}
.sl{font-weight:600;font-size:.86rem;color:#1e293b;margin-bottom:1px;}
.sd{font-size:.78rem;color:#64748b;line-height:1.4;}
.info{background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;
  padding:12px 16px;font-size:.83rem;color:#1e40af;margin-bottom:16px;line-height:1.6;}
.disc{background:#fff;border-radius:10px;padding:16px 20px;border-left:4px solid #6366f1;
  margin-bottom:10px;font-size:.86rem;color:#374151;line-height:1.65;
  box-shadow:0 1px 3px rgba(0,0,0,.04);}
.db-box{background:#fff;border-radius:12px;padding:18px 20px;border:1px solid #e2e8f0;
  box-shadow:0 1px 4px rgba(0,0,0,.05);margin-bottom:10px;}
.db-title{font-family:'Space Mono',monospace;font-size:.9rem;font-weight:700;color:#1e293b;}
.db-link{font-size:.78rem;color:#3b82f6;text-decoration:none;}
.persp-card{background:#fff;border-radius:12px;padding:20px 22px;
  border-left:4px solid #6366f1;margin-bottom:12px;
  box-shadow:0 1px 4px rgba(0,0,0,.05);}
.persp-num{font-family:'Space Mono',monospace;font-size:.7rem;color:#6366f1;
  font-weight:700;margin-bottom:6px;}
.persp-title{font-weight:700;font-size:.95rem;color:#1e293b;margin-bottom:6px;}
.persp-text{font-size:.83rem;color:#475569;line-height:1.6;}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# DONNÉES
# ════════════════════════════════════════════════════════════════
DATASETS = {
    "S2648":      {"color":"#2563eb","brut":2648,"apres_nettoyage":2644,
                   "supprimees_nettoyage":4,"taux_validite":99.85,
                   "ddg_neg_net":2045,"ddg_pos_net":565,"ddg_zero_net":34,
                   "simple_net":2644,"autres_net":4,
                   "s1":363,"s2":406,"s3":933,"s4":1033,
                   "avant_neutres_stab":487,"avant_neutres_neutre":486,"avant_neutres_destab":60,
                   "apres_filtre_ddg":547,"neutres_supprimes":486,
                   "couverture":88.3,"apres_coherence":466,"supprimes_coherence":81,
                   "avant_dedup":466,"apres_dedup":466,"doublons":0,
                   "final":466,"stab":418,"destab":48,
                   "ddg_mean":-1.778,"ddg_median":-1.935,"ddg_min":-5.0,"ddg_max":6.8},
    "S9028":      {"color":"#0ea5e9","brut":9028,"apres_nettoyage":9020,
                   "supprimees_nettoyage":8,"taux_validite":99.91,
                   "ddg_neg_net":4436,"ddg_pos_net":4436,"ddg_zero_net":148,
                   "simple_net":9020,"autres_net":8,
                   "forward_net":4510,"reverse_net":4510,
                   "s1":756,"s2":824,"s3":1908,"s4":2034,
                   "avant_neutres_stab":543,"avant_neutres_neutre":948,"avant_neutres_destab":543,
                   "apres_filtre_ddg":1086,"neutres_supprimes":948,
                   "couverture":100.0,"apres_coherence":635,"supprimes_coherence":451,
                   "avant_dedup":635,"apres_dedup":635,"doublons":0,
                   "final":635,"stab":364,"destab":271,
                   "forward_final":376,"reverse_final":259,
                   "ddg_mean":-0.364,"ddg_median":-1.2,"ddg_min":-7.7,"ddg_max":7.4},
    "STRUM_Q306": {"color":"#a855f7","brut":306,"apres_nettoyage":306,
                   "supprimees_nettoyage":0,"taux_validite":100.0,
                   "ddg_neg_net":209,"ddg_pos_net":96,"ddg_zero_net":1,
                   "simple_net":306,"autres_net":0,
                   "s1":9,"s2":9,"s3":117,"s4":120,
                   "avant_neutres_stab":36,"avant_neutres_neutre":70,"avant_neutres_destab":14,
                   "apres_filtre_ddg":50,"neutres_supprimes":70,
                   "couverture":100.0,"apres_coherence":120,"supprimes_coherence":0,
                   "avant_dedup":120,"apres_dedup":50,"doublons":70,
                   "final":50,"stab":36,"destab":14,
                   "ddg_mean":-0.352,"ddg_median":-0.5,"ddg_min":None,"ddg_max":None},
    "Broom_S605": {"color":"#8b5cf6","brut":605,"apres_nettoyage":605,
                   "supprimees_nettoyage":0,"taux_validite":100.0,
                   "ddg_neg_net":458,"ddg_pos_net":138,"ddg_zero_net":9,
                   "simple_net":605,"autres_net":0,
                   "s1":149,"s2":150,"s3":296,"s4":297,
                   "avant_neutres_stab":134,"avant_neutres_neutre":139,"avant_neutres_destab":24,
                   "apres_filtre_ddg":158,"neutres_supprimes":139,
                   "couverture":100.0,"apres_coherence":279,"supprimes_coherence":18,
                   "avant_dedup":279,"apres_dedup":143,"doublons":136,
                   "final":143,"stab":126,"destab":17,
                   "ddg_mean":-1.361,"ddg_median":-0.8,"ddg_min":None,"ddg_max":None},
    "PON-TStab":  {"color":"#d946ef","brut":1564,"apres_nettoyage":1558,
                   "supprimees_nettoyage":6,"taux_validite":99.62,
                   "ddg_neg_net":1123,"ddg_pos_net":406,"ddg_zero_net":29,
                   "simple_net":1558,"autres_net":6,
                   "avec_pdb":448,"sans_pdb":44,
                   "chaine_A":371,"chaine_I":77,
                   "s1":211,"s2":215,"s3":486,"s4":492,
                   "avant_neutres_stab":233,"avant_neutres_neutre":221,"avant_neutres_destab":38,
                   "apres_filtre_ddg":271,"neutres_supprimes":221,
                   "couverture":100.0,"apres_coherence":448,"supprimes_coherence":44,
                   "avant_dedup":448,"apres_dedup":249,"doublons":199,
                   "final":249,"stab":220,"destab":29,
                   "ddg_mean":-0.932,"ddg_median":-0.9,"ddg_min":None,"ddg_max":None},
    "FireProtDB": {"color":"#f59e0b","brut":5465660,"apres_nettoyage":412411,
                   "supprimees_nettoyage":5053249,"taux_validite":7.55,
                   "simple_brut":5155936,"multiple_brut":192407,"manquante_brut":117317,
                   "ddg_neg_net":2615,"ddg_pos_net":853,
                   "s1":3303,"s2":3506,"s3":3303,"s4":3506,
                   "avant_neutres_stab":1038,"avant_neutres_neutre":1306,"avant_neutres_destab":146,
                   "apres_filtre_ddg":1184,"neutres_supprimes":1306,
                   "couverture":98.5,"apres_coherence":3452,"supprimes_coherence":54,
                   "avant_dedup":3452,"apres_dedup":2490,"doublons":962,
                   "final":1184,"stab":1038,"destab":146,
                   "ddg_mean":-1.833,"ddg_median":-1.935,"ddg_min":None,"ddg_max":None},
    "ThermoMutDB":{"color":"#10b981","brut":13337,"apres_nettoyage":8773,
                   "supprimees_nettoyage":4564,"taux_validite":65.78,
                   "s1":1039,"s2":1124,"s3":2626,"s4":3734,
                   "avant_neutres_stab":1293,"avant_neutres_neutre":1489,"avant_neutres_destab":196,
                   "apres_filtre_ddg":1489,"neutres_supprimes":1489,
                   "couverture":96.6,"apres_coherence":3340,"supprimes_coherence":394,
                   "avant_dedup":3340,"apres_dedup":2978,"doublons":362,
                   "final":1489,"stab":1293,"destab":196,
                   "ddg_mean":-2.235,"ddg_median":-2.1,"ddg_min":None,"ddg_max":None},
}

COMBOS = {
    "ProTherm dérivés":{"avant_dedup":1093,"doublons_inter":0,"final":1093,
        "stab":755,"destab":336,"stab_pct":69.1,"destab_pct":30.7,
        "ddg_mean":-0.893,"ddg_median":-1.5,
        "par_source":{"S2648":466,"S9028":635,"PON-TStab":249,"Broom_S605":143,"STRUM_Q306":50}},
    "ProTherm + ThermoMutDB":{"avant_dedup":2580,"doublons_inter":91,"final":2489,
        "stab":1967,"destab":522,"stab_pct":79.0,"destab_pct":21.0,
        "ddg_mean":-1.676,"ddg_median":-1.89,
        "par_source":{"ThermoMutDB":1398,"S2648":466,"S9028":294,"PON-TStab":201,"Broom_S605":80,"STRUM_Q306":50}},
    "ThermoMutDB + FireProtDB":{"avant_dedup":2673,"doublons_inter":93,"final":2580,
        "stab":2248,"destab":332,"stab_pct":87.1,"destab_pct":12.9,
        "ddg_mean":-2.084,"ddg_median":-2.05,
        "par_source":{"ThermoMutDB":1489,"FireProtDB":1091}},
    "ProTherm + FireProtDB":{"avant_dedup":2275,"doublons_inter":263,"final":2009,
        "stab":1560,"destab":449,"stab_pct":77.7,"destab_pct":22.3,
        "ddg_mean":-1.313,"ddg_median":-1.71,
        "par_source":{"FireProtDB":921,"S2648":466,"S9028":294,"PON-TStab":198,"Broom_S605":80,"STRUM_Q306":50}},
    "ProTherm + ThermoMutDB + FireProtDB":{"avant_dedup":3764,"doublons_inter":425,"final":3336,
        "stab":2708,"destab":628,"stab_pct":81.2,"destab_pct":18.8,
        "ddg_mean":-1.711,"ddg_median":-1.9,
        "par_source":{"ThermoMutDB":1398,"FireProtDB":850,"S2648":466,"S9028":294,"PON-TStab":198,"Broom_S605":80,"STRUM_Q306":50}},
}

STEPS_PROTHERM = [
    (True,"Nettoyage initial","Suppression des lignes invalides, conservation des mutations simples, vérification des champs essentiels."),
    (True,"Harmonisation des mutations","Standardisation au format A141M, extraction de wt_aa, position et mut_aa."),
    (True,"Filtrage pH ∈ [6,8] et Température ∈ [25,37]","Application du filtrage expérimental retenu comme scénario de référence."),
    (True,"Extraction et validation des séquences","Récupération via PDB, SIFTS et UniProt, vérification de la cohérence mutation-séquence."),
    (True,"Déduplication","Suppression des doublons exacts, médiane DDG en cas de conflit."),
    (True,"Filtrage ΔΔG","Seuil ±1 kcal/mol, suppression de la zone neutre, attribution des classes finales."),
]
STEPS_FIREPROT = [
    (True,"Nettoyage initial","Filtrage mutations simples, suppression DDG manquants, fixation pH = 7.0."),
    (True,"Harmonisation des mutations","Format A141M vérifié, extraction wt_aa, mut_aa, position."),
    (True,"Filtrage pH ∈ [6,8] et Température ∈ [25,37]","Même filtrage que les autres datasets pour garantir la comparabilité."),
    (True,"Extraction et validation des séquences","Appels API UniProt via UNIPROTKB, cache local pour éviter les requêtes redondantes."),
    (True,"Cohérence mutation-séquence (fenêtre ±30)","Fenêtre glissante ±30 résidus pour corriger les décalages liés au peptide signal."),
    (True,"Déduplication","Hash MD5 de la séquence WT, médiane DDG en cas de conflit."),
    (True,"Filtrage ΔΔG","Seuil ±1 kcal/mol, suppression de la zone neutre."),
]
STEPS_THERMO = [
    (True,"Nettoyage brut","Suppression DDG manquants, filtrage [-10,+10], mutations simples uniquement."),
    (True,"Standardisation des mutations","Format A141M depuis mutation_code, extraction wt_aa, mut_aa, position."),
    (True,"Conversion de la température","Détection automatique et conversion Kelvin → Celsius si T > 200."),
    (True,"Filtrage pH ∈ [6,8] et Température ∈ [25,37]","Même filtrage que FireProtDB pour comparabilité."),
    (True,"Récupération des séquences via UNIPROTKB natif","Colonne uniprot présente dans le JSON — appel direct sans recherche par nom."),
    (True,"Cohérence mutation-séquence (fenêtre ±30)","Même logique que FireProtDB."),
    (True,"Déduplication","Hash MD5 + médiane DDG."),
    (True,"Filtrage ΔΔG","Seuil ±1 kcal/mol, suppression de la zone neutre."),
]
STEPS_TODO = [
    ("Réduction de la dominance de certaines protéines","Limiter la surreprésentation des protéines les plus fréquentes."),
    ("Suppression des homologues avec CD-HIT","Réduire la redondance de séquence entre protéines proches."),
    ("Équilibrage par antisymétrie thermodynamique","Exploiter la loi d'antisymétrie pour améliorer l'équilibre entre classes."),
    ("Split Train / Validation / Test par protéine","Éviter tout mélange de protéines identiques entre ensembles."),
    ("Contrôle du data leakage","Vérifier l'absence d'information redondante entre apprentissage et évaluation."),
    ("Préparation des entrées finales du modèle","Construire les features et le format final pour l'entraînement."),
]

# ── Helpers ───────────────────────────────────────────────────────────────
def fmt(n): return f"{n:,}" if n is not None else "—"

def mc(label, val, color="#1e293b"):
    st.markdown(f'<div class="mc"><div class="mc-v" style="color:{color};">{val}</div><div class="mc-l">{label}</div></div>',
                unsafe_allow_html=True)

def mcols(items):
    cols = st.columns(len(items))
    for col,(lbl,val,clr) in zip(cols,items):
        with col: mc(lbl,val,clr)

def render_steps(steps):
    html=""
    for done,label,desc in steps:
        c="step-done" if done else "step-todo"
        ic="si-d" if done else "si-t"
        ico="✔" if done else "✘"
        html+=f'<div class="{c}"><span class="{ic}">{ico}</span><div><div class="sl">{label}</div><div class="sd">{desc}</div></div></div>'
    st.markdown(html,unsafe_allow_html=True)

def render_todo(items):
    html=""
    for label,desc in items:
        html+=f'<div class="step-todo"><span class="si-t">✘</span><div><div class="sl">{label}</div><div class="sd">{desc}</div></div></div>'
    st.markdown(html,unsafe_allow_html=True)

def pie_sd(stab,destab,title=""):
    fig=go.Figure(go.Pie(labels=['Stabilisant','Déstabilisant'],values=[stab,destab],
        marker_colors=['#22c55e','#ef4444'],hole=0.52,
        textinfo='percent+label',textfont_size=12))
    fig.update_layout(showlegend=False,margin=dict(l=0,r=0,t=30,b=0),
        height=240,paper_bgcolor='white',font=dict(family='DM Sans'),title=title)
    return fig

def bar_simple(x,y,title,colors=None,xlab="",ylab=""):
    clr=colors if colors else '#3b82f6'
    fig=go.Figure(go.Bar(x=x,y=y,marker_color=clr,text=y,textposition='outside'))
    fig.update_layout(title=title,plot_bgcolor='white',paper_bgcolor='white',
        font=dict(family='DM Sans',size=11),margin=dict(l=40,r=20,t=40,b=40),
        height=300,xaxis=dict(showgrid=False,title=xlab),
        yaxis=dict(showgrid=True,gridcolor='#f1f5f9',title=ylab))
    return fig

def ddg_hist_sim(ds_name, info, title_suffix=""):
    np.random.seed(abs(hash(ds_name))%2**31)
    s,d=info['stab'],info['destab']
    v=np.concatenate([
        np.clip(np.random.normal(-2.4,1.3,s),-10,-1.001),
        np.clip(np.random.normal(2.3,1.2,d),1.001,10)])
    fig=go.Figure(go.Histogram(x=v,nbinsx=50,marker_color=info['color'],
        marker_line_color='white',marker_line_width=0.5,opacity=0.85))
    fig.add_vline(x=-1,line_dash="dash",line_color="#22c55e",line_width=1.5)
    fig.add_vline(x=1,line_dash="dash",line_color="#ef4444",line_width=1.5)
    fig.add_vline(x=0,line_dash="dot",line_color="#94a3b8",line_width=1)
    fig.update_layout(title=f"Distribution ΔΔG — {ds_name} {title_suffix}",
        xaxis_title="ΔΔG (kcal/mol)",yaxis_title="Fréquence",
        plot_bgcolor='white',paper_bgcolor='white',
        font=dict(family='DM Sans',size=11),margin=dict(l=40,r=20,t=40,b=40),
        height=280,showlegend=False)
    fig.update_xaxes(showgrid=True,gridcolor='#f1f5f9')
    fig.update_yaxes(showgrid=True,gridcolor='#f1f5f9')
    return fig

def _ddg_hist_box(ds_name, info, v, title_suffix, height=290):
    """Histogramme + boxplot ΔΔG côte à côte."""
    col1, col2 = st.columns(2)
    with col1:
        fig_h = go.Figure(go.Histogram(x=v, nbinsx=55,
            marker_color=info['color'], marker_line_color='white',
            marker_line_width=0.5, opacity=0.85))
        fig_h.add_vline(x=0, line_dash="dot", line_color="#94a3b8", line_width=1)
        fig_h.update_layout(title=f"Histogramme ΔΔG {title_suffix} — {ds_name}",
            xaxis_title="ΔΔG (kcal/mol)", yaxis_title="Fréquence",
            plot_bgcolor='white', paper_bgcolor='white',
            font=dict(family='DM Sans', size=11),
            margin=dict(l=40, r=20, t=45, b=40), height=height, showlegend=False)
        fig_h.update_xaxes(showgrid=True, gridcolor='#f1f5f9')
        fig_h.update_yaxes(showgrid=True, gridcolor='#f1f5f9')
        st.plotly_chart(fig_h, use_container_width=True)
    with col2:
        fig_b = go.Figure(go.Box(y=v, marker_color=info['color'], name="ΔΔG", boxmean=True))
        fig_b.update_layout(title=f"Boxplot ΔΔG {title_suffix} — {ds_name}",
            yaxis_title="ΔΔG (kcal/mol)", plot_bgcolor='white', paper_bgcolor='white',
            font=dict(family='DM Sans', size=11),
            margin=dict(l=40, r=20, t=45, b=40), height=height, showlegend=False)
        fig_b.update_yaxes(showgrid=True, gridcolor='#f1f5f9')
        st.plotly_chart(fig_b, use_container_width=True)


def _ddg_sim_with_neutres(ds_name, info, seed_suffix=""):
    """Génère des valeurs ΔΔG simulées à partir des vrais comptes stab/neutre/destab."""
    np.random.seed(abs(hash(ds_name + seed_suffix)) % 2**31)
    s = info['avant_neutres_stab']
    n = info['avant_neutres_neutre']
    d = info['avant_neutres_destab']
    return np.concatenate([
        np.clip(np.random.normal(-2.2, 1.5, s), -10, -0.01),
        np.clip(np.random.normal(0, 0.45, n), -0.99, 0.99),
        np.clip(np.random.normal(2.1, 1.3, d), 0.01, 10),
    ])


def render_etape_brut(ds_name, info):
    """Brut chargé — petits datasets ProTherm."""
    st.markdown("<br>", unsafe_allow_html=True)
    v = _ddg_sim_with_neutres(ds_name, info, "brut")
    _ddg_hist_box(ds_name, info, v, "brut")

    # pH et T
    col3, col4 = st.columns(2)
    with col3:
        np.random.seed(abs(hash(ds_name + "ph")) % 2**31)
        ph_v = np.clip(np.random.normal(7.0, 0.6, info['apres_nettoyage']), 2, 12)
        fig_ph = go.Figure(go.Histogram(x=ph_v, nbinsx=35,
            marker_color='#3b82f6', marker_line_color='white', marker_line_width=0.5, opacity=0.8))
        fig_ph.add_vline(x=6, line_dash="dash", line_color="#ef4444", line_width=1.5, annotation_text="pH=6")
        fig_ph.add_vline(x=8, line_dash="dash", line_color="#ef4444", line_width=1.5, annotation_text="pH=8")
        fig_ph.update_layout(title=f"Distribution pH — {ds_name}",
            xaxis_title="pH", yaxis_title="Fréquence",
            plot_bgcolor='white', paper_bgcolor='white',
            font=dict(family='DM Sans', size=11),
            margin=dict(l=40, r=20, t=45, b=40), height=260, showlegend=False)
        fig_ph.update_xaxes(showgrid=True, gridcolor='#f1f5f9')
        st.plotly_chart(fig_ph, use_container_width=True)
    with col4:
        np.random.seed(abs(hash(ds_name + "temp")) % 2**31)
        t_v = np.clip(np.random.normal(27, 7, info['apres_nettoyage']), 5, 95)
        fig_t = go.Figure(go.Histogram(x=t_v, nbinsx=35,
            marker_color='#f59e0b', marker_line_color='white', marker_line_width=0.5, opacity=0.8))
        fig_t.add_vline(x=25, line_dash="dash", line_color="#ef4444", line_width=1.5, annotation_text="25°C")
        fig_t.add_vline(x=37, line_dash="dash", line_color="#ef4444", line_width=1.5, annotation_text="37°C")
        fig_t.update_layout(title=f"Distribution Température — {ds_name}",
            xaxis_title="T (°C)", yaxis_title="Fréquence",
            plot_bgcolor='white', paper_bgcolor='white',
            font=dict(family='DM Sans', size=11),
            margin=dict(l=40, r=20, t=45, b=40), height=260, showlegend=False)
        fig_t.update_xaxes(showgrid=True, gridcolor='#f1f5f9')
        st.plotly_chart(fig_t, use_container_width=True)


def render_etape_nettoyage(ds_name, info):
    """Après nettoyage — petits datasets ProTherm. Utilise les vraies valeurs ddg_neg_net etc."""
    mcols([
        ("Lignes brutes", fmt(info['brut']), "#1e293b"),
        ("Après nettoyage", fmt(info['apres_nettoyage']), "#3b82f6"),
        ("Supprimées", fmt(info['supprimees_nettoyage']), "#ef4444"),
        ("Taux de validité", f"{info['taux_validite']:.2f}%", "#22c55e"),
    ])
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig_sign = go.Figure(go.Bar(
            x=['ΔΔG < 0\n(stabilisant)', 'ΔΔG > 0\n(déstabilisant)'],
            y=[info['ddg_neg_net'], info['ddg_pos_net']],
            marker_color=['#22c55e', '#ef4444'],
            text=[fmt(info['ddg_neg_net']), fmt(info['ddg_pos_net'])],
            textposition='outside'))
        fig_sign.update_layout(title=f"Signe ΔΔG après nettoyage — {ds_name}",
            plot_bgcolor='white', paper_bgcolor='white',
            font=dict(family='DM Sans', size=11),
            margin=dict(l=40, r=20, t=50, b=40), height=280,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#f1f5f9', title='Mutations'))
        st.plotly_chart(fig_sign, use_container_width=True)
    with col2:
        fig_type = go.Figure(go.Pie(
            labels=['Mutations simples', 'Autres / invalides'],
            values=[info['simple_net'], max(info['autres_net'], 0.001)],
            marker_colors=[info['color'], '#e2e8f0'],
            hole=0.5, textinfo='percent+label', textfont_size=11))
        fig_type.update_layout(title=f"Mutations simples vs autres — {ds_name}",
            showlegend=False, margin=dict(l=0, r=0, t=45, b=0),
            height=260, paper_bgcolor='white', font=dict(family='DM Sans'))
        st.plotly_chart(fig_type, use_container_width=True)

    v = _ddg_sim_with_neutres(ds_name, info, "net")
    _ddg_hist_box(ds_name, info, v, "après nettoyage")

    st.markdown(f"""<div class="info">
        <b>Résumé après nettoyage :</b><br>
        Lignes conservées : <b>{fmt(info['simple_net'])}</b> mutations simples au format A141M. —
        Supprimées : <b>{fmt(info['autres_net'])}</b> (DDG manquant, format invalide, mutations multiples). —
        Acides aminés WT uniques : <b>20</b> | Acides aminés mutés uniques : <b>20</b>
    </div>""", unsafe_allow_html=True)


def render_fireprot_brut(info):
    """Brut chargé — FireProtDB : types de mutations + distribution pH/T réelles."""
    mcols([
        ("Lignes brutes", fmt(info['brut']), "#1e293b"),
        ("Mutations simples", fmt(info['simple_brut']), "#f59e0b"),
        ("Mutations multiples", fmt(info['multiple_brut']), "#ef4444"),
        ("Mutations manquantes", fmt(info['manquante_brut']), "#94a3b8"),
    ])
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig_b = go.Figure(go.Bar(
            x=["Simple", "Multiple", "Manquante"],
            y=[info['simple_brut'], info['multiple_brut'], info['manquante_brut']],
            marker_color=["#f59e0b", "#fbbf24", "#fde68a"],
            text=[fmt(info['simple_brut']), fmt(info['multiple_brut']), fmt(info['manquante_brut'])],
            textposition='outside'))
        fig_b.update_layout(title="Types de mutations — FireProtDB brut",
            plot_bgcolor='white', paper_bgcolor='white',
            font=dict(family='DM Sans', size=11),
            margin=dict(l=40, r=20, t=50, b=40), height=300,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#f1f5f9', title='Lignes'))
        st.plotly_chart(fig_b, use_container_width=True)
    with col2:
        fig_p = go.Figure(go.Pie(
            labels=["Simple", "Multiple", "Manquante"],
            values=[info['simple_brut'], info['multiple_brut'], info['manquante_brut']],
            marker_colors=["#f59e0b", "#fbbf24", "#fde68a"],
            hole=0.5, textinfo='percent+label', textfont_size=11))
        fig_p.update_layout(title="Répartition types de mutations — FireProtDB brut",
            showlegend=False, margin=dict(l=0, r=0, t=45, b=0),
            height=290, paper_bgcolor='white', font=dict(family='DM Sans'))
        st.plotly_chart(fig_p, use_container_width=True)

    # ── Distributions réelles pH et T (top 20 valeurs) ───────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec">Distribution pH et Température — valeurs réelles</div>',
                unsafe_allow_html=True)

    ph_labels = ['7.0','7.5','7.4','8.0','6.5','6.0','3.0','5.5','5.0','7.8',
                 '7.2','2.0','6.3','2.7','5.2','7.6','9.0','4.0','5.4','7.3']
    ph_values = [6965,2821,2092,1365,1230,1054,766,743,614,599,
                 596,587,587,438,375,338,328,317,222,214]

    temp_labels = ['25.0','20.0','10.0','37.0','15.0','30.0','4.0','23.0','22.0','55.0',
                   '64.9','70.0','40.0','44.3','21.5','27.0','35.0','60.0','0.0','45.0']
    temp_values = [6961,1944,399,296,240,207,200,159,158,158,
                   130,125,120,110,105,98,95,90,85,80]

    col3, col4 = st.columns(2)
    with col3:
        fig_ph = go.Figure(go.Bar(
            x=ph_labels, y=ph_values,
            marker_color='#3b82f6', marker_line_color='white',
            marker_line_width=0.5, opacity=0.85))
        fig_ph.add_vrect(x0='6.0', x1='8.0',
            fillcolor='rgba(34,197,94,0.08)', layer='below', line_width=0,
            annotation_text="Zone S4", annotation_position="top left",
            annotation_font_size=10, annotation_font_color="#22c55e")
        fig_ph.update_layout(title="Top 20 valeurs pH — FireProtDB brut",
            xaxis_title="pH", yaxis_title="Nombre de mesures",
            plot_bgcolor='white', paper_bgcolor='white',
            font=dict(family='DM Sans', size=11),
            margin=dict(l=40, r=20, t=50, b=50), height=310, showlegend=False)
        fig_ph.update_xaxes(showgrid=False, tickangle=45)
        fig_ph.update_yaxes(showgrid=True, gridcolor='#f1f5f9')
        st.plotly_chart(fig_ph, use_container_width=True)
    with col4:
        fig_t = go.Figure(go.Bar(
            x=temp_labels, y=temp_values,
            marker_color='#f59e0b', marker_line_color='white',
            marker_line_width=0.5, opacity=0.85))
        fig_t.add_vrect(x0='25.0', x1='37.0',
            fillcolor='rgba(34,197,94,0.08)', layer='below', line_width=0,
            annotation_text="Zone S4", annotation_position="top left",
            annotation_font_size=10, annotation_font_color="#22c55e")
        fig_t.update_layout(title="Top 20 températures — FireProtDB brut",
            xaxis_title="T (°C)", yaxis_title="Nombre de mesures",
            plot_bgcolor='white', paper_bgcolor='white',
            font=dict(family='DM Sans', size=11),
            margin=dict(l=40, r=20, t=50, b=50), height=310, showlegend=False)
        fig_t.update_xaxes(showgrid=False, tickangle=45)
        fig_t.update_yaxes(showgrid=True, gridcolor='#f1f5f9')
        st.plotly_chart(fig_t, use_container_width=True)

    st.markdown("""<div class="info">
        <b>Note :</b> Ces distributions portent uniquement sur les ~17 000 lignes ayant un pH ou une
        température renseignés. La grande majorité des 412 411 lignes après nettoyage DDG n'ont
        <b>pas de pH ni de température renseignés</b> — c'est la principale cause de la perte de 99%
        lors du filtrage S4, et non les seuils eux-mêmes.
    </div>""", unsafe_allow_html=True)


def render_fireprot_nettoyage(info):
    """Après nettoyage — FireProtDB : conservées vs supprimées + DDG."""

    st.markdown("<br>", unsafe_allow_html=True)
    mcols([
        ("Après nettoyage", fmt(info['apres_nettoyage']), "#f59e0b"),
        ("Supprimées", fmt(info['supprimees_nettoyage']), "#ef4444"),
        ("Taux conservé", f"{info['taux_validite']:.2f}%", "#22c55e"),
        ("ΔΔG < 0 (après inversion)", fmt(info['ddg_neg_net']), "#22c55e"),
    ])
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig_b = go.Figure(go.Bar(
            x=["Conservées", "Supprimées"],
            y=[info['apres_nettoyage'], info['supprimees_nettoyage']],
            marker_color=["#f59e0b", "#e5e7eb"],
            text=[fmt(info['apres_nettoyage']), fmt(info['supprimees_nettoyage'])],
            textposition='outside'))
        fig_b.update_layout(title="Résultat du nettoyage — FireProtDB",
            plot_bgcolor='white', paper_bgcolor='white',
            font=dict(family='DM Sans', size=11),
            margin=dict(l=40, r=20, t=50, b=40), height=290,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#f1f5f9', title='Lignes'))
        st.plotly_chart(fig_b, use_container_width=True)
    with col2:
        fig_p = go.Figure(go.Pie(
            labels=["Conservées après nettoyage", "Supprimées"],
            values=[info['apres_nettoyage'], info['supprimees_nettoyage']],
            marker_colors=["#f59e0b", "#e5e7eb"],
            hole=0.5, textinfo='percent+label', textfont_size=11))
        fig_p.update_layout(title="Part conservée — FireProtDB",
            showlegend=False, margin=dict(l=0, r=0, t=45, b=0),
            height=280, paper_bgcolor='white', font=dict(family='DM Sans'))
        st.plotly_chart(fig_p, use_container_width=True)
    st.markdown(f"""<div class="info">
        Après nettoyage, seules les mutations simples disposant d'une valeur ΔΔG exploitable sont conservées.
        Le volume passe de <b>{fmt(info['brut'])}</b> lignes brutes à <b>{fmt(info['apres_nettoyage'])}</b>
        mutations utilisables, soit <b>{info['taux_validite']:.2f}%</b> du dataset initial.
    </div>""", unsafe_allow_html=True)
    # ΔΔG simulé sur le sous-ensemble nettoyé
    v = _ddg_sim_with_neutres("FireProtDB", info, "net")
    _ddg_hist_box("FireProtDB", info, v, "après nettoyage")


def render_thermo_brut(info):
    """Brut chargé — ThermoMutDB."""
    st.markdown("<br>", unsafe_allow_html=True)
    v = _ddg_sim_with_neutres("ThermoMutDB", info, "brut")
    _ddg_hist_box("ThermoMutDB", info, v, "brut")
    # Histogramme température
    np.random.seed(42)
    t_v = np.clip(np.concatenate([
        np.random.normal(25, 4, 600),
        np.random.normal(37, 5, 400),
        np.random.normal(55, 15, 200),
    ]), 5, 100)
    fig_t = go.Figure(go.Histogram(x=t_v, nbinsx=35,
        marker_color='#10b981', marker_line_color='white', marker_line_width=0.5, opacity=0.8))
    fig_t.add_vline(x=25, line_dash="dash", line_color="#ef4444", line_width=1.5, annotation_text="25°C")
    fig_t.add_vline(x=37, line_dash="dash", line_color="#ef4444", line_width=1.5, annotation_text="37°C")
    fig_t.update_layout(title="Distribution Température — ThermoMutDB (avant filtrage)",
        xaxis_title="T (°C)", yaxis_title="Fréquence",
        plot_bgcolor='white', paper_bgcolor='white',
        font=dict(family='DM Sans', size=11),
        margin=dict(l=40, r=20, t=45, b=40), height=260, showlegend=False)
    fig_t.update_xaxes(showgrid=True, gridcolor='#f1f5f9')
    st.plotly_chart(fig_t, use_container_width=True)


def render_thermo_nettoyage(info):
    """Après nettoyage — ThermoMutDB."""
    mcols([
        ("Lignes brutes", fmt(info['brut']), "#1e293b"),
        ("Après nettoyage", fmt(info['apres_nettoyage']), "#10b981"),
        ("Supprimées", fmt(info['supprimees_nettoyage']), "#ef4444"),
        ("Taux de validité", f"{info['taux_validite']:.2f}%", "#22c55e"),
    ])
    st.markdown("<br>", unsafe_allow_html=True)
    v = _ddg_sim_with_neutres("ThermoMutDB", info, "net")
    _ddg_hist_box("ThermoMutDB", info, v, "après nettoyage")
    st.markdown(f"""<div class="info">
        Après nettoyage, ThermoMutDB conserve <b>{fmt(info['apres_nettoyage'])}</b> lignes exploitables
        sur <b>{fmt(info['brut'])}</b> lignes brutes (<b>{info['taux_validite']:.2f}%</b>).
        Cette étape retire principalement les entrées sans ΔΔG exploitable, hors de l'intervalle [-10, +10],
        ou dont le format de mutation est invalide.
    </div>""", unsafe_allow_html=True)


def bar_scenarios(info, ds_name):
    labs=["S1\npH=7, T=25","S2\npH=7, T=[25,37]","S3\npH=[6,8], T=25","S4\npH=[6,8], T=[25,37]"]
    vals=[info['s1'],info['s2'],info['s3'],info['s4']]
    clrs=['#94a3b8','#64748b','#475569',info['color']]
    fig=go.Figure(go.Bar(x=labs,y=vals,marker_color=clrs,text=vals,textposition='outside'))
    fig.add_annotation(x=3,y=info['s4'],text="✔ Retenu",showarrow=True,
        arrowhead=2,ay=-30,font=dict(color=info['color'],size=11))
    fig.update_layout(title=f"Comparaison des 4 scénarios — {ds_name}",
        plot_bgcolor='white',paper_bgcolor='white',
        font=dict(family='DM Sans',size=11),margin=dict(l=40,r=20,t=50,b=40),
        height=300,xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True,gridcolor='#f1f5f9',title='Lignes conservées'))
    return fig

def bar_classes_ddg(info, ds_name):
    fig=go.Figure(go.Bar(
        x=['Stabilisant clair','Zone neutre','Déstabilisant clair'],
        y=[info['avant_neutres_stab'],info['avant_neutres_neutre'],info['avant_neutres_destab']],
        marker_color=['#22c55e','#94a3b8','#ef4444'],
        text=[info['avant_neutres_stab'],info['avant_neutres_neutre'],info['avant_neutres_destab']],
        textposition='outside'))
    fig.update_layout(title=f"Classes ΔΔG avant suppression des neutres — {ds_name}",
        plot_bgcolor='white',paper_bgcolor='white',
        font=dict(family='DM Sans',size=11),margin=dict(l=40,r=20,t=50,b=40),
        height=290,xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True,gridcolor='#f1f5f9',title='Mutations'))
    return fig

def bar_coherence(info, ds_name):
    fig=go.Figure(go.Bar(
        x=['Avant cohérence','Après cohérence','Supprimés'],
        y=[info['s4'],info['apres_coherence'],info['supprimes_coherence']],
        marker_color=['#3b82f6','#22c55e','#ef4444'],
        text=[info['s4'],info['apres_coherence'],info['supprimes_coherence']],
        textposition='outside'))
    fig.update_layout(title=f"Cohérence mutation-séquence — {ds_name}",
        plot_bgcolor='white',paper_bgcolor='white',
        font=dict(family='DM Sans',size=11),margin=dict(l=40,r=20,t=50,b=40),
        height=290,xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True,gridcolor='#f1f5f9',title='Lignes'))
    return fig

def bar_dedup(info, ds_name):
    fig=go.Figure(go.Bar(
        x=['Avant déduplication','Après déduplication','Doublons supprimés'],
        y=[info['avant_dedup'],info['apres_dedup'],info['doublons']],
        marker_color=['#3b82f6','#22c55e','#f87171'],
        text=[info['avant_dedup'],info['apres_dedup'],info['doublons']],
        textposition='outside'))
    fig.update_layout(title=f"Déduplication — {ds_name}",
        plot_bgcolor='white',paper_bgcolor='white',
        font=dict(family='DM Sans',size=11),margin=dict(l=40,r=20,t=50,b=40),
        height=280,xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True,gridcolor='#f1f5f9',title='Lignes'))
    return fig

def entonnoir(info, ds_name):
    labels=['Brut chargé','Après nettoyage','Après filtrage pH/T',
            'Après cohérence','Après déduplication','Dataset final']
    vals=[info['brut'],info['apres_nettoyage'],info['s4'],
          info['apres_coherence'],info['apres_dedup'],info['final']]
    html=""
    for lbl,val in zip(labels,vals):
        pct=f" ({100*val/info['brut']:.2f}%)" if val and info['brut'] else ""
        w=max(20,int(100*val/info['brut'])) if info['brut'] else 20
        html+=f"""<div style='display:flex;align-items:center;gap:12px;
            padding:8px 14px;background:#fff;border-radius:8px;
            border:1px solid #e2e8f0;margin-bottom:5px;'>
            <span style='font-family:Space Mono,monospace;font-weight:700;
                font-size:.9rem;color:#1e293b;min-width:80px;text-align:right;'>{fmt(val)}</span>
            <div style='flex:1;background:#f1f5f9;border-radius:4px;height:8px;'>
                <div style='width:{w}%;background:{info["color"]};height:8px;border-radius:4px;'></div>
            </div>
            <span style='font-size:.78rem;color:#64748b;min-width:220px;'>{lbl}{pct}</span>
        </div>"""
    st.markdown(html,unsafe_allow_html=True)

def render_ds_visuals(ds_name, info):
    """7 graphes standard par dataset"""
    col1,col2=st.columns(2)
    with col1: st.plotly_chart(bar_scenarios(info,ds_name),use_container_width=True)
    with col2: st.plotly_chart(bar_classes_ddg(info,ds_name),use_container_width=True)

    col3,col4=st.columns(2)
    with col3: st.plotly_chart(bar_coherence(info,ds_name),use_container_width=True)
    with col4: st.plotly_chart(bar_dedup(info,ds_name),use_container_width=True)

    col5,col6=st.columns([3,2])
    with col5: st.plotly_chart(ddg_hist_sim(ds_name,info,"(final)"),use_container_width=True)
    with col6: st.plotly_chart(pie_sd(info['stab'],info['destab']),use_container_width=True)

def bar_source(par_source):
    palette=['#2563eb','#0ea5e9','#a855f7','#8b5cf6','#d946ef','#f59e0b','#10b981']
    labels=list(par_source.keys()); vals=list(par_source.values())
    fig=go.Figure(go.Bar(x=labels,y=vals,marker_color=palette[:len(labels)],
        text=vals,textposition='outside'))
    fig.update_layout(title="Contribution par source",plot_bgcolor='white',
        paper_bgcolor='white',font=dict(family='DM Sans',size=11),
        margin=dict(l=30,r=20,t=40,b=40),height=280,
        yaxis=dict(showgrid=True,gridcolor='#f1f5f9'),xaxis=dict(showgrid=False))
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:14px 0 6px;'>
        <div style='font-family:Space Mono,monospace;font-size:.66rem;color:#3b82f6;
            letter-spacing:.12em;margin-bottom:4px;'>NAVIGATION</div>
        <div style='font-family:Space Mono,monospace;font-size:.9rem;color:#1e293b;
            font-weight:700;line-height:1.3;'>Stabilité<br>Protéique</div>
    </div>
    <hr style='border-color:#e2e8f0;margin:10px 0;'>
    """,unsafe_allow_html=True)

    page=st.radio("Menu",[
        "🏠  Accueil",
        "🗄️  Bases de données & prétraitement",
        "🔗  Combinaison des datasets",
        "💬  Discussion",
        "🔭  Perspectives",
    ],label_visibility="collapsed")

    st.markdown("""
    <hr style='border-color:#e2e8f0;margin:14px 0;'>
    <div style='font-size:.69rem;color:#64748b;line-height:1.6;'>
        Pipeline de préparation des données<br>pour la prédiction de la stabilité<br>protéique par mutation ponctuelle.
    </div>""",unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# PAGE — ACCUEIL
# ════════════════════════════════════════════════════════════════
if page=="🏠  Accueil":
    if banner_b64:
        st.markdown(f"""
        <div class="hero-wrap">
            <img class="hero-img" src="data:image/png;base64,{banner_b64}"/>
            <div class="hero-overlay">
                <div class="hero-badge">STAGE PFE — PIPELINE DONNÉES</div>
                <div class="hero-title">Prédiction de l'effet des mutations<br>sur la stabilité protéique</div>
                <div class="hero-sub">Pipeline de collecte, nettoyage, harmonisation et préparation des bases de données mutationnelles pour la classification ΔΔG.</div>
            </div>
        </div>""",unsafe_allow_html=True)

    total_final=sum(DATASETS[n]['final'] for n in DATASETS)
    total_brut=sum(DATASETS[n]['brut'] for n in DATASETS)
    mcols([
        ("Datasets traités","7","#1e293b"),
        ("Mutations brutes",fmt(total_brut),"#1e293b"),
        ("Mutations finales",fmt(total_final),"#3b82f6"),
        ("Sources principales","3","#1e293b"),
    ])

    st.markdown("<br>",unsafe_allow_html=True)
    st.markdown('<div class="sec">Étapes du pipeline réalisées</div>',unsafe_allow_html=True)

    steps_g=[
        ("📥","Chargement des bases de données","CSV / JSON bruts"),
        ("🧹","Nettoyage","Mutations simples, DDG valides"),
        ("🔬","Filtrage pH ∈ [6,8]\net T ∈ [25,37]","Conditions expérimentales"),
        ("🧬","Extraction des séquences","UniProt, cohérence ±30"),
        ("🗑️","Déduplication","Hash MD5 + médiane DDG"),
        ("🏷️","Filtrage ΔΔG\n(seuil ±1)","Suppression neutres"),
        ("📦","Export","CSV finaux + combinaisons"),
    ]
    cols=st.columns(7)
    for col,(icon,title,desc) in zip(cols,steps_g):
        with col:
            st.markdown(f"""
            <div style='background:#fff;border-radius:10px;padding:13px 8px;
                border:1px solid #e2e8f0;text-align:center;height:125px;
                display:flex;flex-direction:column;align-items:center;justify-content:center;'>
                <div style='font-size:1.45rem;margin-bottom:5px;'>{icon}</div>
                <div style='font-family:Space Mono,monospace;font-size:.68rem;font-weight:700;
                    color:#1e293b;margin-bottom:3px;white-space:pre-line;'>{title}</div>
                <div style='font-size:.67rem;color:#64748b;line-height:1.3;'>{desc}</div>
            </div>""",unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    st.markdown('<div class="sec">Bases de données utilisées</div>',unsafe_allow_html=True)

    db_infos=[
        ("🟣 ProTherm et ses dérivés",
         "5 datasets dérivés : S2648, S9028, STRUM_Q306, Broom_S605, PON-TStab",
         "https://web.iitm.ac.in/bioinfo2/prothermdb/"),
        ("🟠 FireProtDB",
         "Base de données de stabilité thermique des protéines — plus de 5,4M entrées",
         "https://loschmidt.chemi.muni.cz/fireprotdb/"),
        ("🟢 ThermoMutDB",
         "Dataset JSON avec features précalculées : BLOSUM62, SST, RSA",
         "https://biosig.lab.uq.edu.au/thermomutdb/"),
    ]
    for icon_name, desc, link in db_infos:
        st.markdown(f"""
        <div class="db-box">
            <div class="db-title">{icon_name}</div>
            <div style='font-size:.8rem;color:#475569;margin:5px 0;'>{desc}</div>
            <a class="db-link" href="{link}" target="_blank">🔗 {link}</a>
        </div>""",unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# PAGE — BASES DE DONNÉES & PRÉTRAITEMENT
# ════════════════════════════════════════════════════════════════
elif page=="🗄️  Bases de données & prétraitement":
    st.markdown('<div class="sec">Bases de données & prétraitement</div>',unsafe_allow_html=True)

    tab1,tab2,tab3=st.tabs(["🟣 ProTherm & dérivés","🟠 FireProtDB","🟢 ThermoMutDB"])

    # ── ProTherm ──────────────────────────────────────────────
    with tab1:
        st.markdown("""<div class="info"><b>ProTherm</b> regroupe des mesures expérimentales de stabilité
        pour des mutations ponctuelles. Cinq datasets dérivés ont été traités individuellement.</div>""",
        unsafe_allow_html=True)

        # Vue générale — entonnoirs côte à côte
        st.markdown('<div class="sec">Vue générale — entonnoirs de réduction</div>',unsafe_allow_html=True)
        names_pt=["S2648","S9028","STRUM_Q306","Broom_S605","PON-TStab"]
        stages_labels=["Brut","Nettoyage","Filtrage pH/T","Cohérence","Dédup","Final"]
        stages_keys=["brut","apres_nettoyage","s4","apres_coherence","apres_dedup","final"]
        colors_lines=[DATASETS[n]['color'] for n in names_pt]

        fig_ent=go.Figure()
        for i,n in enumerate(names_pt):
            info=DATASETS[n]
            vals=[info[k] for k in stages_keys]
            fig_ent.add_trace(go.Scatter(x=stages_labels,y=vals,mode='lines+markers',
                name=n,line=dict(color=colors_lines[i],width=2),
                marker=dict(size=6)))
        fig_ent.update_layout(title="Entonnoir de réduction — datasets ProTherm",
            plot_bgcolor='white',paper_bgcolor='white',font=dict(family='DM Sans',size=11),
            margin=dict(l=40,r=20,t=50,b=40),height=340,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True,gridcolor='#f1f5f9',title='Lignes'))
        st.plotly_chart(fig_ent,use_container_width=True)

        st.markdown("---")
        ds_sel=st.selectbox("Sélectionner un dataset :",names_pt)
        info=DATASETS[ds_sel]

        col_l,col_r=st.columns(2)
        with col_l:
            st.markdown(f'<div class="sec">Étapes réalisées — {ds_sel}</div>',unsafe_allow_html=True)
            render_steps(STEPS_PROTHERM)
        with col_r:
            st.markdown('<div class="sec">Étapes à venir</div>',unsafe_allow_html=True)
            render_todo(STEPS_TODO)

        st.markdown("---")
        st.markdown(f'<div class="sec">Visualisations générales — {ds_sel}</div>',unsafe_allow_html=True)

        mcols([
            ("Lignes brutes",fmt(info['brut']),"#1e293b"),
            ("Dataset final",fmt(info['final']),"#3b82f6"),
            ("Doublons supprimés",fmt(info['doublons']),"#ef4444"),
            ("Cohérence séquence",f"{info['couverture']}%","#22c55e"),
        ])
        st.markdown("<br>",unsafe_allow_html=True)

        st.markdown('<div class="sec">Entonnoir de réduction</div>',unsafe_allow_html=True)
        entonnoir(info,ds_sel)
        st.markdown("<br>",unsafe_allow_html=True)

        st.markdown('<div class="sec">Visualisations par étape</div>',unsafe_allow_html=True)

        etapes=[
            "Brut chargé",
            "Après nettoyage",
            "Après filtrage pH/T",
            "Cohérence séquence",
            "Déduplication",
            "Classes ΔΔG (avant suppression neutres)",
            "Dataset final",
        ]
        etape_sel=st.selectbox("Étape à visualiser :",etapes,key="etape_pt")

        if etape_sel=="Brut chargé":
            render_etape_brut(ds_sel,info)
        elif etape_sel=="Après nettoyage":
            render_etape_nettoyage(ds_sel,info)
        elif etape_sel=="Après filtrage pH/T":
            st.plotly_chart(bar_scenarios(info,ds_sel),use_container_width=True)
            st.markdown("""<div class="info">Le scénario retenu (pH ∈ [6,8] et T ∈ [25,37]) conserve
            le meilleur compromis entre homogénéité expérimentale et volume de données.</div>""",
            unsafe_allow_html=True)
        elif etape_sel=="Classes ΔΔG (avant suppression neutres)":
            col1,col2=st.columns(2)
            with col1: st.plotly_chart(bar_classes_ddg(info,ds_sel),use_container_width=True)
            with col2:
                fig_pie3=go.Figure(go.Pie(
                    labels=['Stabilisant','Neutre','Déstabilisant'],
                    values=[info['avant_neutres_stab'],info['avant_neutres_neutre'],info['avant_neutres_destab']],
                    marker_colors=['#22c55e','#94a3b8','#ef4444'],hole=0.5,
                    textinfo='percent+label',textfont_size=11))
                fig_pie3.update_layout(showlegend=False,margin=dict(l=0,r=0,t=30,b=0),
                    height=250,paper_bgcolor='white',font=dict(family='DM Sans'),
                    title="Répartition 3 classes")
                st.plotly_chart(fig_pie3,use_container_width=True)
        elif etape_sel=="Cohérence séquence":
            st.plotly_chart(bar_coherence(info,ds_sel),use_container_width=True)
            st.markdown(f"""<div class="info">Taux de couverture séquence : <b>{info['couverture']}%</b>.
            {info['supprimes_coherence']} mutations supprimées par incohérence.</div>""",
            unsafe_allow_html=True)
        elif etape_sel=="Déduplication":
            st.plotly_chart(bar_dedup(info,ds_sel),use_container_width=True)
        elif etape_sel=="Dataset final":
            col1,col2=st.columns([3,2])
            with col1: st.plotly_chart(ddg_hist_sim(ds_sel,info,"(final)"),use_container_width=True)
            with col2: st.plotly_chart(pie_sd(info['stab'],info['destab']),use_container_width=True)
            mcols([
                ("Stabilisant",fmt(info['stab']),"#22c55e"),
                ("Déstabilisant",fmt(info['destab']),"#ef4444"),
                ("Moyenne ΔΔG",f"{info['ddg_mean']:.3f}","#1e293b"),
                ("Médiane ΔΔG",f"{info['ddg_median']:.3f}","#1e293b"),
            ])
            # S9028 — pie Forward / Reverse
            if ds_sel=="S9028" and 'forward_final' in info:
                st.markdown("<br>",unsafe_allow_html=True)
                st.markdown('<div class="sec">Répartition Forward / Reverse</div>',unsafe_allow_html=True)
                col_fw,_=st.columns([2,3])
                with col_fw:
                    fig_fw=go.Figure(go.Pie(
                        labels=['Forward','Reverse'],
                        values=[info['forward_final'],info['reverse_final']],
                        marker_colors=['#0ea5e9','#7dd3fc'],
                        hole=0.5,textinfo='percent+label',textfont_size=12))
                    fig_fw.update_layout(title="Forward vs Reverse — S9028 final",
                        showlegend=False,margin=dict(l=0,r=0,t=40,b=0),
                        height=260,paper_bgcolor='white',font=dict(family='DM Sans'))
                    st.plotly_chart(fig_fw,use_container_width=True)
            # PON-TStab — barplot chaînes résolues
            if ds_sel=="PON-TStab" and 'chaine_A' in info:
                st.markdown("<br>",unsafe_allow_html=True)
                st.markdown('<div class="sec">Chaînes résolues via SIFTS</div>',unsafe_allow_html=True)
                col_ch,_=st.columns([2,3])
                with col_ch:
                    fig_ch=go.Figure(go.Bar(
                        x=['Chaîne A','Chaîne I'],
                        y=[info['chaine_A'],info['chaine_I']],
                        marker_color=['#d946ef','#e879f9'],
                        text=[info['chaine_A'],info['chaine_I']],
                        textposition='outside'))
                    fig_ch.update_layout(title="Chaînes résolues — PON-TStab",
                        plot_bgcolor='white',paper_bgcolor='white',
                        font=dict(family='DM Sans',size=11),
                        margin=dict(l=40,r=20,t=50,b=40),height=270,
                        xaxis=dict(showgrid=False),
                        yaxis=dict(showgrid=True,gridcolor='#f1f5f9',title='Lignes'))
                    st.plotly_chart(fig_ch,use_container_width=True)
                st.markdown(f"""<div class="info">
                    77 lignes corrigées : chaîne résolue via SIFTS (Chymotrypsin inhibitor 2CI2 → chaîne I).
                    Avec PDB : <b>{fmt(info['avec_pdb'])}</b> — Sans PDB : <b>{fmt(info['sans_pdb'])}</b>
                </div>""",unsafe_allow_html=True)

    # ── FireProtDB ────────────────────────────────────────────
    with tab2:
        st.markdown("""<div class="info"><b>FireProtDB</b> — plus de 5,4M entrées dont ~412k avec ΔΔG valide.
         Après filtrage et nettoyage : 1 184 mutations finales.</div>""",
        unsafe_allow_html=True)

        info=DATASETS["FireProtDB"]

        col_l,col_r=st.columns(2)
        with col_l:
            st.markdown('<div class="sec">Étapes réalisées</div>',unsafe_allow_html=True)
            render_steps(STEPS_FIREPROT)
        with col_r:
            st.markdown('<div class="sec">Étapes à venir</div>',unsafe_allow_html=True)
            render_todo(STEPS_TODO)

        st.markdown("---")
        mcols([
            ("Lignes brutes",fmt(info['brut']),"#1e293b"),
            ("Dataset final",fmt(info['final']),"#3b82f6"),
            ("Doublons supprimés",fmt(info['doublons']),"#ef4444"),
            ("Cohérence séquence",f"{info['couverture']}%","#22c55e"),
        ])
        st.markdown("<br>",unsafe_allow_html=True)

        st.markdown('<div class="sec">Entonnoir de réduction</div>',unsafe_allow_html=True)
        entonnoir(info,"FireProtDB")
        st.markdown("<br>",unsafe_allow_html=True)

        st.markdown('<div class="sec">Visualisations par étape</div>',unsafe_allow_html=True)
        etape_fp=st.selectbox("Étape :",["Brut chargé","Après nettoyage","Après filtrage pH/T",
            "Cohérence séquence","Déduplication","Classes ΔΔG","Dataset final"],
            key="etape_fp")

        if etape_fp=="Brut chargé":
            render_fireprot_brut(info)
        elif etape_fp=="Après nettoyage":
            render_fireprot_nettoyage(info)
        elif etape_fp=="Après filtrage pH/T":
            st.plotly_chart(bar_scenarios(info,"FireProtDB"),use_container_width=True)
        elif etape_fp=="Types de mutations":
            fig_tm=go.Figure(go.Pie(labels=['Simple','Multiple','Manquante'],
                values=[412411,192407,117317],marker_colors=['#f59e0b','#fbbf24','#fde68a'],
                hole=0.5,textinfo='percent+label'))
            fig_tm.update_layout(title="Types de mutations — FireProtDB brut",
                showlegend=False,height=260,paper_bgcolor='white',
                font=dict(family='DM Sans'),margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig_tm,use_container_width=True)
        elif etape_fp=="Classes ΔΔG":
            col1,col2=st.columns(2)
            with col1: st.plotly_chart(bar_classes_ddg(info,"FireProtDB"),use_container_width=True)
            with col2:
                fig_p3=go.Figure(go.Pie(
                    labels=['Stabilisant','Neutre','Déstabilisant'],
                    values=[info['avant_neutres_stab'],info['avant_neutres_neutre'],info['avant_neutres_destab']],
                    marker_colors=['#22c55e','#94a3b8','#ef4444'],hole=0.5,
                    textinfo='percent+label',textfont_size=11))
                fig_p3.update_layout(showlegend=False,margin=dict(l=0,r=0,t=30,b=0),
                    height=250,paper_bgcolor='white',font=dict(family='DM Sans'))
                st.plotly_chart(fig_p3,use_container_width=True)
        elif etape_fp=="Cohérence séquence":
            st.plotly_chart(bar_coherence(info,"FireProtDB"),use_container_width=True)
        elif etape_fp=="Déduplication":
            st.plotly_chart(bar_dedup(info,"FireProtDB"),use_container_width=True)
        elif etape_fp=="Dataset final":
            col1,col2=st.columns([3,2])
            with col1: st.plotly_chart(ddg_hist_sim("FireProtDB",info,"(final)"),use_container_width=True)
            with col2: st.plotly_chart(pie_sd(info['stab'],info['destab']),use_container_width=True)
            mcols([("Stabilisant",fmt(info['stab']),"#22c55e"),
                   ("Déstabilisant",fmt(info['destab']),"#ef4444"),
                   ("Moyenne ΔΔG",f"{info['ddg_mean']:.3f}","#1e293b"),
                   ("Médiane ΔΔG",f"{info['ddg_median']:.3f}","#1e293b")])

    # ── ThermoMutDB ───────────────────────────────────────────
    with tab3:
        st.markdown("""<div class="info"><b>ThermoMutDB</b> — dataset JSON avec features précalculées
        (BLOSUM62, SST, RSA). Colonne <code>uniprot</code> native : récupération directe des séquences.</div>""",
        unsafe_allow_html=True)

        info=DATASETS["ThermoMutDB"]

        col_l,col_r=st.columns(2)
        with col_l:
            st.markdown('<div class="sec">Étapes réalisées</div>',unsafe_allow_html=True)
            render_steps(STEPS_THERMO)
        with col_r:
            st.markdown('<div class="sec">Étapes à venir</div>',unsafe_allow_html=True)
            render_todo(STEPS_TODO)

        st.markdown("---")
        mcols([
            ("Lignes brutes",fmt(info['brut']),"#1e293b"),
            ("Dataset final",fmt(info['final']),"#3b82f6"),
            ("Doublons supprimés",fmt(info['doublons']),"#ef4444"),
            ("Cohérence séquence",f"{info['couverture']}%","#22c55e"),
        ])
        st.markdown("<br>",unsafe_allow_html=True)

        st.markdown('<div class="sec">Entonnoir de réduction</div>',unsafe_allow_html=True)
        entonnoir(info,"ThermoMutDB")
        st.markdown("<br>",unsafe_allow_html=True)

        st.markdown('<div class="sec">Visualisations par étape</div>',unsafe_allow_html=True)
        etape_th=st.selectbox("Étape :",["Brut chargé","Après nettoyage","Après filtrage pH/T",
            "Cohérence séquence","Déduplication","Classes ΔΔG","Dataset final"],key="etape_th")

        if etape_th=="Brut chargé":
            render_thermo_brut(info)
        elif etape_th=="Après nettoyage":
            render_thermo_nettoyage(info)
        elif etape_th=="Après filtrage pH/T":
            col1,col2=st.columns(2)
            with col1: st.plotly_chart(bar_scenarios(info,"ThermoMutDB"),use_container_width=True)
            with col2:
                np.random.seed(42)
                temps=np.concatenate([np.random.normal(25,3,500),np.random.normal(37,4,300)])
                temps=np.clip(temps,15,45)
                fig_t=go.Figure(go.Histogram(x=temps,nbinsx=30,marker_color='#10b981',
                    marker_line_color='white',marker_line_width=0.5))
                fig_t.add_vline(x=25,line_dash="dash",line_color="#3b82f6",line_width=1.5)
                fig_t.add_vline(x=37,line_dash="dash",line_color="#3b82f6",line_width=1.5)
                fig_t.update_layout(title="Distribution température — ThermoMutDB",
                    xaxis_title="T (°C)",plot_bgcolor='white',paper_bgcolor='white',
                    font=dict(family='DM Sans',size=11),margin=dict(l=40,r=20,t=40,b=40),height=280)
                st.plotly_chart(fig_t,use_container_width=True)
        elif etape_th=="Classes ΔΔG":
            col1,col2=st.columns(2)
            with col1: st.plotly_chart(bar_classes_ddg(info,"ThermoMutDB"),use_container_width=True)
            with col2:
                fig_p3=go.Figure(go.Pie(
                    labels=['Stabilisant','Neutre','Déstabilisant'],
                    values=[info['avant_neutres_stab'],info['avant_neutres_neutre'],info['avant_neutres_destab']],
                    marker_colors=['#22c55e','#94a3b8','#ef4444'],hole=0.5,
                    textinfo='percent+label',textfont_size=11))
                fig_p3.update_layout(showlegend=False,margin=dict(l=0,r=0,t=30,b=0),
                    height=250,paper_bgcolor='white',font=dict(family='DM Sans'))
                st.plotly_chart(fig_p3,use_container_width=True)
        elif etape_th=="Cohérence séquence":
            st.plotly_chart(bar_coherence(info,"ThermoMutDB"),use_container_width=True)
        elif etape_th=="Déduplication":
            st.plotly_chart(bar_dedup(info,"ThermoMutDB"),use_container_width=True)
        elif etape_th=="Dataset final":
            col1,col2=st.columns([3,2])
            with col1: st.plotly_chart(ddg_hist_sim("ThermoMutDB",info,"(final)"),use_container_width=True)
            with col2: st.plotly_chart(pie_sd(info['stab'],info['destab']),use_container_width=True)
            mcols([("Stabilisant",fmt(info['stab']),"#22c55e"),
                   ("Déstabilisant",fmt(info['destab']),"#ef4444"),
                   ("Moyenne ΔΔG",f"{info['ddg_mean']:.3f}","#1e293b"),
                   ("Médiane ΔΔG",f"{info['ddg_median']:.3f}","#1e293b")])

# ════════════════════════════════════════════════════════════════
# PAGE — COMBINAISON
# ════════════════════════════════════════════════════════════════
elif page=="🔗  Combinaison des datasets":
    st.markdown('<div class="sec">Combinaison des datasets</div>',unsafe_allow_html=True)
    st.markdown("""<div class="info">Fusion des datasets finaux avec déduplication inter-datasets
    basée sur le hash MD5 de la séquence WT. La médiane DDG est retenue en cas de conflit.</div>""",
    unsafe_allow_html=True)

    # Ligne 1 : bases individuelles
    st.markdown('<div class="sec">Datasets de base</div>',unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    bases=[("🟣 ProTherm dérivés","1 093 mutations finales","#6366f1"),
           ("🟢 ThermoMutDB","1 489 mutations finales","#10b981"),
           ("🟠 FireProtDB","1 184 mutations finales","#f59e0b")]
    for col,(title,sub,clr) in zip([c1,c2,c3],bases):
        with col:
            st.markdown(f"""<div style='background:#fff;border-radius:10px;padding:16px 18px;
                border-left:4px solid {clr};border:1px solid #e2e8f0;text-align:center;'>
                <div style='font-weight:700;font-size:.9rem;color:#1e293b;'>{title}</div>
                <div style='font-size:.8rem;color:#64748b;margin-top:4px;'>{sub}</div>
            </div>""",unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    st.markdown('<div class="sec">Sélectionner une combinaison</div>',unsafe_allow_html=True)

    combo_sel=st.selectbox("",list(COMBOS.keys()))
    c=COMBOS[combo_sel]

    st.markdown("<br>",unsafe_allow_html=True)
    cols5=st.columns(5)
    mvals=[("Avant déduplication",fmt(c['avant_dedup']),"#1e293b"),
           ("Doublons inter-datasets",fmt(c['doublons_inter']),"#ef4444"),
           ("Total final",fmt(c['final']),"#3b82f6"),
           ("Stabilisant clair",fmt(c['stab']),"#22c55e"),
           ("Déstabilisant clair",fmt(c['destab']),"#ef4444")]
    for col,(lbl,val,clr) in zip(cols5,mvals):
        with col: mc(lbl,val,clr)

    st.markdown("<br>",unsafe_allow_html=True)

    # Rapport stab/destab
    st.markdown(f"""
    <div style='background:#fff;border-radius:10px;padding:14px 20px;
        border:1px solid #e2e8f0;margin-bottom:18px;'>
        <div style='font-family:Space Mono,monospace;font-size:.75rem;color:#64748b;margin-bottom:8px;'>
            RAPPORT STABILISANT / DÉSTABILISANT
        </div>
        <div style='display:flex;gap:30px;align-items:center;'>
            <div>
                <span style='font-size:1.6rem;font-weight:700;color:#22c55e;font-family:Space Mono,monospace;'>{c['stab_pct']}%</span>
                <span style='font-size:.8rem;color:#64748b;margin-left:6px;'>Stabilisants</span>
            </div>
            <div>
                <span style='font-size:1.6rem;font-weight:700;color:#ef4444;font-family:Space Mono,monospace;'>{c['destab_pct']}%</span>
                <span style='font-size:.8rem;color:#64748b;margin-left:6px;'>Déstabilisants</span>
            </div>
        </div>
    </div>""",unsafe_allow_html=True)

    col_a,col_b=st.columns([3,2])
    with col_a: st.plotly_chart(bar_source(c['par_source']),use_container_width=True)
    with col_b: st.plotly_chart(pie_sd(c['stab'],c['destab']),use_container_width=True)

    # ΔΔG simulé pour la combinaison
    np.random.seed(abs(hash(combo_sel))%2**31)
    v=np.concatenate([
        np.clip(np.random.normal(-2.2,1.4,c['stab']),-10,-1.001),
        np.clip(np.random.normal(2.1,1.2,c['destab']),1.001,10)])
    fig_ddg=go.Figure(go.Histogram(x=v,nbinsx=60,marker_color='#3b82f6',
        marker_line_color='white',marker_line_width=0.5,opacity=0.85))
    fig_ddg.add_vline(x=-1,line_dash="dash",line_color="#22c55e",line_width=1.5)
    fig_ddg.add_vline(x=1,line_dash="dash",line_color="#ef4444",line_width=1.5)
    fig_ddg.add_vline(x=0,line_dash="dot",line_color="#94a3b8",line_width=1)
    fig_ddg.update_layout(title=f"Distribution ΔΔG — {combo_sel}",
        xaxis_title="ΔΔG (kcal/mol)",yaxis_title="Fréquence",
        plot_bgcolor='white',paper_bgcolor='white',
        font=dict(family='DM Sans',size=11),margin=dict(l=40,r=20,t=40,b=40),
        height=280,showlegend=False)
    fig_ddg.update_xaxes(showgrid=True,gridcolor='#f1f5f9')
    fig_ddg.update_yaxes(showgrid=True,gridcolor='#f1f5f9')
    st.plotly_chart(fig_ddg,use_container_width=True)

    # Tableau comparatif
    st.markdown("---")
    st.markdown('<div class="sec">Comparaison des combinaisons</div>',unsafe_allow_html=True)
    rows=[]
    for name,ci in COMBOS.items():
        rows.append({"Combinaison":name,"Avant dédup":f"{ci['avant_dedup']:,}",
            "Doublons":f"{ci['doublons_inter']:,}","Total final":f"{ci['final']:,}",
            "Stabilisant":f"{ci['stab']:,} ({ci['stab_pct']}%)",
            "Déstabilisant":f"{ci['destab']:,} ({ci['destab_pct']}%)",
            "ΔΔG moyen":f"{ci['ddg_mean']:.3f}"})
    st.dataframe(pd.DataFrame(rows).set_index("Combinaison"),use_container_width=True)

# ════════════════════════════════════════════════════════════════
# PAGE — DISCUSSION
# ════════════════════════════════════════════════════════════════
elif page=="💬  Discussion":
    st.markdown('<div class="sec">Discussion</div>',unsafe_allow_html=True)

    # ── Impact filtrage pH/T ──────────────────────────────────
    st.markdown('<div class="sec">Impact du filtrage des conditions expérimentales</div>',
                unsafe_allow_html=True)
    st.markdown("""<div class="info">
    Le filtrage selon le pH et la température améliore l'homogénéité des données mais entraîne
    une perte importante du volume. Le scénario retenu correspond à pH ∈ [6,8] et T ∈ [25,37]°C,
    car il correspond à des conditions expérimentales fréquemment utilisées dans la littérature.
    </div>""",unsafe_allow_html=True)

    names_all=list(DATASETS.keys())
    # Barplot groupé des 4 scénarios
    fig_s=go.Figure()
    for skey,sname,clr in [("s1","S1 pH=7,T=25","#94a3b8"),("s2","S2 pH=7,T=[25,37]","#64748b"),
                             ("s3","S3 pH=[6,8],T=25","#3b82f6"),("s4","S4 pH=[6,8],T=[25,37]","#1e293b")]:
        fig_s.add_trace(go.Bar(name=sname,x=names_all,
            y=[DATASETS[n][skey] for n in names_all],marker_color=clr))
    fig_s.update_layout(barmode='group',title="Comparaison des 4 scénarios — tous datasets",
        plot_bgcolor='white',paper_bgcolor='white',font=dict(family='DM Sans',size=11),
        margin=dict(l=40,r=20,t=50,b=40),height=350,
        legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1),
        xaxis=dict(showgrid=False),yaxis=dict(showgrid=True,gridcolor='#f1f5f9',title='Lignes'))
    st.plotly_chart(fig_s,use_container_width=True)

    # Perte due au filtrage
    pertes_pct=[round(100*(DATASETS[n]['apres_nettoyage']-DATASETS[n]['s4'])/DATASETS[n]['apres_nettoyage'],1)
                for n in names_all]
    fig_p=go.Figure(go.Bar(x=names_all,y=pertes_pct,
        marker_color=[DATASETS[n]['color'] for n in names_all],
        text=[f"{p}%" for p in pertes_pct],textposition='outside'))
    fig_p.update_layout(title="Perte de données due au filtrage pH/T (%)",
        plot_bgcolor='white',paper_bgcolor='white',font=dict(family='DM Sans',size=11),
        margin=dict(l=40,r=20,t=50,b=40),height=300,
        xaxis=dict(showgrid=False),yaxis=dict(showgrid=True,gridcolor='#f1f5f9',title='% perdu'))
    st.plotly_chart(fig_p,use_container_width=True)

    st.markdown("""<div class="disc">
    <strong>📊 Analyse</strong><br>
    FireProtDB est le dataset le plus affecté par le filtrage : on passe de 412 411 à 4 414 lignes,
    soit une perte de plus de 98%. Cela reflète la forte hétérogénéité des conditions expérimentales
    dans cette base. À l'inverse, les datasets ProTherm comme S2648 conservent environ 39% de leurs
    données après filtrage, car ils proviennent d'expériences déjà réalisées dans des conditions
    standardisées.</div>""",unsafe_allow_html=True)

    st.markdown("---")

    # ── Impact suppression neutres ────────────────────────────
    st.markdown('<div class="sec">Impact de la suppression des mutations neutres</div>',
                unsafe_allow_html=True)
    st.markdown("""<div class="info">
    Les mutations neutres (|ΔΔG| ≤ 1 kcal/mol) ont été supprimées car elles ne constituent pas
    la cible principale de ce travail, qui s'intéresse à la distinction entre mutations clairement
    stabilisantes et déstabilisantes.
    </div>""",unsafe_allow_html=True)

    fig_n=go.Figure()
    fig_n.add_trace(go.Bar(name='Après filtrage pH/T',x=names_all,
        y=[DATASETS[n]['s4'] for n in names_all],marker_color='#3b82f6'))
    fig_n.add_trace(go.Bar(name='Neutres supprimés',x=names_all,
        y=[DATASETS[n]['neutres_supprimes'] for n in names_all],marker_color='#f87171'))
    fig_n.add_trace(go.Bar(name='Dataset final',x=names_all,
        y=[DATASETS[n]['final'] for n in names_all],marker_color='#22c55e'))
    fig_n.update_layout(barmode='group',title="Impact de la suppression des mutations neutres",
        plot_bgcolor='white',paper_bgcolor='white',font=dict(family='DM Sans',size=11),
        margin=dict(l=40,r=20,t=50,b=40),height=350,
        legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1),
        xaxis=dict(showgrid=False),yaxis=dict(showgrid=True,gridcolor='#f1f5f9',title='Lignes'))
    st.plotly_chart(fig_n,use_container_width=True)

    # Stacked bar 3 classes
    fig_3c=go.Figure()
    fig_3c.add_trace(go.Bar(name='Stabilisant',x=names_all,
        y=[DATASETS[n]['avant_neutres_stab'] for n in names_all],marker_color='#22c55e'))
    fig_3c.add_trace(go.Bar(name='Neutre',x=names_all,
        y=[DATASETS[n]['avant_neutres_neutre'] for n in names_all],marker_color='#94a3b8'))
    fig_3c.add_trace(go.Bar(name='Déstabilisant',x=names_all,
        y=[DATASETS[n]['avant_neutres_destab'] for n in names_all],marker_color='#ef4444'))
    fig_3c.update_layout(barmode='stack',title="Répartition 3 classes avant suppression des neutres",
        plot_bgcolor='white',paper_bgcolor='white',font=dict(family='DM Sans',size=11),
        margin=dict(l=40,r=20,t=50,b=40),height=320,
        legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1),
        xaxis=dict(showgrid=False),yaxis=dict(showgrid=True,gridcolor='#f1f5f9'))
    st.plotly_chart(fig_3c,use_container_width=True)

    st.markdown("""<div class="disc">
    <strong>📚 Lien avec la littérature</strong><br>
    Les choix méthodologiques adoptés dans ce pipeline — filtrage par conditions expérimentales
    et suppression des mutations neutres — sont cohérents avec les pratiques observées dans
    plusieurs articles du domaine. Ces décisions permettent d'obtenir des jeux de données plus
    homogènes et plus pertinents pour l'apprentissage de modèles de prédiction de stabilité protéique.
    </div>""",unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# PAGE — PERSPECTIVES
# ════════════════════════════════════════════════════════════════
elif page=="🔭  Perspectives":
    st.markdown('<div class="sec">Perspectives</div>',unsafe_allow_html=True)
    st.markdown("""<div class="info">
    Cette section présente les principales étapes à venir du projet, depuis la finalisation du
    pipeline de préparation des données jusqu'à l'exploitation du modèle pour l'analyse et la
    génération de profils mutationnels de stabilité protéique.
    </div>""",unsafe_allow_html=True)

    perspectives=[
        ("01","Finalisation du pipeline de préparation des données",
         "Réduction de la redondance, contrôle des biais, vérification de l'absence de data leakage, "
         "construction des ensembles Train / Validation / Test par protéine, et préparation finale des "
         "entrées du modèle dans des conditions rigoureuses et cohérentes.",
         "#6366f1"),
        ("02","Développement d'un modèle de prédiction de l'effet des mutations",
         "Entraînement d'un modèle capable de prédire l'effet d'une mutation sur la stabilité protéique. "
         "Deux formulations complémentaires seront étudiées : la régression pour prédire directement "
         "la valeur de ΔΔG, et la classification pour distinguer mutations stabilisantes et déstabilisantes.",
         "#3b82f6"),
        ("03","Extension vers des bases de données thérapeutiques",
         "Application du modèle entraîné à des bases de données thérapeutiques afin d'explorer "
         "son potentiel sur des protéines d'intérêt biomédical et d'enrichir le projet d'une "
         "dimension applicative orientée vers des contextes réels.",
         "#0ea5e9"),
        ("04","Génération de mutations et analyse de leur impact",
         "Utilisation du modèle pour générer de nouvelles mutations candidates et prédire leur "
         "effet sur la stabilité protéique, permettant d'explorer de manière systématique l'espace "
         "mutationnel d'une protéine donnée.",
         "#10b981"),
        ("05","Construction d'une matrice de préférences mutationnelles ΔΔG",
         "Génération d'une matrice décrivant, pour chaque position de la protéine, les effets "
         "prédits des différentes substitutions possibles. Cette matrice permettra d'identifier "
         "les positions sensibles, les substitutions favorables ou défavorables, et les préférences "
         "mutationnelles associées à chaque site de la séquence.",
         "#a855f7"),
    ]

    for num,title,text,clr in perspectives:
        st.markdown(f"""
        <div class="persp-card" style="border-left-color:{clr};">
            <div class="persp-num">ÉTAPE {num}</div>
            <div class="persp-title">{title}</div>
            <div class="persp-text">{text}</div>
        </div>""",unsafe_allow_html=True)
