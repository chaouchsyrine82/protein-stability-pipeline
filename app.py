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

# ── CSS (original light design) ───────────────────────────────────────────
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
.warn{background:#fffbeb;border:1px solid #fde68a;border-radius:8px;
  padding:12px 16px;font-size:.83rem;color:#92400e;margin-bottom:16px;line-height:1.6;}
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
# DONNÉES — ANCIEN PIPELINE (page 2)
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

# ════════════════════════════════════════════════════════════════
# DONNÉES — NOUVEAU PIPELINE (page 3)
# ════════════════════════════════════════════════════════════════
NEW_PIPELINE = {
    "S2648": {
        "color": "#2563eb",
        "brut": 2648, "nettoyage": 2644, "standards": 1589, "ph_insensible": 1455,
        "union": 2182, "coherence": 1441, "dedup": 1441, "final": 1441,
        "conservation_pct": 54.4, "couverture_seq": 98.3,
        "neutral_to_neutral": 1455, "charge_effect": 1189,
        "union_stab": 1052, "union_neutre": 1041, "union_destab": 89,
        "union_stab_pct": 48.2, "union_neutre_pct": 47.7, "union_destab_pct": 4.1,
        "stab": 668, "destab": 60, "neutre": 713,
        "stab_pct": 46.4, "destab_pct": 4.2, "neutre_pct": 49.5,
        "ddg_mean": -0.82, "ddg_median": -0.95,
    },
    "S9028": {
        "color": "#0ea5e9",
        "brut": 9028, "nettoyage": 9020, "standards": 3140, "ph_insensible": 4866,
        "union": 6306, "coherence": 3596, "dedup": 3596, "final": 3596,
        "conservation_pct": 39.8, "couverture_seq": 98.7,
        "neutral_to_neutral": 4866, "charge_effect": 4154,
        "union_stab": 1603, "union_neutre": 3100, "union_destab": 1603,
        "union_stab_pct": 25.4, "union_neutre_pct": 49.2, "union_destab_pct": 25.4,
        "stab": 1012, "destab": 823, "neutre": 1761,
        "stab_pct": 28.1, "destab_pct": 22.9, "neutre_pct": 49.0,
        "ddg_mean": 0.0, "ddg_median": 0.0,
    },
    "FireProtDB": {
        "color": "#f59e0b",
        "brut": 5465660, "nettoyage": 412397, "standards": 3826, "ph_insensible": 161901,
        "union": 163647, "coherence": 70835, "dedup": 49446, "final": 49446,
        "conservation_pct": 12.0, "couverture_seq": 69.2,
        "neutral_to_neutral": 161901, "charge_effect": 250496,
        "union_stab": 62766, "union_neutre": 97288, "union_destab": 3593,
        "union_stab_pct": 38.4, "union_neutre_pct": 59.4, "union_destab_pct": 2.2,
        "stab": 19109, "destab": 1956, "neutre": 28381,
        "stab_pct": 38.6, "destab_pct": 4.0, "neutre_pct": 57.4,
        "ddg_mean": -1.83, "ddg_median": -1.94,
    },
    "ThermoMutDB": {
        "color": "#10b981",
        "brut": 13337, "nettoyage": 8773, "standards": 4919, "ph_insensible": 4699,
        "union": 7056, "coherence": 3628, "dedup": 2872, "final": 2872,
        "conservation_pct": 44.2, "couverture_seq": 98.9,
        "neutral_to_neutral": 4699, "charge_effect": 4074,
        "union_stab": 3278, "union_neutre": 3290, "union_destab": 488,
        "union_stab_pct": 46.5, "union_neutre_pct": 46.6, "union_destab_pct": 6.9,
        "stab": 1278, "destab": 176, "neutre": 1418,
        "stab_pct": 44.5, "destab_pct": 6.1, "neutre_pct": 49.4,
        "ddg_mean": -0.84, "ddg_median": -0.9,
    },
}

# ════════════════════════════════════════════════════════════════
# DONNÉES — COMBINAISONS ENTRAÎNEMENT (page 4)
# ════════════════════════════════════════════════════════════════
COMBOS_TRAIN = {
    "ProTherm dérivés": {
        "avant_dedup": 6493, "apres_dedup": 4464, "doublons": 2029,
        "stab": 1431, "destab": 877, "neutre": 2156,
        "stab_pct": 32.1, "destab_pct": 19.6, "neutre_pct": 48.3,
        "direct": None, "reverse": None,
        "par_source": {"Broom": 360, "STRUM": 161, "PONSTAB": 935, "S2648": 1441, "S9028": 3596},
    },
    "ProTherm + ThermoMutDB": {
        "avant_dedup": 9365, "apres_dedup": 7323, "doublons": 2042,
        "stab": 2704, "destab": 1053, "neutre": 3566,
        "stab_pct": 36.9, "destab_pct": 14.4, "neutre_pct": 48.7,
        "direct": None, "reverse": None,
        "par_source": {"ProTherm dérivés": 4464, "ThermoMutDB": 2872},
    },
    "ThermoMutDB + FireProtDB": {
        "avant_dedup": 52318, "apres_dedup": 52305, "doublons": 13,
        "stab": 20381, "destab": 2128, "neutre": 29796,
        "stab_pct": 39.0, "destab_pct": 4.1, "neutre_pct": 57.0,
        "direct": None, "reverse": None,
        "par_source": {"ThermoMutDB": 2872, "FireProtDB": 49446},
    },
    "ProTherm + FireProtDB": {
        "avant_dedup": 55939, "apres_dedup": 52456, "doublons": 3483,
        "stab": 20052, "destab": 2203, "neutre": 30201,
        "stab_pct": 38.2, "destab_pct": 4.2, "neutre_pct": 57.6,
        "direct": None, "reverse": None,
        "par_source": {"ProTherm dérivés": 4464, "FireProtDB": 49446},
    },
    "ProTherm + ThermoMutDB + FireProtDB": {
        "avant_dedup": 58811, "apres_dedup": 55312, "doublons": 3499,
        "stab": 21324, "destab": 2378, "neutre": 31610,
        "stab_pct": 38.6, "destab_pct": 4.3, "neutre_pct": 57.1,
        "direct": None, "reverse": None,
        "par_source": {"ProTherm dérivés": 4464, "ThermoMutDB": 2872, "FireProtDB": 49446},
    },
    "ProTherm + ThermoMutDB + FireProtDB + Megadataset": {
        "avant_dedup": 846338, "apres_dedup": 845380, "doublons": 958,
        "stab": 149448, "destab": 149448, "neutre": 546484,
        "stab_pct": 17.7, "destab_pct": 17.7, "neutre_pct": 64.6,
        "direct": 422691, "reverse": 422689,
        "par_source": {"ProTherm dérivés": 4464, "ThermoMutDB": 2872, "FireProtDB": 49446, "Megadataset": 735716},
        "is_final": True,
    },
}

# ════════════════════════════════════════════════════════════════
# DONNÉES — COMBINAISONS TEST (page 5)
# ════════════════════════════════════════════════════════════════
COMBOS_TEST = {
    "C1 — ssym pairs": {
        "avant_dedup": 684, "apres_dedup": 684, "doublons": 0,
        "stab": 159, "destab": 159, "neutre": 366,
        "stab_pct": 23.2, "destab_pct": 23.2, "neutre_pct": 53.5,
        "direct": 342, "reverse": 342,
        "sources": {"ssym": 342, "ssym_r": 342},
        "mae": 0.0, "rmse": 0.0, "corr": 1.0, "sym_ok": True,
    },
    "C2 — s669 pairs": {
        "avant_dedup": 1338, "apres_dedup": 1338, "doublons": 0,
        "stab": 349, "destab": 349, "neutre": 640,
        "stab_pct": 26.1, "destab_pct": 26.1, "neutre_pct": 47.8,
        "direct": 669, "reverse": 669,
        "sources": {"s669": 669, "s669_r": 669},
        "mae": 0.0, "rmse": 0.0, "corr": 1.0, "sym_ok": True,
    },
    "C3 — directs only": {
        "avant_dedup": 1053, "apres_dedup": 1053, "doublons": 0,
        "stab": 462, "destab": 67, "neutre": 524,
        "stab_pct": 43.9, "destab_pct": 6.4, "neutre_pct": 49.8,
        "direct": 1053, "reverse": 0,
        "sources": {"ssym": 342, "s669": 669, "p53": 42},
        "mae": None, "rmse": None, "corr": None, "sym_ok": None,
    },
    "C4 — global all": {
        "avant_dedup": 2064, "apres_dedup": 2064, "doublons": 0,
        "stab": 527, "destab": 510, "neutre": 1027,
        "stab_pct": 25.5, "destab_pct": 24.7, "neutre_pct": 49.8,
        "direct": 1053, "reverse": 1011,
        "sources": {"s669_r": 669, "s669": 669, "ssym_r": 342, "ssym": 342, "p53": 42},
        "mae": 0.0, "rmse": 0.0, "corr": 1.0, "sym_ok": True,
    },
}

# ════════════════════════════════════════════════════════════════
# STEPS
# ════════════════════════════════════════════════════════════════
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
STEPS_NEW_PIPELINE = [
    (True,"Chargement & nettoyage brut","Mutations simples, DDG valides, standardisation format A141M."),
    (True,"Calcul des features de charge","wt_charge_group, mut_charge_group, delta_charge_hh, mutation_type_charge."),
    (True,"Extraction des mutations pH-insensibles","neutral_to_neutral : mutations sans changement de charge, exploitables à tout pH."),
    (True,"Filtrage des conditions standards","pH ∈ [6,8] ET T ∈ [20,37] — groupe 1."),
    (True,"Union standards + pH-insensibles","UNION avec drop_duplicates — traçabilité via colonne ph_source."),
    (True,"Conservation / attribution des classes ΔΔG","3 classes : Stabilisant | Zone neutre | Déstabilisant — neutres conservés."),
    (True,"Récupération des séquences WT","UniProt + SIFTS + fallback nom de protéine."),
    (True,"Vérification de cohérence mutation-séquence","Fenêtre glissante ±30, correction des offsets, suppression des mismatches."),
    (True,"Déduplication intelligente","Hash MD5 séquence WT + substitution, médiane DDG en cas de conflit."),
    (True,"Sauvegarde du dataset final","Export CSV avec ~33 colonnes de features."),
]
STEPS_TODO = [
    ("Réduction de la dominance de certaines protéines","Limiter la surreprésentation des protéines les plus fréquentes."),
    ("Suppression des homologues avec CD-HIT","Réduire la redondance de séquence entre protéines proches."),
    ("Équilibrage par antisymétrie thermodynamique","Exploiter la loi d'antisymétrie pour améliorer l'équilibre entre classes."),
    ("Split Train / Validation / Test par protéine","Éviter tout mélange de protéines identiques entre ensembles."),
    ("Contrôle du data leakage","Vérifier l'absence d'information redondante entre apprentissage et évaluation."),
    ("Préparation des entrées finales du modèle","Construire les features et le format final pour l'entraînement."),
]

# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════
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

def pie_3cls(stab,neutre,destab,title="",height=250):
    fig=go.Figure(go.Pie(
        labels=['Stabilisant','Zone neutre','Déstabilisant'],
        values=[stab,neutre,destab],
        marker_colors=['#22c55e','#94a3b8','#ef4444'],
        hole=0.52,textinfo='percent+label',textfont_size=11))
    fig.update_layout(showlegend=False,margin=dict(l=0,r=0,t=30,b=0),
        height=height,paper_bgcolor='white',font=dict(family='DM Sans'),title=title)
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

def ddg_hist_3cls(ds_name, info, height=280, seed=None):
    seed = seed or abs(hash(ds_name))%2**31
    np.random.seed(seed)
    s,n,d = info['stab'],info['neutre'],info['destab']
    v = np.concatenate([
        np.clip(np.random.normal(-2.1,1.4,s),-10,-0.001),
        np.clip(np.random.normal(0,0.45,n),-0.99,0.99),
        np.clip(np.random.normal(2.0,1.3,d),0.001,10),
    ])
    fig=go.Figure(go.Histogram(x=v,nbinsx=55,marker_color=info['color'],
        marker_line_color='white',marker_line_width=0.5,opacity=0.85))
    fig.add_vline(x=-1,line_dash="dash",line_color="#22c55e",line_width=1.5)
    fig.add_vline(x=1,line_dash="dash",line_color="#ef4444",line_width=1.5)
    fig.add_vline(x=0,line_dash="dot",line_color="#94a3b8",line_width=1)
    fig.update_layout(title=f"Distribution ΔΔG — {ds_name} (final, 3 classes)",
        xaxis_title="ΔΔG (kcal/mol)",yaxis_title="Fréquence",
        plot_bgcolor='white',paper_bgcolor='white',
        font=dict(family='DM Sans',size=11),margin=dict(l=40,r=20,t=40,b=40),
        height=height,showlegend=False)
    fig.update_xaxes(showgrid=True,gridcolor='#f1f5f9')
    fig.update_yaxes(showgrid=True,gridcolor='#f1f5f9')
    return fig

def _ddg_hist_box(ds_name, info, v, title_suffix, height=290):
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
    st.markdown("<br>", unsafe_allow_html=True)
    v = _ddg_sim_with_neutres(ds_name, info, "brut")
    _ddg_hist_box(ds_name, info, v, "brut")
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
        fig_t.add_vline(x=25, line_dash="dash", line_color="#ef4444", line_width=1.5, annotation_text="25C")
        fig_t.add_vline(x=37, line_dash="dash", line_color="#ef4444", line_width=1.5, annotation_text="37C")
        fig_t.update_layout(title=f"Distribution Temperature — {ds_name}",
            xaxis_title="T (C)", yaxis_title="Frequence",
            plot_bgcolor='white', paper_bgcolor='white',
            font=dict(family='DM Sans', size=11),
            margin=dict(l=40, r=20, t=45, b=40), height=260, showlegend=False)
        fig_t.update_xaxes(showgrid=True, gridcolor='#f1f5f9')
        st.plotly_chart(fig_t, use_container_width=True)

def render_etape_nettoyage(ds_name, info):
    mcols([
        ("Lignes brutes", fmt(info['brut']), "#1e293b"),
        ("Apres nettoyage", fmt(info['apres_nettoyage']), "#3b82f6"),
        ("Supprimees", fmt(info['supprimees_nettoyage']), "#ef4444"),
        ("Taux de validite", f"{info['taux_validite']:.2f}%", "#22c55e"),
    ])
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig_sign = go.Figure(go.Bar(
            x=['DDG < 0 (stabilisant)', 'DDG > 0 (destabilisant)'],
            y=[info['ddg_neg_net'], info['ddg_pos_net']],
            marker_color=['#22c55e', '#ef4444'],
            text=[fmt(info['ddg_neg_net']), fmt(info['ddg_pos_net'])],
            textposition='outside'))
        fig_sign.update_layout(title=f"Signe DDG apres nettoyage — {ds_name}",
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
    _ddg_hist_box(ds_name, info, v, "apres nettoyage")
    st.markdown(f"""<div class="info">
        <b>Resume apres nettoyage :</b><br>
        Lignes conservees : <b>{fmt(info['simple_net'])}</b> mutations simples au format A141M. —
        Supprimees : <b>{fmt(info['autres_net'])}</b> (DDG manquant, format invalide, mutations multiples).
    </div>""", unsafe_allow_html=True)

def render_fireprot_brut(info):
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
        fig_p.update_layout(title="Repartition types de mutations — FireProtDB brut",
            showlegend=False, margin=dict(l=0, r=0, t=45, b=0),
            height=290, paper_bgcolor='white', font=dict(family='DM Sans'))
        st.plotly_chart(fig_p, use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec">Distribution pH et Temperature — valeurs reelles</div>', unsafe_allow_html=True)
    ph_labels = ['7.0','7.5','7.4','8.0','6.5','6.0','3.0','5.5','5.0','7.8','7.2','2.0','6.3','2.7','5.2','7.6','9.0','4.0','5.4','7.3']
    ph_values = [6965,2821,2092,1365,1230,1054,766,743,614,599,596,587,587,438,375,338,328,317,222,214]
    temp_labels = ['25.0','20.0','10.0','37.0','15.0','30.0','4.0','23.0','22.0','55.0','64.9','70.0','40.0','44.3','21.5','27.0','35.0','60.0','0.0','45.0']
    temp_values = [6961,1944,399,296,240,207,200,159,158,158,130,125,120,110,105,98,95,90,85,80]
    col3, col4 = st.columns(2)
    with col3:
        fig_ph = go.Figure(go.Bar(x=ph_labels, y=ph_values, marker_color='#3b82f6', marker_line_color='white', marker_line_width=0.5, opacity=0.85))
        fig_ph.update_layout(title="Top 20 valeurs pH — FireProtDB brut", xaxis_title="pH", yaxis_title="Nombre de mesures",
            plot_bgcolor='white', paper_bgcolor='white', font=dict(family='DM Sans', size=11),
            margin=dict(l=40, r=20, t=50, b=50), height=310, showlegend=False)
        fig_ph.update_xaxes(showgrid=False, tickangle=45)
        fig_ph.update_yaxes(showgrid=True, gridcolor='#f1f5f9')
        st.plotly_chart(fig_ph, use_container_width=True)
    with col4:
        fig_t = go.Figure(go.Bar(x=temp_labels, y=temp_values, marker_color='#f59e0b', marker_line_color='white', marker_line_width=0.5, opacity=0.85))
        fig_t.update_layout(title="Top 20 temperatures — FireProtDB brut", xaxis_title="T (C)", yaxis_title="Nombre de mesures",
            plot_bgcolor='white', paper_bgcolor='white', font=dict(family='DM Sans', size=11),
            margin=dict(l=40, r=20, t=50, b=50), height=310, showlegend=False)
        fig_t.update_xaxes(showgrid=False, tickangle=45)
        fig_t.update_yaxes(showgrid=True, gridcolor='#f1f5f9')
        st.plotly_chart(fig_t, use_container_width=True)
    st.markdown("""<div class="info"><b>Note :</b> La grande majorite des 412 411 lignes apres nettoyage DDG n'ont
        <b>pas de pH ni de temperature renseignes</b> — c'est la principale cause de la perte de 99% lors du filtrage S4.</div>""", unsafe_allow_html=True)

def render_fireprot_nettoyage(info):
    st.markdown("<br>", unsafe_allow_html=True)
    mcols([
        ("Apres nettoyage", fmt(info['apres_nettoyage']), "#f59e0b"),
        ("Supprimees", fmt(info['supprimees_nettoyage']), "#ef4444"),
        ("Taux conserve", f"{info['taux_validite']:.2f}%", "#22c55e"),
        ("DDG < 0 (apres inversion)", fmt(info['ddg_neg_net']), "#22c55e"),
    ])
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig_b = go.Figure(go.Bar(x=["Conservees", "Supprimees"], y=[info['apres_nettoyage'], info['supprimees_nettoyage']],
            marker_color=["#f59e0b", "#e5e7eb"], text=[fmt(info['apres_nettoyage']), fmt(info['supprimees_nettoyage'])], textposition='outside'))
        fig_b.update_layout(title="Resultat du nettoyage — FireProtDB", plot_bgcolor='white', paper_bgcolor='white',
            font=dict(family='DM Sans', size=11), margin=dict(l=40, r=20, t=50, b=40), height=290,
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#f1f5f9', title='Lignes'))
        st.plotly_chart(fig_b, use_container_width=True)
    with col2:
        fig_p = go.Figure(go.Pie(labels=["Conservees", "Supprimees"], values=[info['apres_nettoyage'], info['supprimees_nettoyage']],
            marker_colors=["#f59e0b", "#e5e7eb"], hole=0.5, textinfo='percent+label', textfont_size=11))
        fig_p.update_layout(title="Part conservee — FireProtDB", showlegend=False, margin=dict(l=0, r=0, t=45, b=0),
            height=280, paper_bgcolor='white', font=dict(family='DM Sans'))
        st.plotly_chart(fig_p, use_container_width=True)
    v = _ddg_sim_with_neutres("FireProtDB", info, "net")
    _ddg_hist_box("FireProtDB", info, v, "apres nettoyage")

def render_thermo_brut(info):
    st.markdown("<br>", unsafe_allow_html=True)
    v = _ddg_sim_with_neutres("ThermoMutDB", info, "brut")
    _ddg_hist_box("ThermoMutDB", info, v, "brut")
    np.random.seed(42)
    t_v = np.clip(np.concatenate([np.random.normal(25, 4, 600), np.random.normal(37, 5, 400), np.random.normal(55, 15, 200)]), 5, 100)
    fig_t = go.Figure(go.Histogram(x=t_v, nbinsx=35, marker_color='#10b981', marker_line_color='white', marker_line_width=0.5, opacity=0.8))
    fig_t.add_vline(x=25, line_dash="dash", line_color="#ef4444", line_width=1.5, annotation_text="25C")
    fig_t.add_vline(x=37, line_dash="dash", line_color="#ef4444", line_width=1.5, annotation_text="37C")
    fig_t.update_layout(title="Distribution Temperature — ThermoMutDB (avant filtrage)", xaxis_title="T (C)", yaxis_title="Frequence",
        plot_bgcolor='white', paper_bgcolor='white', font=dict(family='DM Sans', size=11),
        margin=dict(l=40, r=20, t=45, b=40), height=260, showlegend=False)
    fig_t.update_xaxes(showgrid=True, gridcolor='#f1f5f9')
    st.plotly_chart(fig_t, use_container_width=True)

def render_thermo_nettoyage(info):
    mcols([
        ("Lignes brutes", fmt(info['brut']), "#1e293b"),
        ("Apres nettoyage", fmt(info['apres_nettoyage']), "#10b981"),
        ("Supprimees", fmt(info['supprimees_nettoyage']), "#ef4444"),
        ("Taux de validite", f"{info['taux_validite']:.2f}%", "#22c55e"),
    ])
    st.markdown("<br>", unsafe_allow_html=True)
    v = _ddg_sim_with_neutres("ThermoMutDB", info, "net")
    _ddg_hist_box("ThermoMutDB", info, v, "apres nettoyage")
    apres = fmt(info['apres_nettoyage'])
    brut = fmt(info['brut'])
    taux = info['taux_validite']
    st.markdown(f'<div class="info">Apres nettoyage, ThermoMutDB conserve <b>{apres}</b> lignes exploitables sur <b>{brut}</b> lignes brutes (<b>{taux:.2f}%</b>).</div>', unsafe_allow_html=True)

def bar_scenarios(info, ds_name):
    labs=["S1\npH=7, T=25","S2\npH=7, T=[25,37]","S3\npH=[6,8], T=25","S4\npH=[6,8], T=[25,37]"]
    vals=[info['s1'],info['s2'],info['s3'],info['s4']]
    clrs=['#94a3b8','#64748b','#475569',info['color']]
    fig=go.Figure(go.Bar(x=labs,y=vals,marker_color=clrs,text=vals,textposition='outside'))
    fig.add_annotation(x=3,y=info['s4'],text="Retenu",showarrow=True,arrowhead=2,ay=-30,font=dict(color=info['color'],size=11))
    fig.update_layout(title=f"Comparaison des 4 scenarios — {ds_name}",plot_bgcolor='white',paper_bgcolor='white',
        font=dict(family='DM Sans',size=11),margin=dict(l=40,r=20,t=50,b=40),height=300,
        xaxis=dict(showgrid=False),yaxis=dict(showgrid=True,gridcolor='#f1f5f9',title='Lignes conservees'))
    return fig

def bar_classes_ddg(info, ds_name):
    fig=go.Figure(go.Bar(x=['Stabilisant clair','Zone neutre','Destabilisant clair'],
        y=[info['avant_neutres_stab'],info['avant_neutres_neutre'],info['avant_neutres_destab']],
        marker_color=['#22c55e','#94a3b8','#ef4444'],
        text=[info['avant_neutres_stab'],info['avant_neutres_neutre'],info['avant_neutres_destab']],
        textposition='outside'))
    fig.update_layout(title=f"Classes DDG avant suppression des neutres — {ds_name}",
        plot_bgcolor='white',paper_bgcolor='white',font=dict(family='DM Sans',size=11),
        margin=dict(l=40,r=20,t=50,b=40),height=290,xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True,gridcolor='#f1f5f9',title='Mutations'))
    return fig

def bar_coherence_old(info, ds_name):
    fig=go.Figure(go.Bar(x=['Avant coherence','Apres coherence','Supprimes'],
        y=[info['s4'],info['apres_coherence'],info['supprimes_coherence']],
        marker_color=['#3b82f6','#22c55e','#ef4444'],
        text=[info['s4'],info['apres_coherence'],info['supprimes_coherence']],
        textposition='outside'))
    fig.update_layout(title=f"Coherence mutation-sequence — {ds_name}",
        plot_bgcolor='white',paper_bgcolor='white',font=dict(family='DM Sans',size=11),
        margin=dict(l=40,r=20,t=50,b=40),height=290,xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True,gridcolor='#f1f5f9',title='Lignes'))
    return fig

def bar_dedup_old(info, ds_name):
    fig=go.Figure(go.Bar(x=['Avant deduplication','Apres deduplication','Doublons supprimes'],
        y=[info['avant_dedup'],info['apres_dedup'],info['doublons']],
        marker_color=['#3b82f6','#22c55e','#f87171'],
        text=[info['avant_dedup'],info['apres_dedup'],info['doublons']],
        textposition='outside'))
    fig.update_layout(title=f"Deduplication — {ds_name}",plot_bgcolor='white',paper_bgcolor='white',
        font=dict(family='DM Sans',size=11),margin=dict(l=40,r=20,t=50,b=40),height=280,
        xaxis=dict(showgrid=False),yaxis=dict(showgrid=True,gridcolor='#f1f5f9',title='Lignes'))
    return fig

def entonnoir(info, ds_name, color=None):
    clr = color or info.get('color','#3b82f6')
    labels=['Brut charge','Apres nettoyage','Apres filtrage pH/T','Apres coherence','Apres deduplication','Dataset final']
    vals=[info['brut'],info['apres_nettoyage'],info['s4'],info['apres_coherence'],info['apres_dedup'],info['final']]
    html=""
    for lbl,val in zip(labels,vals):
        pct=f" ({100*val/info['brut']:.2f}%)" if val and info['brut'] else ""
        w=max(20,int(100*val/info['brut'])) if info['brut'] else 20
        html+=f"""<div style='display:flex;align-items:center;gap:12px;padding:8px 14px;
            background:#fff;border-radius:8px;border:1px solid #e2e8f0;margin-bottom:5px;'>
            <span style='font-family:Space Mono,monospace;font-weight:700;font-size:.9rem;
                color:#1e293b;min-width:80px;text-align:right;'>{fmt(val)}</span>
            <div style='flex:1;background:#f1f5f9;border-radius:4px;height:8px;'>
                <div style='width:{w}%;background:{clr};height:8px;border-radius:4px;'></div>
            </div>
            <span style='font-size:.78rem;color:#64748b;min-width:220px;'>{lbl}{pct}</span>
        </div>"""
    st.markdown(html,unsafe_allow_html=True)

def entonnoir_new(info, ds_name):
    clr = info.get('color','#3b82f6')
    labels=['Brut charge','Apres nettoyage','Apres union pH','Apres coherence sequence','Apres deduplication','Dataset final (3 classes)']
    vals=[info['brut'],info['nettoyage'],info['union'],info['coherence'],info['dedup'],info['final']]
    html=""
    base = info['brut']
    for lbl,val in zip(labels,vals):
        pct=f" ({100*val/base:.1f}%)" if val and base else ""
        w=max(4,int(100*val/base)) if base else 4
        html+=f"""<div style='display:flex;align-items:center;gap:12px;padding:8px 14px;
            background:#fff;border-radius:8px;border:1px solid #e2e8f0;margin-bottom:5px;'>
            <span style='font-family:Space Mono,monospace;font-weight:700;font-size:.9rem;
                color:#1e293b;min-width:80px;text-align:right;'>{fmt(val)}</span>
            <div style='flex:1;background:#f1f5f9;border-radius:4px;height:8px;'>
                <div style='width:{w}%;background:{clr};height:8px;border-radius:4px;'></div>
            </div>
            <span style='font-size:.78rem;color:#64748b;min-width:260px;'>{lbl}{pct}</span>
        </div>"""
    st.markdown(html,unsafe_allow_html=True)

def bar_source_train(par_source, title="Contribution par source"):
    palette=['#2563eb','#0ea5e9','#a855f7','#8b5cf6','#d946ef','#f59e0b','#10b981','#6366f1']
    labels=list(par_source.keys()); vals=list(par_source.values())
    fig=go.Figure(go.Bar(x=labels,y=vals,marker_color=palette[:len(labels)],
        text=[fmt(v) for v in vals],textposition='outside'))
    fig.update_layout(title=title,plot_bgcolor='white',paper_bgcolor='white',
        font=dict(family='DM Sans',size=11),margin=dict(l=30,r=20,t=40,b=40),height=280,
        yaxis=dict(showgrid=True,gridcolor='#f1f5f9'),xaxis=dict(showgrid=False))
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:14px 0 6px;'>
        <div style='font-family:Space Mono,monospace;font-size:.66rem;color:#3b82f6;
            letter-spacing:.12em;margin-bottom:4px;'>NAVIGATION</div>
        <div style='font-family:Space Mono,monospace;font-size:.9rem;color:#1e293b;
            font-weight:700;line-height:1.3;'>Stabilite<br>Proteique</div>
    </div>
    <hr style='border-color:#e2e8f0;margin:10px 0;'>
    """, unsafe_allow_html=True)

    page = st.radio("Menu", [
        "🏠  Accueil",
        "🗄️  Bases de données & prétraitement",
        "⚗️  Enrichissement du dataset & logique pH",
        "🔗  Combinaison des datasets d'entraînement",
        "🧪  Combinaison des datasets de test",
        "💬  Discussion",
        "🔭  Perspectives",
    ], label_visibility="collapsed")

    st.markdown("""
    <hr style='border-color:#e2e8f0;margin:14px 0;'>
    <div style='font-size:.69rem;color:#64748b;line-height:1.6;'>
        Pipeline de preparation des donnees<br>pour la prediction de la stabilite<br>proteique par mutation ponctuelle.
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# PAGE 1 — ACCUEIL
# ════════════════════════════════════════════════════════════════
if page == "🏠  Accueil":
    if banner_b64:
        st.markdown(f"""
        <div class="hero-wrap">
            <img class="hero-img" src="data:image/png;base64,{banner_b64}"/>
            <div class="hero-overlay">
                <div class="hero-badge">STAGE PFE — PIPELINE DONNEES</div>
                <div class="hero-title">Prediction de l'effet des mutations<br>sur la stabilite proteique</div>
                <div class="hero-sub">Pipeline de collecte, nettoyage, enrichissement et preparation des bases de donnees mutationnelles pour la classification DDG.</div>
            </div>
        </div>""", unsafe_allow_html=True)

    mcols([
        ("Datasets traites", "8", "#1e293b"),
        ("Mutations brutes", "6,228,864", "#1e293b"),
        ("Mutations finales", "845,380", "#3b82f6"),
        ("Sources principales", "4", "#1e293b"),
    ])

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec">Etapes du pipeline realisees</div>', unsafe_allow_html=True)

    steps_g = [
        ("📥", "Chargement", "CSV / JSON bruts"),
        ("🧹", "Nettoyage", "Mutations simples, DDG valides"),
        ("⚡", "Logique pH", "Features charge + pH-insensibles"),
        ("🔗", "Union", "Standards ∪ pH-insensibles"),
        ("🧬", "Sequences WT", "UniProt, coherence +/-30"),
        ("🗑️", "Deduplication", "Hash MD5 + mediane DDG"),
        ("🏷️", "Classes DDG", "3 classes incl. neutres"),
        ("📦", "Export", "CSV finaux + combinaisons"),
    ]
    cols = st.columns(8)
    for col, (icon, title, desc) in zip(cols, steps_g):
        with col:
            st.markdown(f"""
            <div style='background:#fff;border-radius:10px;padding:13px 8px;
                border:1px solid #e2e8f0;text-align:center;height:125px;
                display:flex;flex-direction:column;align-items:center;justify-content:center;'>
                <div style='font-size:1.45rem;margin-bottom:5px;'>{icon}</div>
                <div style='font-family:Space Mono,monospace;font-size:.68rem;font-weight:700;
                    color:#1e293b;margin-bottom:3px;white-space:pre-line;'>{title}</div>
                <div style='font-size:.67rem;color:#64748b;line-height:1.3;'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec">Bases de donnees utilisees</div>', unsafe_allow_html=True)

    db_infos = [
        ("🟣 ProTherm et ses derives",
         "5 datasets derives : S2648, S9028, STRUM_Q306, Broom_S605, PON-TStab",
         "https://web.iitm.ac.in/bioinfo2/prothermdb/"),
        ("🟠 FireProtDB",
         "Base de donnees de stabilite thermique des proteines — plus de 5,4M entrees",
         "https://loschmidt.chemi.muni.cz/fireprotdb/"),
        ("🟢 ThermoMutDB",
         "Dataset JSON avec features precalculees : BLOSUM62, SST, RSA",
         "https://biosig.lab.uq.edu.au/thermomutdb/"),
        ("🔵 Megadataset",
         "Megadataset symetrique nettoye (mutations directes et reverse), utilise comme base principale d'entrainement, contient 735,716 lignes",
         None),
    ]
    for icon_name, desc, link in db_infos:
        link_html = f'<a class="db-link" href="{link}" target="_blank">🔗 {link}</a>' if link else '<span style="font-size:.78rem;color:#94a3b8;">— Dataset interne</span>'
        st.markdown(f"""
        <div class="db-box">
            <div class="db-title">{icon_name}</div>
            <div style='font-size:.8rem;color:#475569;margin:5px 0;'>{desc}</div>
            {link_html}
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# PAGE 2 — BASES DE DONNÉES & PRÉTRAITEMENT (PIPELINE INITIAL)
# ════════════════════════════════════════════════════════════════
elif page == "🗄️  Bases de données & prétraitement":
    st.markdown('<div class="sec">Bases de donnees et pretraitement</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info">
    Cette page presente le <b>pipeline initial de reference (baseline)</b>. Elle est conservee telle quelle
    afin de servir de point de comparaison avec le nouveau pipeline enrichi (page ⚗️).
    Le pipeline baseline applique un filtrage strict : pH dans [6,8], T dans [25,37] C, suppression des mutations neutres.
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🟣 ProTherm & derives", "🟠 FireProtDB", "🟢 ThermoMutDB"])

    with tab1:
        st.markdown("""<div class="info"><b>ProTherm</b> regroupe des mesures experimentales de stabilite
        pour des mutations ponctuelles. Cinq datasets derives ont ete traites individuellement.</div>""", unsafe_allow_html=True)
        st.markdown('<div class="sec">Vue generale — entonnoirs de reduction</div>', unsafe_allow_html=True)
        names_pt = ["S2648","S9028","STRUM_Q306","Broom_S605","PON-TStab"]
        stages_labels = ["Brut","Nettoyage","Filtrage pH/T","Coherence","Dedup","Final"]
        stages_keys = ["brut","apres_nettoyage","s4","apres_coherence","apres_dedup","final"]
        colors_lines = [DATASETS[n]['color'] for n in names_pt]
        fig_ent = go.Figure()
        for i, n in enumerate(names_pt):
            info = DATASETS[n]
            vals = [info[k] for k in stages_keys]
            fig_ent.add_trace(go.Scatter(x=stages_labels, y=vals, mode='lines+markers', name=n,
                line=dict(color=colors_lines[i], width=2), marker=dict(size=6)))
        fig_ent.update_layout(title="Entonnoir de reduction — datasets ProTherm",
            plot_bgcolor='white', paper_bgcolor='white', font=dict(family='DM Sans', size=11),
            margin=dict(l=40, r=20, t=50, b=40), height=340,
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#f1f5f9', title='Lignes'))
        st.plotly_chart(fig_ent, use_container_width=True)
        st.markdown("---")
        ds_sel = st.selectbox("Selectionner un dataset :", names_pt)
        info = DATASETS[ds_sel]
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown(f'<div class="sec">Etapes realisees — {ds_sel}</div>', unsafe_allow_html=True)
            render_steps(STEPS_PROTHERM)
        with col_r:
            st.markdown('<div class="sec">Etapes a venir</div>', unsafe_allow_html=True)
            render_todo(STEPS_TODO)
        st.markdown("---")
        st.markdown(f'<div class="sec">Visualisations generales — {ds_sel}</div>', unsafe_allow_html=True)
        mcols([
            ("Lignes brutes", fmt(info['brut']), "#1e293b"),
            ("Dataset final", fmt(info['final']), "#3b82f6"),
            ("Doublons supprimes", fmt(info['doublons']), "#ef4444"),
            ("Coherence sequence", f"{info['couverture']}%", "#22c55e"),
        ])
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec">Entonnoir de reduction</div>', unsafe_allow_html=True)
        entonnoir(info, ds_sel)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec">Visualisations par etape</div>', unsafe_allow_html=True)
        etapes = ["Brut charge","Apres nettoyage","Apres filtrage pH/T","Coherence sequence",
                  "Deduplication","Classes DDG (avant suppression neutres)","Dataset final"]
        etape_sel = st.selectbox("Etape a visualiser :", etapes, key="etape_pt")
        if etape_sel == "Brut charge":
            render_etape_brut(ds_sel, info)
        elif etape_sel == "Apres nettoyage":
            render_etape_nettoyage(ds_sel, info)
        elif etape_sel == "Apres filtrage pH/T":
            st.plotly_chart(bar_scenarios(info, ds_sel), use_container_width=True)
            st.markdown("""<div class="info">Le scenario retenu (pH dans [6,8] et T dans [25,37]) conserve
            le meilleur compromis entre homogeneite experimentale et volume de donnees.</div>""", unsafe_allow_html=True)
        elif etape_sel == "Classes DDG (avant suppression neutres)":
            col1, col2 = st.columns(2)
            with col1: st.plotly_chart(bar_classes_ddg(info, ds_sel), use_container_width=True)
            with col2:
                fig_pie3 = go.Figure(go.Pie(labels=['Stabilisant','Neutre','Destabilisant'],
                    values=[info['avant_neutres_stab'],info['avant_neutres_neutre'],info['avant_neutres_destab']],
                    marker_colors=['#22c55e','#94a3b8','#ef4444'], hole=0.5, textinfo='percent+label', textfont_size=11))
                fig_pie3.update_layout(showlegend=False, margin=dict(l=0,r=0,t=30,b=0),
                    height=250, paper_bgcolor='white', font=dict(family='DM Sans'), title="Repartition 3 classes")
                st.plotly_chart(fig_pie3, use_container_width=True)
        elif etape_sel == "Coherence sequence":
            st.plotly_chart(bar_coherence_old(info, ds_sel), use_container_width=True)
            couv = info['couverture']
            suppr = info['supprimes_coherence']
            st.markdown(f'<div class="info">Taux de couverture sequence : <b>{couv}%</b>. {suppr} mutations supprimees par incoherence.</div>', unsafe_allow_html=True)
        elif etape_sel == "Deduplication":
            st.plotly_chart(bar_dedup_old(info, ds_sel), use_container_width=True)
        elif etape_sel == "Dataset final":
            col1, col2 = st.columns([3,2])
            with col1: st.plotly_chart(ddg_hist_sim(ds_sel, info, "(final)"), use_container_width=True)
            with col2: st.plotly_chart(pie_sd(info['stab'], info['destab']), use_container_width=True)
            mcols([("Stabilisant",fmt(info['stab']),"#22c55e"),("Destabilisant",fmt(info['destab']),"#ef4444"),
                   ("Moyenne DDG",f"{info['ddg_mean']:.3f}","#1e293b"),("Mediane DDG",f"{info['ddg_median']:.3f}","#1e293b")])
            if ds_sel == "S9028" and 'forward_final' in info:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="sec">Repartition Forward / Reverse</div>', unsafe_allow_html=True)
                col_fw, _ = st.columns([2,3])
                with col_fw:
                    fig_fw = go.Figure(go.Pie(labels=['Forward','Reverse'],
                        values=[info['forward_final'],info['reverse_final']],
                        marker_colors=['#0ea5e9','#7dd3fc'], hole=0.5, textinfo='percent+label', textfont_size=12))
                    fig_fw.update_layout(title="Forward vs Reverse — S9028 final",
                        showlegend=False, margin=dict(l=0,r=0,t=40,b=0), height=260,
                        paper_bgcolor='white', font=dict(family='DM Sans'))
                    st.plotly_chart(fig_fw, use_container_width=True)
            if ds_sel == "PON-TStab" and 'chaine_A' in info:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="sec">Chaines resolues via SIFTS</div>', unsafe_allow_html=True)
                col_ch, _ = st.columns([2,3])
                with col_ch:
                    fig_ch = go.Figure(go.Bar(x=['Chaine A','Chaine I'], y=[info['chaine_A'],info['chaine_I']],
                        marker_color=['#d946ef','#e879f9'], text=[info['chaine_A'],info['chaine_I']], textposition='outside'))
                    fig_ch.update_layout(title="Chaines resolues — PON-TStab", plot_bgcolor='white', paper_bgcolor='white',
                        font=dict(family='DM Sans',size=11), margin=dict(l=40,r=20,t=50,b=40), height=270,
                        xaxis=dict(showgrid=False), yaxis=dict(showgrid=True,gridcolor='#f1f5f9',title='Lignes'))
                    st.plotly_chart(fig_ch, use_container_width=True)
                avec = fmt(info['avec_pdb'])
                sans = fmt(info['sans_pdb'])
                st.markdown(f'<div class="info">77 lignes corrigees via SIFTS. Avec PDB : <b>{avec}</b> — Sans PDB : <b>{sans}</b></div>', unsafe_allow_html=True)

    with tab2:
        st.markdown("""<div class="info"><b>FireProtDB</b> — plus de 5,4M entrees dont ~412k avec DDG valide.
         Apres filtrage et nettoyage (baseline) : <b>1 184 mutations finales</b>.</div>""", unsafe_allow_html=True)
        info = DATASETS["FireProtDB"]
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown('<div class="sec">Etapes realisees</div>', unsafe_allow_html=True)
            render_steps(STEPS_FIREPROT)
        with col_r:
            st.markdown('<div class="sec">Etapes a venir</div>', unsafe_allow_html=True)
            render_todo(STEPS_TODO)
        st.markdown("---")
        mcols([("Lignes brutes",fmt(info['brut']),"#1e293b"),("Dataset final",fmt(info['final']),"#3b82f6"),
               ("Doublons supprimes",fmt(info['doublons']),"#ef4444"),("Coherence sequence",f"{info['couverture']}%","#22c55e")])
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec">Entonnoir de reduction</div>', unsafe_allow_html=True)
        entonnoir(info, "FireProtDB")
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec">Visualisations par etape</div>', unsafe_allow_html=True)
        etape_fp = st.selectbox("Etape :", ["Brut charge","Apres nettoyage","Apres filtrage pH/T",
            "Coherence sequence","Deduplication","Classes DDG","Dataset final"], key="etape_fp")
        if etape_fp == "Brut charge": render_fireprot_brut(info)
        elif etape_fp == "Apres nettoyage": render_fireprot_nettoyage(info)
        elif etape_fp == "Apres filtrage pH/T": st.plotly_chart(bar_scenarios(info,"FireProtDB"),use_container_width=True)
        elif etape_fp == "Classes DDG":
            col1,col2=st.columns(2)
            with col1: st.plotly_chart(bar_classes_ddg(info,"FireProtDB"),use_container_width=True)
            with col2:
                fig_p3=go.Figure(go.Pie(labels=['Stabilisant','Neutre','Destabilisant'],
                    values=[info['avant_neutres_stab'],info['avant_neutres_neutre'],info['avant_neutres_destab']],
                    marker_colors=['#22c55e','#94a3b8','#ef4444'],hole=0.5,textinfo='percent+label',textfont_size=11))
                fig_p3.update_layout(showlegend=False,margin=dict(l=0,r=0,t=30,b=0),height=250,paper_bgcolor='white',font=dict(family='DM Sans'))
                st.plotly_chart(fig_p3,use_container_width=True)
        elif etape_fp == "Coherence sequence": st.plotly_chart(bar_coherence_old(info,"FireProtDB"),use_container_width=True)
        elif etape_fp == "Deduplication": st.plotly_chart(bar_dedup_old(info,"FireProtDB"),use_container_width=True)
        elif etape_fp == "Dataset final":
            col1,col2=st.columns([3,2])
            with col1: st.plotly_chart(ddg_hist_sim("FireProtDB",info,"(final)"),use_container_width=True)
            with col2: st.plotly_chart(pie_sd(info['stab'],info['destab']),use_container_width=True)
            mcols([("Stabilisant",fmt(info['stab']),"#22c55e"),("Destabilisant",fmt(info['destab']),"#ef4444"),
                   ("Moyenne DDG",f"{info['ddg_mean']:.3f}","#1e293b"),("Mediane DDG",f"{info['ddg_median']:.3f}","#1e293b")])

    with tab3:
        st.markdown("""<div class="info"><b>ThermoMutDB</b> — dataset JSON avec features precalculees
        (BLOSUM62, SST, RSA). Colonne uniprot native : recuperation directe des sequences.</div>""", unsafe_allow_html=True)
        info = DATASETS["ThermoMutDB"]
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown('<div class="sec">Etapes realisees</div>', unsafe_allow_html=True)
            render_steps(STEPS_THERMO)
        with col_r:
            st.markdown('<div class="sec">Etapes a venir</div>', unsafe_allow_html=True)
            render_todo(STEPS_TODO)
        st.markdown("---")
        mcols([("Lignes brutes",fmt(info['brut']),"#1e293b"),("Dataset final",fmt(info['final']),"#3b82f6"),
               ("Doublons supprimes",fmt(info['doublons']),"#ef4444"),("Coherence sequence",f"{info['couverture']}%","#22c55e")])
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec">Entonnoir de reduction</div>', unsafe_allow_html=True)
        entonnoir(info,"ThermoMutDB")
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec">Visualisations par etape</div>', unsafe_allow_html=True)
        etape_th = st.selectbox("Etape :",["Brut charge","Apres nettoyage","Apres filtrage pH/T",
            "Coherence sequence","Deduplication","Classes DDG","Dataset final"],key="etape_th")
        if etape_th == "Brut charge": render_thermo_brut(info)
        elif etape_th == "Apres nettoyage": render_thermo_nettoyage(info)
        elif etape_th == "Apres filtrage pH/T":
            col1,col2=st.columns(2)
            with col1: st.plotly_chart(bar_scenarios(info,"ThermoMutDB"),use_container_width=True)
            with col2:
                np.random.seed(42)
                temps=np.clip(np.concatenate([np.random.normal(25,3,500),np.random.normal(37,4,300)]),15,45)
                fig_t=go.Figure(go.Histogram(x=temps,nbinsx=30,marker_color='#10b981',marker_line_color='white',marker_line_width=0.5))
                fig_t.add_vline(x=25,line_dash="dash",line_color="#3b82f6",line_width=1.5)
                fig_t.add_vline(x=37,line_dash="dash",line_color="#3b82f6",line_width=1.5)
                fig_t.update_layout(title="Distribution temperature — ThermoMutDB",xaxis_title="T (C)",
                    plot_bgcolor='white',paper_bgcolor='white',font=dict(family='DM Sans',size=11),
                    margin=dict(l=40,r=20,t=40,b=40),height=280)
                st.plotly_chart(fig_t,use_container_width=True)
        elif etape_th == "Classes DDG":
            col1,col2=st.columns(2)
            with col1: st.plotly_chart(bar_classes_ddg(info,"ThermoMutDB"),use_container_width=True)
            with col2:
                fig_p3=go.Figure(go.Pie(labels=['Stabilisant','Neutre','Destabilisant'],
                    values=[info['avant_neutres_stab'],info['avant_neutres_neutre'],info['avant_neutres_destab']],
                    marker_colors=['#22c55e','#94a3b8','#ef4444'],hole=0.5,textinfo='percent+label',textfont_size=11))
                fig_p3.update_layout(showlegend=False,margin=dict(l=0,r=0,t=30,b=0),height=250,paper_bgcolor='white',font=dict(family='DM Sans'))
                st.plotly_chart(fig_p3,use_container_width=True)
        elif etape_th == "Coherence sequence": st.plotly_chart(bar_coherence_old(info,"ThermoMutDB"),use_container_width=True)
        elif etape_th == "Deduplication": st.plotly_chart(bar_dedup_old(info,"ThermoMutDB"),use_container_width=True)
        elif etape_th == "Dataset final":
            col1,col2=st.columns([3,2])
            with col1: st.plotly_chart(ddg_hist_sim("ThermoMutDB",info,"(final)"),use_container_width=True)
            with col2: st.plotly_chart(pie_sd(info['stab'],info['destab']),use_container_width=True)
            mcols([("Stabilisant",fmt(info['stab']),"#22c55e"),("Destabilisant",fmt(info['destab']),"#ef4444"),
                   ("Moyenne DDG",f"{info['ddg_mean']:.3f}","#1e293b"),("Mediane DDG",f"{info['ddg_median']:.3f}","#1e293b")])

# ════════════════════════════════════════════════════════════════
# PAGE 3 — ENRICHISSEMENT & LOGIQUE pH
# ════════════════════════════════════════════════════════════════
elif page == "⚗️  Enrichissement du dataset & logique pH":
    st.markdown('<div class="sec">Enrichissement du dataset et logique pH — Nouveau pipeline</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info">
    <b>Principe cle :</b> Les mutations neutral_to_neutral (sans changement de charge) sont
    <b>insensibles au pH</b>. Elles peuvent donc etre exploitees quelle que soit la condition experimentale de pH.
    La <b>UNION</b> des conditions standards et des mutations pH-insensibles permet de multiplier significativement
    le volume de donnees sans introduire de biais biochimique.
    </div>""", unsafe_allow_html=True)

    ds_main = st.selectbox("Dataset principal :", ["ProTherm derives", "FireProtDB", "ThermoMutDB"], key="p3_main")
    if ds_main == "ProTherm derives":
        ds_sub = st.selectbox("Sous-dataset :", ["S2648", "S9028"], key="p3_sub")
        ds_key = ds_sub
    elif ds_main == "FireProtDB":
        ds_key = "FireProtDB"
    else:
        ds_key = "ThermoMutDB"

    info_d = NEW_PIPELINE[ds_key]

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown(f'<div class="sec">Etapes realisees — {ds_key}</div>', unsafe_allow_html=True)
        render_steps(STEPS_NEW_PIPELINE)
    with col_r:
        st.markdown('<div class="sec">Etapes a venir</div>', unsafe_allow_html=True)
        render_todo(STEPS_TODO)

    st.markdown("---")
    mcols([
        ("Brut charge", fmt(info_d['brut']), "#1e293b"),
        ("Apres union pH", fmt(info_d['union']), "#6366f1"),
        ("Dataset final", fmt(info_d['final']), "#22c55e"),
        ("Conservation vs brut", f"{info_d['conservation_pct']}%", "#f59e0b"),
    ])
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="sec">Entonnoir de reduction — nouveau pipeline</div>', unsafe_allow_html=True)
    entonnoir_new(info_d, ds_key)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="sec">Visualisations par etape</div>', unsafe_allow_html=True)
    # FIX 1 — "Classes DDG (3 classes)" renomme en "Classes DDG apres union"
    etape_sel = st.selectbox("Etape a visualiser :", [
        "Distribution des types de charge",
        "S4 vs pH-insensibles vs UNION",
        "Classes DDG apres union",
        "Coherence sequence",
        "Deduplication",
        "Dataset final",
    ], key="p3_etape")

    if etape_sel == "Distribution des types de charge":
        col1, col2 = st.columns(2)
        with col1:
            fig_ch = go.Figure(go.Bar(
                x=["neutral_to_neutral\n(pH-insensible)", "charge_effect_potential\n(pH-sensible)"],
                y=[info_d['neutral_to_neutral'], info_d['charge_effect']],
                marker_color=["#6366f1", "#f59e0b"],
                text=[fmt(info_d['neutral_to_neutral']), fmt(info_d['charge_effect'])],
                textposition='outside'))
            fig_ch.update_layout(title=f"Types de charge — {ds_key} (apres nettoyage)",
                plot_bgcolor='white', paper_bgcolor='white', font=dict(family='DM Sans', size=11),
                margin=dict(l=40, r=20, t=50, b=40), height=290,
                xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#f1f5f9', title='Mutations'))
            st.plotly_chart(fig_ch, use_container_width=True)
        with col2:
            pct = 100 * info_d['neutral_to_neutral'] / info_d['nettoyage']
            fig_pie_ch = go.Figure(go.Pie(
                labels=["neutral_to_neutral", "charge_effect_potential"],
                values=[info_d['neutral_to_neutral'], info_d['charge_effect']],
                marker_colors=["#6366f1", "#f59e0b"],
                hole=0.5, textinfo='percent+label', textfont_size=11))
            fig_pie_ch.update_layout(title="Repartition types de charge", showlegend=False,
                margin=dict(l=0,r=0,t=35,b=0), height=270, paper_bgcolor='white', font=dict(family='DM Sans'))
            st.plotly_chart(fig_pie_ch, use_container_width=True)
        pct_fmt = f"{pct:.1f}"
        st.markdown(f'<div class="info"><b>{pct_fmt}%</b> des mutations sont pH-insensibles (neutral_to_neutral) — exploitables sans condition de pH.</div>', unsafe_allow_html=True)

    elif etape_sel == "S4 vs pH-insensibles vs UNION":
        col1, col2 = st.columns(2)
        with col1:
            fig_union = go.Figure(go.Bar(
                x=["Conditions standards\n(pH [6-8] + T [20-37])", "Mutations pH-insensibles\n(neutral_to_neutral)", "UNION"],
                y=[info_d['standards'], info_d['ph_insensible'], info_d['union']],
                marker_color=["#3b82f6", "#6366f1", "#22c55e"],
                text=[fmt(info_d['standards']), fmt(info_d['ph_insensible']), fmt(info_d['union'])],
                textposition='outside'))
            fig_union.update_layout(title=f"Composition de l'UNION — {ds_key}",
                plot_bgcolor='white', paper_bgcolor='white', font=dict(family='DM Sans', size=11),
                margin=dict(l=40, r=20, t=50, b=40), height=300,
                xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#f1f5f9', title='Mutations'))
            st.plotly_chart(fig_union, use_container_width=True)
        with col2:
            ph_insens_only = max(info_d['union'] - info_d['standards'], 0)
            fig_src = go.Figure(go.Pie(
                labels=["Standards (pH [6,8], T [20,37])", "pH-insensibles uniquement"],
                values=[info_d['standards'], ph_insens_only],
                marker_colors=["#3b82f6", "#6366f1"],
                hole=0.5, textinfo='percent+label', textfont_size=11))
            fig_src.update_layout(title="Source dans l'UNION", showlegend=False,
                margin=dict(l=0,r=0,t=35,b=0), height=280,
                paper_bgcolor='white', font=dict(family='DM Sans'))
            st.plotly_chart(fig_src, use_container_width=True)
        old_s4 = DATASETS.get(ds_key, {}).get('s4', 0)
        gain = info_d['union'] - old_s4
        st.markdown(f'<div class="info"><b>Gain vs filtrage strict (baseline) :</b> {fmt(old_s4)} mutations (S4 baseline) → <b>{fmt(info_d["union"])} apres union</b> (+{fmt(gain)} mutations recuperees).</div>', unsafe_allow_html=True)

    # FIX 1 — bloc classes DDG apres union : utilise union_stab / union_neutre / union_destab
    elif etape_sel == "Classes DDG apres union":
        col1, col2 = st.columns(2)
        with col1:
            u_stab = info_d['union_stab']
            u_neutre = info_d['union_neutre']
            u_destab = info_d['union_destab']
            u_stab_pct = info_d['union_stab_pct']
            u_neutre_pct = info_d['union_neutre_pct']
            u_destab_pct = info_d['union_destab_pct']
            fig_cls = go.Figure(go.Bar(
                x=["Stabilisant clair\n(DDG < -1)", "Zone neutre\n(|DDG| <= 1)", "Destabilisant clair\n(DDG > +1)"],
                y=[u_stab, u_neutre, u_destab],
                marker_color=["#22c55e", "#94a3b8", "#ef4444"],
                text=[f"{fmt(u_stab)} ({u_stab_pct}%)", f"{fmt(u_neutre)} ({u_neutre_pct}%)", f"{fmt(u_destab)} ({u_destab_pct}%)"],
                textposition='outside'))
            fig_cls.update_layout(title=f"3 classes DDG apres UNION — {ds_key}",
                plot_bgcolor='white', paper_bgcolor='white', font=dict(family='DM Sans', size=11),
                margin=dict(l=40, r=20, t=50, b=40), height=290,
                xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#f1f5f9', title='Mutations'))
            st.plotly_chart(fig_cls, use_container_width=True)
        with col2:
            st.plotly_chart(pie_3cls(info_d['union_stab'], info_d['union_neutre'], info_d['union_destab'],
                f"3 classes apres union — {ds_key}", 270), use_container_width=True)
        n_val = fmt(info_d['union_neutre'])
        n_pct = info_d['union_neutre_pct']
        st.markdown(f'<div class="info">L\'enrichissement vient d\'une meilleure exploitation des donnees existantes, pas d\'une augmentation artificielle.<br>Zone neutre apres union : <b>{n_val}</b> mutations ({n_pct}%) — ces valeurs seront reduites apres coherence et deduplication.</div>', unsafe_allow_html=True)

    elif etape_sel == "Coherence sequence":
        new_coh_sup = info_d['union'] - info_d['coherence']
        col1, col2 = st.columns(2)
        with col1:
            fig_coh = go.Figure(go.Bar(
                x=['Avant coherence', 'Apres coherence', 'Supprimes'],
                y=[info_d['union'], info_d['coherence'], new_coh_sup],
                marker_color=['#3b82f6', '#22c55e', '#ef4444'],
                text=[fmt(info_d['union']), fmt(info_d['coherence']), fmt(new_coh_sup)],
                textposition='outside'))
            fig_coh.update_layout(title=f"Coherence sequence — {ds_key}",
                plot_bgcolor='white', paper_bgcolor='white', font=dict(family='DM Sans', size=11),
                margin=dict(l=40, r=20, t=50, b=40), height=280,
                xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#f1f5f9'))
            st.plotly_chart(fig_coh, use_container_width=True)
        with col2:
            mcols([("Couverture sequences WT", f"{info_d['couverture_seq']}%", "#22c55e")])
            st.markdown("<br>", unsafe_allow_html=True)
            mcols([("Mutations supprimees", fmt(new_coh_sup), "#ef4444")])

    elif etape_sel == "Deduplication":
        new_doublons = info_d['coherence'] - info_d['dedup']
        fig_dd = go.Figure(go.Bar(
            x=['Avant deduplication', 'Apres deduplication', 'Doublons supprimes'],
            y=[info_d['coherence'], info_d['dedup'], new_doublons],
            marker_color=['#3b82f6', '#22c55e', '#f87171'],
            text=[fmt(info_d['coherence']), fmt(info_d['dedup']), fmt(new_doublons)],
            textposition='outside'))
        fig_dd.update_layout(title=f"Deduplication — {ds_key}",
            plot_bgcolor='white', paper_bgcolor='white', font=dict(family='DM Sans', size=11),
            margin=dict(l=40, r=20, t=50, b=40), height=280,
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#f1f5f9', title='Lignes'))
        st.plotly_chart(fig_dd, use_container_width=True)

    elif etape_sel == "Dataset final":
        col1, col2 = st.columns([3,2])
        with col1:
            st.plotly_chart(ddg_hist_3cls(ds_key, info_d, height=280), use_container_width=True)
        with col2:
            # FIX 1 — Dataset final utilise stab / neutre / destab (pas union_*)
            st.plotly_chart(pie_3cls(info_d['stab'], info_d['neutre'], info_d['destab'],
                f"Classes finales — {ds_key}", 270), use_container_width=True)
        mcols([
            ("Stabilisant", f"{fmt(info_d['stab'])} ({info_d['stab_pct']}%)", "#22c55e"),
            ("Zone neutre", f"{fmt(info_d['neutre'])} ({info_d['neutre_pct']}%)", "#64748b"),
            ("Destabilisant", f"{fmt(info_d['destab'])} ({info_d['destab_pct']}%)", "#ef4444"),
            ("DDG moyen", f"{info_d['ddg_mean']:.3f}", "#1e293b"),
        ])

# ════════════════════════════════════════════════════════════════
# PAGE 4 — COMBINAISON DATASETS D'ENTRAÎNEMENT
# ════════════════════════════════════════════════════════════════
elif page == "🔗  Combinaison des datasets d'entraînement":
    st.markdown('<div class="sec">Combinaison des datasets d\'entrainement</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info">Fusion des datasets finaux (nouveau pipeline) avec deduplication inter-datasets.
    Les neutres sont conserves. La mediane DDG est retenue en cas de conflit.</div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec">Datasets de base</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    bases = [
        ("🟣 ProTherm derives", "S2648 (1,441) + S9028 (3,596) + broom(360) + strum(161) + Ponstab(935) =6,493", "#6366f1"),
        ("🟢 ThermoMutDB", "2,872 mutations finales", "#10b981"),
        ("🟠 FireProtDB", "49,446 mutations finales", "#f59e0b"),
        ("🔵 Megadataset", "Megadataset symetrique — base d'entrainement principale et contient 735,716 lignes", "#3b82f6"),
    ]
    for col, (title, sub, clr) in zip([c1,c2,c3,c4], bases):
        with col:
            st.markdown(f"""<div style='background:#fff;border-radius:10px;padding:16px 18px;
                border-left:4px solid {clr};border:1px solid #e2e8f0;text-align:center;'>
                <div style='font-weight:700;font-size:.9rem;color:#1e293b;'>{title}</div>
                <div style='font-size:.78rem;color:#64748b;margin-top:4px;'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec">Selectionner une combinaison</div>', unsafe_allow_html=True)
    combo_sel = st.selectbox("", list(COMBOS_TRAIN.keys()))
    c = COMBOS_TRAIN[combo_sel]
    is_final = c.get("is_final", False)

    if is_final:
        st.markdown("""<div style='background:#f0fdf4;border:1.5px solid #22c55e;border-radius:8px;
            padding:10px 16px;font-size:.83rem;color:#15803d;margin-bottom:12px;'>
            ⭐ <b>Combinaison FINALE</b> — Dataset complet utilise pour l'entrainement du modele.
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    mcols([
        ("Avant deduplication", fmt(c['avant_dedup']), "#1e293b"),
        ("Doublons supprimes", fmt(c['doublons']), "#ef4444"),
        ("Total final", fmt(c['apres_dedup']), "#3b82f6"),
        ("Stabilisant", f"{fmt(c['stab'])} ({c['stab_pct']}%)", "#22c55e"),
        ("Zone neutre", f"{fmt(c['neutre'])} ({c['neutre_pct']}%)", "#64748b"),
        ("Destabilisant", f"{fmt(c['destab'])} ({c['destab_pct']}%)", "#ef4444"),
    ])
    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns([3,2])
    with col_a:
        st.plotly_chart(bar_source_train(c['par_source'], f"Contribution par source — {combo_sel[:40]}"), use_container_width=True)
    with col_b:
        st.plotly_chart(pie_3cls(c['stab'], c['neutre'], c['destab'], "Repartition 3 classes", 260), use_container_width=True)

    np.random.seed(abs(hash(combo_sel))%2**31)
    v = np.concatenate([
        np.clip(np.random.normal(-2.1, 1.4, c['stab']), -10, -0.001),
        np.clip(np.random.normal(0, 0.45, c['neutre']), -0.99, 0.99),
        np.clip(np.random.normal(2.0, 1.3, c['destab']), 0.001, 10),
    ])
    fig_ddg = go.Figure(go.Histogram(x=v, nbinsx=60, marker_color='#3b82f6',
        marker_line_color='white', marker_line_width=0.5, opacity=0.85))
    fig_ddg.add_vline(x=-1, line_dash="dash", line_color="#22c55e", line_width=1.5)
    fig_ddg.add_vline(x=1, line_dash="dash", line_color="#ef4444", line_width=1.5)
    fig_ddg.add_vline(x=0, line_dash="dot", line_color="#94a3b8", line_width=1)
    fig_ddg.update_layout(title=f"Distribution DDG simulee — {combo_sel[:40]}",
        xaxis_title="DDG (kcal/mol)", yaxis_title="Frequence",
        plot_bgcolor='white', paper_bgcolor='white', font=dict(family='DM Sans', size=11),
        margin=dict(l=40, r=20, t=40, b=40), height=280, showlegend=False)
    fig_ddg.update_xaxes(showgrid=True, gridcolor='#f1f5f9')
    fig_ddg.update_yaxes(showgrid=True, gridcolor='#f1f5f9')
    st.plotly_chart(fig_ddg, use_container_width=True)

    if is_final:
        st.markdown("---")
        st.markdown('<div class="sec">Comparaison de toutes les combinaisons</div>', unsafe_allow_html=True)
        rows = []
        for name, ci in COMBOS_TRAIN.items():
            rows.append({
                "Combinaison": name,
                "Avant dedup": fmt(ci['avant_dedup']),
                "Doublons": fmt(ci['doublons']),
                "Total final": fmt(ci['apres_dedup']),
                "Stabilisant": f"{fmt(ci['stab'])} ({ci['stab_pct']}%)",
                "Zone neutre": f"{fmt(ci['neutre'])} ({ci['neutre_pct']}%)",
                "Destabilisant": f"{fmt(ci['destab'])} ({ci['destab_pct']}%)",
            })
        st.dataframe(pd.DataFrame(rows).set_index("Combinaison"), use_container_width=True)

# ════════════════════════════════════════════════════════════════
# PAGE 5 — COMBINAISON DATASETS DE TEST
# ════════════════════════════════════════════════════════════════
elif page == "🧪  Combinaison des datasets de test":
    st.markdown('<div class="sec">Combinaison des datasets de test</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info">Datasets de test : <b>ssym</b> (342), <b>ssym_r</b> (342),
    <b>s669</b> (669), <b>s669_r</b> (669), <b>p53</b> (42) — Total : <b>2,064 lignes</b>.<br>
    La symetrie thermodynamique DDG(A→B) = -DDG(B→A) est verifiee pour les paires directes/reverse.</div>""",
    unsafe_allow_html=True)

    col1,col2,col3,col4,col5 = st.columns(5)
    test_srcs = [("ssym","342 lignes","#2563eb"),("ssym_r","342 lignes","#0ea5e9"),
                 ("s669","669 lignes","#6366f1"),("s669_r","669 lignes","#8b5cf6"),("p53","42 lignes","#d946ef")]
    for col,(name,sub,clr) in zip([col1,col2,col3,col4,col5],test_srcs):
        with col:
            st.markdown(f"""<div style='background:#fff;border-radius:10px;padding:12px 10px;
                border-left:3px solid {clr};border:1px solid #e2e8f0;text-align:center;'>
                <div style='font-weight:700;font-size:.9rem;color:#1e293b;'>{name}</div>
                <div style='font-size:.75rem;color:#64748b;margin-top:3px;'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    test_sel = st.selectbox("Selectionner une combinaison :", list(COMBOS_TEST.keys()))
    tc = COMBOS_TEST[test_sel]

    mcols([
        ("Avant deduplication", fmt(tc['avant_dedup']), "#1e293b"),
        ("Doublons", fmt(tc['doublons']), "#ef4444"),
        ("Total final", fmt(tc['apres_dedup']), "#3b82f6"),
        ("Stabilisant", f"{fmt(tc['stab'])} ({tc['stab_pct']}%)", "#22c55e"),
        ("Zone neutre", f"{fmt(tc['neutre'])} ({tc['neutre_pct']}%)", "#64748b"),
        ("Destabilisant", f"{fmt(tc['destab'])} ({tc['destab_pct']}%)", "#ef4444"),
        ("Direct", fmt(tc['direct']), "#3b82f6"),
        ("Reverse", fmt(tc['reverse']), "#6366f1"),
    ])
    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns([3,2])
    with col_a:
        st.plotly_chart(bar_source_train(tc['sources'], f"Contribution par source — {test_sel}"), use_container_width=True)
    with col_b:
        st.plotly_chart(pie_3cls(tc['stab'], tc['neutre'], tc['destab'], "Repartition 3 classes", 260), use_container_width=True)

    if tc['sym_ok'] is not None:
        st.markdown("---")
        st.markdown('<div class="sec">Metriques de symetrie thermodynamique</div>', unsafe_allow_html=True)
        st.markdown("""<div class="info">
            Principe : <b>DDG(A→B) = -DDG(B→A)</b><br>
            Verification que les paires directes/reverse respectent l'antisymetrie thermodynamique.
        </div>""", unsafe_allow_html=True)
        sym_color = "#22c55e" if tc['sym_ok'] else "#ef4444"
        sym_label = "OUI" if tc['sym_ok'] else "NON"
        # FIX 2 — MAE et RMSE supprimes, on garde seulement Correlation, Paires verifiees, Symetrie OK
        mcols([
            ("Correlation", f"{tc['corr']:.6f}", "#3b82f6"),
            ("Paires verifiees", fmt(tc['direct']), "#6366f1"),
            ("Symetrie OK", sym_label, sym_color),
        ])
    else:
        st.markdown("""<div class="warn">
            Pas de metriques de symetrie pour C3 — directs seulement, pas de paires reverse.
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sec">Comparaison des 4 combinaisons de test</div>', unsafe_allow_html=True)
    rows_t = []
    for name, ti in COMBOS_TEST.items():
        sym_str = "OUI" if ti['sym_ok'] else ("NON" if ti['sym_ok'] is False else "—")
        rows_t.append({
            "Combinaison": name,
            "Lignes": fmt(ti['apres_dedup']),
            "Stabilisant": f"{fmt(ti['stab'])} ({ti['stab_pct']}%)",
            "Zone neutre": f"{fmt(ti['neutre'])} ({ti['neutre_pct']}%)",
            "Destabilisant": f"{fmt(ti['destab'])} ({ti['destab_pct']}%)",
            "Direct": fmt(ti['direct']),
            "Reverse": fmt(ti['reverse']),
            "Symetrie": sym_str,
        })
    st.dataframe(pd.DataFrame(rows_t).set_index("Combinaison"), use_container_width=True)

# ════════════════════════════════════════════════════════════════
# PAGE 6 — DISCUSSION
# ════════════════════════════════════════════════════════════════
elif page == "💬  Discussion":
    st.markdown('<div class="sec">Discussion</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec">Impact du filtrage des conditions experimentales</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info">
    Comparaison entre le pipeline baseline (filtrage strict) et le nouveau pipeline (logique pH + union).
    </div>""", unsafe_allow_html=True)

    datasets_cmp = ["S2648", "S9028", "FireProtDB", "ThermoMutDB"]
    old_vals = [466, 635, 1184, 1489]
    new_vals = [1441, 3596, 49446, 2872]
    colors_ds = ["#2563eb", "#0ea5e9", "#f59e0b", "#10b981"]

    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(name='Ancien pipeline (baseline)', x=datasets_cmp, y=old_vals,
        marker_color='#94a3b8', text=[fmt(v) for v in old_vals], textposition='outside'))
    fig_cmp.add_trace(go.Bar(name='Nouveau pipeline', x=datasets_cmp, y=new_vals,
        marker_color=colors_ds, text=[fmt(v) for v in new_vals], textposition='outside'))
    fig_cmp.update_layout(barmode='group', title="Mutations finales — ancien vs nouveau pipeline",
        plot_bgcolor='white', paper_bgcolor='white', font=dict(family='DM Sans', size=11),
        margin=dict(l=40, r=20, t=50, b=40), height=340,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#f1f5f9', title='Mutations finales'))
    st.plotly_chart(fig_cmp, use_container_width=True)

    ratios = [n/o for n,o in zip(new_vals, old_vals)]
    fig_gain = go.Figure(go.Bar(x=datasets_cmp, y=ratios, marker_color=colors_ds,
        text=[f"x{r:.1f}" for r in ratios], textposition='outside'))
    fig_gain.update_layout(title="Facteur d'augmentation (nouveau / ancien)",
        plot_bgcolor='white', paper_bgcolor='white', font=dict(family='DM Sans', size=11),
        margin=dict(l=40, r=20, t=50, b=40), height=290,
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#f1f5f9', title='Facteur x'))
    st.plotly_chart(fig_gain, use_container_width=True)

    st.markdown("---")

    discussions = [
        ("Limites du filtrage classique",
         "Le pipeline baseline filtre strictement sur pH dans [6,8] ET T dans [25,37]. "
         "Cette approche elimine une grande partie des donnees disponibles — FireProtDB perd plus de 99% "
         "car la majorite des experiences n'ont pas de pH ni de temperature renseignes."),
        ("Importance de la logique pH",
         "Les mutations neutral_to_neutral impliquent des acides amines sans charge electrique aux "
         "pH physiologiques. Leur stabilite est independante du pH : elles peuvent etre utilisees sans "
         "condition de pH sans biaiser le dataset. Cette distinction biochimique justifie l'union."),
        ("Importance de la conservation des mutations neutres",
         "La zone neutre (|DDG| <= 1 kcal/mol) represente ~49% des mutations dans les datasets finaux. "
         "Les supprimer cree un biais artificiel et reduit la capacite du modele a apprendre la frontiere "
         "de decision. Pour la regression, les neutres sont essentiels pour calibrer les predictions."),
        ("Benefice pour l'apprentissage automatique",
         "Le nouveau pipeline multiplie le volume de donnees par x3 a x42 selon le dataset. "
         "La conservation des 3 classes equilibre mieux l'apprentissage. La symetrie "
         "DDG(A→B) = -DDG(B→A) dans les datasets de test garantit une evaluation rigoureuse."),
    ]
    for title, text in discussions:
        st.markdown(f"""
        <div class="disc">
            <strong>{title}</strong><br>{text}
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# PAGE 7 — PERSPECTIVES
# ════════════════════════════════════════════════════════════════
elif page == "🔭  Perspectives":
    st.markdown('<div class="sec">Perspectives</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info">
    Cette section presente les principales etapes a venir du projet, depuis la finalisation du
    pipeline de preparation des donnees jusqu'a l'exploitation du modele pour l'analyse et la
    generation de profils mutationnels de stabilite proteique.
    </div>""", unsafe_allow_html=True)

    perspectives = [
        ("01","Finalisation du pipeline de preparation des donnees",
         "Reduction de la redondance, controle des biais, verification de l'absence de data leakage, "
         "construction des ensembles Train / Validation / Test par proteine, et preparation finale des "
         "entrees du modele dans des conditions rigoureuses et coherentes.", "#6366f1"),
        ("02","Developpement d'un modele de prediction de l'effet des mutations",
         "Entrainement d'un modele capable de predire l'effet d'une mutation sur la stabilite proteique. "
         "Deux formulations : regression (predire DDG directement) et classification (3 classes).", "#3b82f6"),
        ("03","Embeddings de sequences — modeles de langage proteique (ESM)",
         "Integration de representations issues de modeles ESM-2, ESM-1v pour encoder la sequence WT "
         "et la mutation. Remplacement des features manuelles par des representations vectorielles contextuelles.", "#0ea5e9"),
        ("04","Extension vers des bases de donnees therapeutiques",
         "Application du modele a des proteines d'interet clinique. Exploration du potentiel de "
         "generalisation et du transfert de connaissance vers des contextes biomedicaux.", "#10b981"),
        ("05","Generation de mutations et analyse de leur impact",
         "Utilisation du modele pour predire l'effet de nouvelles mutations candidates et explorer "
         "systematiquement l'espace mutationnel d'une proteine donnee.", "#f59e0b"),
        ("06","Construction d'une matrice de preferences mutationnelles DDG",
         "Generation d'une matrice (positions x substitutions) decrivant les effets predits de chaque "
         "substitution. Identification des positions sensibles et des profils mutationnels complets.", "#a855f7"),
    ]

    for num, title, text, clr in perspectives:
        st.markdown(f"""
        <div class="persp-card" style="border-left-color:{clr};">
            <div class="persp-num">ETAPE {num}</div>
            <div class="persp-title">{title}</div>
            <div class="persp-text">{text}</div>
        </div>""", unsafe_allow_html=True)
