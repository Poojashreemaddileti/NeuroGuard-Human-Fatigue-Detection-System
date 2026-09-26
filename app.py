from pathlib import Path
import base64
import streamlit as st
import pandas as pd
import numpy as np

from backend.eeg_processor import load_eeg_file, extract_band_features
from backend.model import FatigueModel
from backend.database import init_database, save_prediction, fetch_history

st.set_page_config(page_title="NeuroGuard", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")
ROOT = Path(__file__).parent
BRAIN = ROOT / "assets" / "neuroguard_brain_hero.png"
AMBIENT = ROOT / "assets" / "neuroguard_ambient.wav"

# Same visual direction as the original NeuroGuard interface: light futuristic, not Apple-style.
st.markdown("""
<style>
:root{--ink:#17224b;--muted:#65708f;--blue:#4f72ff;--violet:#8a54ff;--cyan:#51c8ff;--bg:#f7f9ff}
html,body,[class*="css"]{font-family:Inter,Segoe UI,Arial,sans-serif}
.stApp{background:radial-gradient(circle at 78% 16%,rgba(150,190,255,.30),transparent 28%),radial-gradient(circle at 72% 76%,rgba(205,178,255,.20),transparent 28%),#f7f9ff;color:var(--ink)}
.block-container{max-width:1600px;padding-top:1rem}
[data-testid="stSidebar"]{background:rgba(255,255,255,.72);border-right:1px solid rgba(75,94,160,.10);backdrop-filter:blur(18px)}
[data-testid="stSidebar"] .stButton>button{border:0!important;background:transparent!important;color:#56617f!important;text-align:left!important;border-radius:14px!important;font-weight:600!important;min-height:44px}
[data-testid="stSidebar"] .stButton>button:hover{background:linear-gradient(100deg,#e7e9ff,#edf7ff)!important;color:#5538df!important}
.brand{display:flex;align-items:center;gap:10px;padding:4px 8px 28px;font-weight:800;font-size:21px}
.brand-mark{width:43px;height:43px;border-radius:14px;display:grid;place-items:center;color:#fff;background:linear-gradient(135deg,#7560ff,#48c7ff);box-shadow:0 10px 25px rgba(94,108,255,.25);font-size:24px}
.brand small{display:block;font-size:9px;letter-spacing:2px;color:#707997;margin-top:2px}
.nav-title{font-size:9px;letter-spacing:2px;color:#8a94ad;margin:0 8px 9px}
.quote{margin-top:22px;padding:18px;border:1px solid rgba(90,110,190,.12);border-radius:20px;background:linear-gradient(145deg,rgba(255,255,255,.9),rgba(236,244,255,.75))}
.quote strong{display:block;font-size:18px;line-height:1.15}.quote span{font-size:12px;color:var(--muted)}
.hero{position:relative;min-height:470px;margin-top:5px;overflow:hidden;border-radius:30px;background:linear-gradient(105deg,rgba(255,255,255,.78),rgba(241,247,255,.46));border:1px solid rgba(105,120,195,.10);box-shadow:0 20px 60px rgba(70,90,150,.06)}
.copy{position:relative;z-index:8;padding:62px 0 45px 5%;width:56%}.eyebrow{letter-spacing:4px;font-size:12px;color:#697596;font-weight:700;margin-bottom:18px}
.hero h1{font-size:clamp(42px,5vw,76px);line-height:.95;margin:0;font-weight:800;letter-spacing:-3px}.grad{background:linear-gradient(90deg,#2767dc,#5d54ff,#a04fff);-webkit-background-clip:text;color:transparent}
.sub{margin:22px 0 25px;max-width:610px;font-size:16px;line-height:1.55;color:#68728f}.cta{display:inline-block;border:0;padding:14px 25px;border-radius:30px;color:#fff;font-weight:700;background:linear-gradient(100deg,#a553ff,#4e82ff);box-shadow:0 12px 28px rgba(102,84,255,.26)}
.hero-brain{position:absolute;right:-2%;top:2%;width:52%;height:96%;object-fit:contain;object-position:center;filter:drop-shadow(0 18px 38px rgba(75,86,170,.18));mix-blend-mode:multiply}
.status{display:flex;gap:16px;margin-top:22px;font-size:12px;color:#5c6783}.status b{color:#24305f}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:18px;margin-top:18px}.card{min-height:155px;border:1px solid rgba(90,110,190,.12);border-radius:22px;background:rgba(255,255,255,.80);padding:22px;box-shadow:0 10px 35px rgba(70,90,150,.06)}
.card .ci{width:45px;height:45px;border-radius:15px;display:grid;place-items:center;background:linear-gradient(135deg,#e4f4ff,#eee5ff);font-size:22px;margin-bottom:15px}.card h3{margin:0 0 8px;font-size:16px}.card p{margin:0;color:var(--muted);font-size:12px;line-height:1.45}
.page-title{font-size:42px;font-weight:800;letter-spacing:-2px;margin:12px 0 6px}.page-desc{color:var(--muted);margin-bottom:25px}.panel{padding:5px 10px}
.metric-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;margin-top:25px}.metric{padding:25px;border-radius:22px;background:#fff;border:1px solid #e5e9f5;box-shadow:0 10px 30px rgba(70,90,150,.05)}
.metric.green{background:linear-gradient(145deg,#eafaf1,#fff)}.metric.orange{background:linear-gradient(145deg,#fff3dc,#fff)}.metric.red{background:linear-gradient(145deg,#ffe7eb,#fff)}.metric.blue{background:linear-gradient(145deg,#e8f1ff,#fff)}
.metric .label{font-size:11px;letter-spacing:1.4px;color:#6e7893;font-weight:700}.metric .value{font-size:31px;font-weight:800;margin-top:8px}.green .value{color:#159565}.orange .value{color:#d18414}.red .value{color:#d43f53}.blue .value{color:#426bd3}
.gauge-wrap{display:flex;align-items:center;gap:45px;flex-wrap:wrap;margin-top:20px}.gauge{width:240px;height:240px;border-radius:50%;display:grid;place-items:center;position:relative}.gauge:after{content:"";width:180px;height:180px;background:#fff;border-radius:50%;display:grid;place-items:center;font-size:38px;font-weight:800;color:#27315c}.gauge.green{background:conic-gradient(#1fa56c 0 var(--g),#e4efe9 var(--g) 100%)}.gauge.orange{background:conic-gradient(#ee9d24 0 var(--g),#f4eadb var(--g) 100%)}.gauge.red{background:conic-gradient(#e34d5f 0 var(--g),#f3e3e6 var(--g) 100%)}
.gauge-text{position:absolute;inset:0;display:grid;place-items:center;z-index:2}.gauge-text b{font-size:38px}.gauge-text span{display:block;text-align:center;font-size:11px;color:#75809b;margin-top:-85px}
.upload-box{padding:18px;border-radius:24px;background:rgba(255,255,255,.72);border:1px solid rgba(90,110,190,.12);margin-bottom:18px}.section-card{padding:22px;border-radius:25px;background:#fff;border:1px solid #e5e9f5;box-shadow:0 10px 30px rgba(70,90,150,.05)}
.stButton>button{border-radius:14px!important;border:1px solid rgba(80,100,170,.12)!important;background:rgba(255,255,255,.82)!important;color:#334064!important;font-weight:700!important;min-height:44px}.stButton>button:hover{border-color:#7967ef!important;color:#5538df!important}
.primary .stButton>button{background:linear-gradient(100deg,#a553ff,#4e82ff)!important;color:#fff!important;border:0!important;box-shadow:0 12px 28px rgba(102,84,255,.22)}
[data-testid="stFileUploader"]{background:rgba(255,255,255,.55);border-radius:18px;border:1px solid rgba(90,110,190,.12)}
[data-testid="stDataFrame"]{border-radius:18px;overflow:hidden}
.small{font-size:11px;color:#8994a8;line-height:1.6}.footer{text-align:center;font-size:10px;color:#939caf;margin:35px 0 18px}
@media(max-width:1000px){.cards{grid-template-columns:repeat(2,1fr)}.copy{width:65%}.hero-brain{width:48%}}
</style>
""", unsafe_allow_html=True)

if "page" not in st.session_state: st.session_state.page="Home"
if "eeg" not in st.session_state: st.session_state.eeg=None
if "prediction" not in st.session_state: st.session_state.prediction=None
if "model" not in st.session_state: st.session_state.model=FatigueModel()
if "db_status" not in st.session_state: st.session_state.db_status=False
if "theme" not in st.session_state: st.session_state.theme="Light"
if "ambient_sound" not in st.session_state: st.session_state.ambient_sound=False

# User-facing theme preference. This changes only presentation; the original layout is preserved.
if st.session_state.theme == "Dark":
    st.markdown("""<style>
    .stApp{background:radial-gradient(circle at 78% 16%,rgba(65,91,170,.25),transparent 28%),radial-gradient(circle at 72% 76%,rgba(94,65,155,.18),transparent 28%),#0d1324;color:#eef2ff}
    [data-testid="stSidebar"]{background:rgba(15,22,39,.88);border-right-color:rgba(160,170,220,.12)}
    .hero{background:linear-gradient(105deg,rgba(25,34,56,.88),rgba(24,32,55,.62));border-color:rgba(150,160,220,.14)}
    .sub,.page-desc,.small,.status,.quote span,.card p{color:#aeb8d3!important}
    .card,.metric,.section-card,.upload-box{background:rgba(22,30,51,.92);border-color:rgba(150,160,220,.14);color:#eef2ff}
    .metric.green{background:linear-gradient(145deg,#15372c,#162033)}.metric.orange{background:linear-gradient(145deg,#3b2c15,#162033)}.metric.red{background:linear-gradient(145deg,#3d1c27,#162033)}.metric.blue{background:linear-gradient(145deg,#182b50,#162033)}
    .gauge:after{background:#182033;color:#eef2ff}
    [data-testid="stFileUploader"]{background:rgba(22,30,51,.72);border-color:rgba(150,160,220,.14)}
    </style>""", unsafe_allow_html=True)



# The Home hero CTA is a real navigation link while preserving the original visual design.
requested_page = st.query_params.get("page")
if requested_page in {"Home", "Fatigue Gauge", "Live EEG", "Drowsiness Analysis", "Prediction", "History", "Settings"}:
    st.session_state.page = requested_page
    st.query_params.clear()

with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-mark">🧠</div><div>Neuro<span style="color:#624fff">Guard</span><small>HUMAN FATIGUE AI</small></div></div>',unsafe_allow_html=True)
    st.markdown('<div class="nav-title">MONITORING</div>',unsafe_allow_html=True)
    for icon,name in [("⌂","Home"),("◉","Fatigue Gauge"),("〰","Live EEG"),("🧠","Drowsiness Analysis"),("▥","Prediction"),("◷","History"),("⚙","Settings")]:
        if st.button(f"{icon}  {name}",key="nav_"+name,use_container_width=True):
            st.session_state.page=name; st.rerun()
    if st.session_state.ambient_sound and AMBIENT.exists():
        st.markdown('<div class="nav-title" style="margin-top:18px">AMBIENT</div>',unsafe_allow_html=True)
        st.audio(str(AMBIENT), format="audio/wav", start_time=0)
    st.markdown('<div class="quote"><strong>Safer Minds<br>Brighter Tomorrows</strong><span>AI for safer human performance.</span></div>',unsafe_allow_html=True)

page=st.session_state.page

# Clear return path on every portal page so users never get stuck inside a section.
if page != "Home":
    if st.button("← Back to Home", key="back_home", use_container_width=False):
        st.session_state.page = "Home"
        st.rerun()

def status_class(level):
    return ["green","orange","red"][int(level)]

def result_cards(result):
    fc=status_class(result["level"])
    st.markdown(f'''<div class="metric-grid">
    <div class="metric {fc}"><div class="label">FATIGUE</div><div class="value">{result['fatigue']}</div><p>{result['fatigue_score']}% fatigue index</p></div>
    <div class="metric {('red' if result['risk_score']>=67 else 'orange' if result['risk_score']>=34 else 'green')}"><div class="label">RISK</div><div class="value">{result['risk_score']}%</div><p>Estimated session risk</p></div>
    <div class="metric blue"><div class="label">MODEL CONFIDENCE</div><div class="value">{result['confidence']:.1f}%</div><p>Random Forest probability</p></div>
    </div>''',unsafe_allow_html=True)

# HOME
if page=="Home":
    img64=base64.b64encode(BRAIN.read_bytes()).decode() if BRAIN.exists() else ""
    st.markdown(f'''<section class="hero"><div class="copy"><div class="eyebrow">DETECT · UNDERSTAND · PREVENT</div><h1><span class="grad">NeuroGuard</span><br>Human Fatigue Detection</h1><div class="sub">AI-powered EEG analysis to monitor mental state, predict fatigue and drowsiness, and support safer human performance.</div><a class="cta" href="?page=Live%20EEG">Get Started&nbsp; →</a><div class="status"><span>● <b>System online</b></span><span>EEG · RANDOM FOREST · MYSQL</span></div></div>{f'<img class="hero-brain" src="data:image/png;base64,{img64}">' if img64 else ''}</section>''',unsafe_allow_html=True)
    st.markdown('<div class="cards">',unsafe_allow_html=True)
    items=[("〰","Analyze EEG Signals","Upload CSV or EDF data and inspect the signal."),("🧠","Detect Fatigue & Drowsiness","Calculate band features and monitor mental state."),("♢","Make Safer Decisions","Run the Random Forest prediction pipeline."),("▥","Track Progress","Store prediction history in MySQL.")]
    cols=st.columns(4)
    for c,(ic,title,desc) in zip(cols,items):
        with c:
            st.markdown(f'<div class="card"><div class="ci">{ic}</div><h3>{title}</h3><p>{desc}</p></div>',unsafe_allow_html=True)
            if st.button("Open",key="open_"+title,use_container_width=True):
                target={"Analyze EEG Signals":"Live EEG","Detect Fatigue & Drowsiness":"Drowsiness Analysis","Make Safer Decisions":"Prediction","Track Progress":"History"}[title]
                st.session_state.page=target;st.rerun()
    st.markdown('</div>',unsafe_allow_html=True)

# LIVE EEG
elif page=="Live EEG":
    st.markdown('<div class="page-title">Live EEG</div><div class="page-desc">First, upload your EEG file here. This is the starting point for processing, prediction and drowsiness analysis. EDF metadata is read with MNE-Python; CSV files use 160 Hz by default.</div>',unsafe_allow_html=True)
    st.info("Step 1 — Upload your EEG file below (CSV or EDF). After the file loads, go to Prediction to run Random Forest, then open Drowsiness Analysis or Fatigue Gauge to view the results.")
    upload=st.file_uploader("Upload your EEG file here",type=["csv","edf"],key="eeg_upload")
    if upload:
        try:
            data,sfreq,channels=load_eeg_file(upload)
            st.session_state.eeg={"data":data,"sfreq":sfreq,"channels":channels,"name":upload.name}
            st.success(f"Loaded {len(channels)} channel(s) • {data.shape[1]} samples • {sfreq:g} Hz")
            idx=st.selectbox("EEG channel",range(min(len(channels),16)),format_func=lambda i:channels[i])
            sig=np.asarray(data[idx],dtype=float)
            n=min(len(sig),6000); sig=sig[-n:]; t=np.arange(n)/sfreq
            try:
                import plotly.graph_objects as go
                fig=go.Figure(go.Scatter(x=t,y=sig,mode="lines",line=dict(color="#5f73ff",width=1.6)))
                fig.update_layout(height=390,margin=dict(l=20,r=20,t=20,b=20),paper_bgcolor="#ffffff",plot_bgcolor="#fbfcff",font=dict(color="#56617f"),xaxis=dict(title="Time (seconds)",titlefont=dict(color="#5f73ff"),tickfont=dict(color="#697596"),gridcolor="#dfe5ff",zerolinecolor="#b9c4ef"),yaxis=dict(title="Amplitude",titlefont=dict(color="#8a54ff"),tickfont=dict(color="#697596"),gridcolor="#eee8ff",zerolinecolor="#cfc5ff"))
                st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
            except Exception:
                st.line_chart(pd.DataFrame({"EEG":sig},index=t),use_container_width=True,height=390)
            features=extract_band_features(data,sfreq)
            cs=st.columns(4)
            for c,(name,val) in zip(cs,zip(["Delta","Theta","Alpha","Beta"],features)):
                with c: st.markdown(f'<div class="metric blue"><div class="label">{name.upper()} BAND POWER</div><div class="value">{val:.4f}</div></div>',unsafe_allow_html=True)
        except Exception as e: st.error(f"Could not read the EEG file: {e}")
    else:
        st.markdown('<div class="section-card"><b>Supported input</b><p class="small">CSV: one or more numeric EEG channels. EDF: standard EEG recording with channel and sampling metadata.</p></div>',unsafe_allow_html=True)

# PREDICTION
elif page=="Prediction":
    st.markdown('<div class="page-title">Prediction</div><div class="page-desc">Run the Random Forest model on the currently loaded EEG recording.</div>',unsafe_allow_html=True)
    eeg=st.session_state.eeg
    if not eeg: st.info("First go to Live EEG and upload your CSV or EDF EEG file. Once it is uploaded, return here to run the Random Forest prediction.")
    else:
        st.markdown(f'<div class="section-card"><b>{eeg["name"]}</b><p class="small">{len(eeg["channels"])} channels • {eeg["sfreq"]:g} Hz</p></div>',unsafe_allow_html=True)
        st.write("")
        st.markdown('<div class="primary">',unsafe_allow_html=True)
        run=st.button("Run Random Forest Prediction",use_container_width=True,key="run_prediction")
        st.markdown('</div>',unsafe_allow_html=True)
        if run:
            features=extract_band_features(eeg["data"],eeg["sfreq"])
            pred,conf=st.session_state.model.predict(features)
            fatigue_score=int(np.clip(pred*50+(1-conf)*25,0,100))
            risk_score=int(np.clip(.65*fatigue_score+.35*(100-conf*100),0,100))
            drowsiness=int(np.clip(((1-features[2])*.55+(1-features[3])*.45)*100,0,100))
            result={"fatigue":["Normal","Average","Danger"][pred],"level":pred,"confidence":conf*100,"fatigue_score":fatigue_score,"risk_score":risk_score,"drowsiness":drowsiness,"alertness":100-drowsiness,"features":features}
            st.session_state.prediction=result
            try:
                init_database(); save_prediction(result,eeg["name"]); st.session_state.db_status=True
            except Exception as e:
                st.session_state.db_status=False
                st.warning("Prediction completed, but this session could not be added to History. Please try again later.")
        if st.session_state.prediction:
            result_cards(st.session_state.prediction)
            st.caption("The model is a project-level Random Forest implementation. It is not clinically validated.")

# FATIGUE GAUGE
elif page=="Fatigue Gauge":
    st.markdown('<div class="page-title">Fatigue Gauge</div><div class="page-desc">Fatigue status uses green for normal, orange for average and red for danger.</div>',unsafe_allow_html=True)
    r=st.session_state.prediction
    if not r: st.info("First upload your EEG file in Live EEG and run the Random Forest prediction. The fatigue gauge will then be populated here.")
    else:
        cls=status_class(r["level"]); score=r["fatigue_score"]
        st.markdown(f'''<div class="section-card"><div class="gauge-wrap"><div class="gauge {cls}" style="--g:{score}%;"><div class="gauge-text"><div><b>{score}%</b><span>fatigue index</span></div></div></div><div><h2 style="margin:0">{r['fatigue']}</h2><p class="page-desc">Current fatigue state from the Random Forest result.</p><div class="metric-grid" style="margin-top:15px"><div class="metric {cls}"><div class="label">FATIGUE</div><div class="value">{r['fatigue_score']}%</div></div><div class="metric {'red' if r['risk_score']>=67 else 'orange' if r['risk_score']>=34 else 'green'}"><div class="label">RISK</div><div class="value">{r['risk_score']}%</div></div></div></div></div></div>''',unsafe_allow_html=True)

# DROWSINESS
elif page=="Drowsiness Analysis":
    st.markdown('<div class="page-title">Drowsiness Analysis</div><div class="page-desc">Three independent visual states for alertness, drowsiness and risk.</div>',unsafe_allow_html=True)
    r=st.session_state.prediction
    if not r: st.info("First upload your EEG file in Live EEG, then run the prediction in Prediction. Your drowsiness results will appear here.")
    else:
        d=r["drowsiness"]; a=r["alertness"]; risk=r["risk_score"]
        ac="green" if a>=67 else "orange" if a>=34 else "red"; dc="red" if d>=67 else "orange" if d>=34 else "green"; rc="red" if risk>=67 else "orange" if risk>=34 else "green"
        st.markdown(f'''<div class="metric-grid"><div class="metric {ac}"><div class="label">ALERTNESS</div><div class="value">{a}%</div><p>{'Alert' if a>=67 else 'Reduced alertness' if a>=34 else 'Low alertness'}</p></div><div class="metric {dc}"><div class="label">DROWSINESS STATE</div><div class="value">{d}%</div><p>{'Drowsy' if d>=67 else 'Mild drowsiness' if d>=34 else 'Alert'}</p></div><div class="metric {rc}"><div class="label">RISK</div><div class="value">{risk}%</div><p>Session risk indicator</p></div></div>''',unsafe_allow_html=True)

# HISTORY
elif page=="History":
    st.markdown('<div class="page-title">History</div><div class="page-desc">Previous predictions saved in MySQL.</div>',unsafe_allow_html=True)
    try:
        init_database(); rows=fetch_history(50)
        if rows:
            df=pd.DataFrame(rows); st.dataframe(df,use_container_width=True,hide_index=True)
        else: st.info("No saved predictions yet.")
    except Exception as e:
        st.warning("History is temporarily unavailable. Please try again later.")

# SETTINGS
else:
    st.markdown('<div class="page-title">Settings</div><div class="page-desc">Personalize your NeuroGuard experience.</div>',unsafe_allow_html=True)
    st.markdown('<div class="section-card"><b>Appearance</b><p class="small">Choose how NeuroGuard looks on your device.</p></div>',unsafe_allow_html=True)
    theme=st.radio("Theme", ["Light", "Dark"], index=0 if st.session_state.theme=="Light" else 1, horizontal=True)
    if theme != st.session_state.theme:
        st.session_state.theme=theme
        st.rerun()
    st.markdown('<div class="section-card" style="margin-top:18px"><b>Ambient sound</b><p class="small">A subtle futuristic EEG-inspired ambient sound for a calmer monitoring experience. Use the player in the sidebar when enabled.</p></div>',unsafe_allow_html=True)
    ambient=st.toggle("Enable ambient sound", value=st.session_state.ambient_sound)
    if ambient != st.session_state.ambient_sound:
        st.session_state.ambient_sound=ambient
        st.rerun()
    if st.session_state.ambient_sound and AMBIENT.exists():
        st.info("Ambient sound is enabled. Use the player in the sidebar to start or pause it.")

st.markdown('<div class="footer">NEUROGUARD · HUMAN FATIGUE AI · PROJECT PROTOTYPE</div>',unsafe_allow_html=True)
