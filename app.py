"""
app.py — BayesCraft v3.5
Sin repr(), sin iframes, partículas CSS puras, botón Inicio funcional.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, io
warnings.filterwarnings("ignore")

from bayes_engine import (
    detectar_columnas, resumen_dataset,
    calcular_bayes, calcular_probabilidades_multiples,
    curva_posterior, entrenar_naive_bayes,
    preparar_serie_temporal, calcular_correlaciones,
)
from minecraft_theme import (
    get_css, get_particles, get_navbar,
    get_hero_html, get_footer, get_sticky_js,
    load_media_info, prepare_static_files,
    apply_mc_style, mc_cmap, CHART_COLORS, MC,
    formula_box, ins, divider, section_hdr,
    prob_table, progress_bar, badge,
)

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="BayesCraft", page_icon="⛏",
    layout="wide", initial_sidebar_state="collapsed",
)
prepare_static_files()
st.markdown(get_css(), unsafe_allow_html=True)

# Estado
for k, v in [("df", None), ("use_demo", False), ("csv_err", "")]:
    if k not in st.session_state:
        st.session_state[k] = v

hay_datos = st.session_state["df"] is not None or st.session_state["use_demo"]

# ── PARTÍCULAS — CSS puro, siempre visibles ──────────────
# Partículas: dimmed=True cuando hay datos (opacidad 0.65, siempre visibles)
st.markdown(get_particles(dimmed=hay_datos), unsafe_allow_html=True)

# ── NAVBAR ───────────────────────────────────────────────
st.markdown(get_navbar(), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# PANTALLA DE BIENVENIDA
# ═══════════════════════════════════════════════════════════
if not hay_datos:

    media_info = load_media_info()
    st.markdown(get_hero_html(media_info), unsafe_allow_html=True)

    # Barra de carga
    st.markdown(
        '<div style="background:rgba(6,6,18,.96);border-bottom:1px solid #1a1a2a;'
        'padding:12px 20px 14px;backdrop-filter:blur(12px);position:relative;z-index:10;">'
        '<div style="font-family:\'Press Start 2P\',monospace;font-size:8px;color:#2a2a3a;'
        'text-transform:uppercase;letter-spacing:2px;margin-bottom:10px;">'
        '&#9935; Cargar dataset</div></div>',
        unsafe_allow_html=True,
    )

    u1, u2, u3, u4 = st.columns([4, 0.65, 0.85, 0.65])
    with u1:
        archivo = st.file_uploader(
            "Selecciona CSV", type=["csv","txt"],
            label_visibility="collapsed", key="up_hero",
            help="Detecta separador automáticamente",
        )
    with u2:
        sep_h = st.selectbox("Separador", [",",";","|","\\t","auto"],
                             label_visibility="collapsed", key="sep_h")
    with u3:
        enc_h = st.selectbox("Codificación", ["utf-8","latin-1","utf-8-sig","cp1252"],
                             label_visibility="collapsed", key="enc_h")
    with u4:
        demo_btn = st.button("⚡ Demo", key="demo_h")

    if st.session_state["csv_err"]:
        st.markdown(ins(f"❌ {st.session_state['csv_err']}", "d"), unsafe_allow_html=True)
        st.session_state["csv_err"] = ""

    if demo_btn:
        st.session_state["use_demo"] = True
        st.rerun()

    if archivo is not None:
        raw  = archivo.read()
        seps = ["\t",";",",","|"] if sep_h == "auto" else ["\t" if sep_h == "\\t" else sep_h]
        encs = ["utf-8","latin-1","utf-8-sig","cp1252"] if sep_h == "auto" else [enc_h]
        df_ok = None; last_e = ""
        for enc in encs:
            for sep in seps:
                try:
                    tmp = pd.read_csv(io.BytesIO(raw), sep=sep, encoding=enc,
                                      engine="python", on_bad_lines="skip")
                    if tmp.shape[1] >= 2 and tmp.shape[0] >= 2:
                        tmp.columns = tmp.columns.str.strip()
                        df_ok = tmp; break
                except Exception as e: last_e = str(e)
            if df_ok is not None: break
        if df_ok is not None:
            st.session_state["df"] = df_ok; st.rerun()
        else:
            st.session_state["csv_err"] = (
                f"No pude leer el archivo. Prueba otro separador. ({last_e[:60]})"
            )
            st.rerun()

    # Descripción
    st.markdown(
        '<div style="background:#141418;padding:52px 40px;text-align:center;z-index:5;position:relative;">'
        '<div style="font-family:\'Press Start 2P\',monospace;font-size:12px;color:#FFAA00;'
        'text-shadow:2px 2px 0 #5a3a00,0 0 20px rgba(255,170,0,.3);margin-bottom:16px;letter-spacing:2px;">'
        'QUE ES BAYESCRAFT?</div>'
        '<div style="font-family:Rajdhani,sans-serif;font-size:16px;color:#7A7A8A;'
        'max-width:680px;margin:0 auto 40px;line-height:1.8;">'
        'Herramienta de análisis estadístico bayesiano para detectar eventos anómalos '
        'en datasets industriales. Carga tu CSV y obtén probabilidades, visualizaciones e insights con IA.'
        '</div>'
        '<div style="display:flex;justify-content:center;flex-wrap:wrap;max-width:800px;margin:0 auto;'
        'border-top:1px solid #28283A;border-bottom:1px solid #28283A;">'
        + "".join([
            f'<div style="flex:1;min-width:130px;padding:22px 16px;'
            f'border-right:{"1px solid #28283A" if i<4 else "none"};text-align:center;">'
            f'<span style="font-family:VT323,monospace;font-size:54px;color:#5D8731;'
            f'text-shadow:0 0 14px rgba(93,135,49,.6);line-height:1;display:block;margin-bottom:6px;">{n}</span>'
            f'<div style="font-family:Rajdhani,sans-serif;font-weight:700;font-size:11px;'
            f'text-transform:uppercase;letter-spacing:1.5px;color:#5a5a6a;">{lbl}</div></div>'
            for i,(n,lbl) in enumerate([
                ("1","Carga tu CSV"),("2","Detecta columnas"),
                ("3","Aplica Bayes"),("4","Entrena modelo"),("5","IA Insights"),
            ])
        ])
        + '</div></div>',
        unsafe_allow_html=True,
    )

    # Franja pasto + feature cards
    st.markdown(
        '<div style="width:100%;height:18px;'
        'background:linear-gradient(180deg,#7AB840 0%,#5D8731 45%,#2D4518 100%);'
        'box-shadow:0 0 22px rgba(122,184,64,.55),0 0 45px rgba(93,135,49,.25);"></div>',
        unsafe_allow_html=True,
    )
    fc = st.columns(4)
    for col,icon,title,desc in [
        (fc[0],"🔍","Detección Auto",   "Identifica numéricas, categóricas, fechas y binarias"),
        (fc[1],"🧮","Teorema de Bayes", "Calcula P(A), P(B|A) y P(A|B) con visualizaciones"),
        (fc[2],"🤖","Naive Bayes",      "Clasificador con matriz de confusión y métricas"),
        (fc[3],"💡","IA Insights",      "Análisis profundo con Claude de Anthropic"),
    ]:
        with col:
            st.markdown(
                f'<div style="background:#0C0C12;border:1px solid #28283A;'
                f'border-bottom:3px solid #5D8731;padding:28px 20px;text-align:center;">'
                f'<div style="font-size:32px;margin-bottom:14px;">{icon}</div>'
                f'<div style="font-family:Rajdhani,sans-serif;font-weight:700;font-size:11px;'
                f'text-transform:uppercase;letter-spacing:2px;color:#FFAA00;'
                f'text-shadow:0 0 10px rgba(255,170,0,.35);margin-bottom:10px;">{title}</div>'
                f'<div style="font-family:Rajdhani,sans-serif;font-size:13px;'
                f'color:#7A7A8A;line-height:1.55;">{desc}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown(get_footer(), unsafe_allow_html=True)
    st.stop()


# ═══════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════
if st.session_state["use_demo"] and st.session_state["df"] is None:
    np.random.seed(42); n = 250
    temp  = np.random.normal(70, 15, n)
    fallo = np.array([np.random.binomial(1, np.clip((t-60)/80,.05,.95)) for t in temp])
    st.session_state["df"] = pd.DataFrame({
        "fecha":       pd.date_range("2023-01-01", periods=n, freq="D").astype(str),
        "temperatura": temp.round(2),
        "presion":     np.random.normal(100,10,n).round(2),
        "vibracion":   np.random.exponential(5,n).round(2),
        "humedad":     np.random.normal(65,8,n).round(2),
        "turno":       np.random.choice(["mañana","tarde","noche"],n),
        "operador":    np.random.choice(["Steve","Alex","Creeper","Enderman"],n),
        "zona":        np.random.choice(["A","B","C"],n),
        "fallo":       fallo,
    })

df = st.session_state["df"]

# Cabecera simple — sin botones duplicados (solo en sticky bar)
st.markdown(
    '<div style="background:rgba(8,8,20,.97);border-bottom:1px solid #1a1a2a;'
    'padding:10px 24px;position:relative;z-index:20;display:flex;align-items:center;gap:14px;">'
    '<span style="font-family:\'Press Start 2P\',monospace;font-size:10px;'
    'color:#FFAA00;text-shadow:0 0 14px rgba(255,170,0,.4);">&#9935; BAYESCRAFT</span>'
    '<span style="font-family:Rajdhani,sans-serif;font-size:11px;color:#2a2a3a;'
    'text-transform:uppercase;letter-spacing:2px;border-left:1px solid #1a1a28;padding-left:14px;">'
    'Dataset activo</span>'
    '</div>',
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════
# STICKY BAR (aparece al hacer scroll)
# ═══════════════════════════════════════════════════════════
st.markdown(
    '<div class="mc-sticky" id="mcStickyBar">'
    '<div style="max-width:1100px;margin:0 auto;">',
    unsafe_allow_html=True,
)
s1,s2,s3,s4,s5 = st.columns([3.2,.55,.8,.6,.55])
with s1:
    arch_s = st.file_uploader("Cambiar CSV", type=["csv","txt"],
                               label_visibility="collapsed", key="up_s")
with s2:
    sep_s = st.selectbox("Sep", [",",";","|","\\t","auto"],
                         label_visibility="collapsed", key="sep_s")
with s3:
    enc_s = st.selectbox("Enc", ["utf-8","latin-1","utf-8-sig","cp1252"],
                         label_visibility="collapsed", key="enc_s")
with s4:
    rst_s = st.button("🗑 Reset", key="rst_s")
with s5:
    home_s = st.button("⌂ Inicio", key="home_s")
st.markdown('</div></div>', unsafe_allow_html=True)
st.markdown(get_sticky_js(), unsafe_allow_html=True)

if rst_s or home_s:
    for k in ["df","use_demo","csv_err"]:
        st.session_state[k] = None if k=="df" else (False if k=="use_demo" else "")
    st.rerun()

if arch_s is not None:
    raw_s = arch_s.read()
    seps_s = ["\t",";",",","|"] if sep_s=="auto" else ["\t" if sep_s=="\\t" else sep_s]
    encs_s = ["utf-8","latin-1","utf-8-sig","cp1252"] if sep_s=="auto" else [enc_s]
    df_new = None
    for enc in encs_s:
        for sep in seps_s:
            try:
                tmp = pd.read_csv(io.BytesIO(raw_s),sep=sep,encoding=enc,
                                  engine="python",on_bad_lines="skip")
                if tmp.shape[1]>=2 and tmp.shape[0]>=2:
                    tmp.columns=tmp.columns.str.strip(); df_new=tmp; break
            except: pass
        if df_new is not None: break
    if df_new is not None:
        st.session_state["df"]=df_new; df=df_new; st.rerun()
    else:
        st.error("No se pudo leer. Intenta otro separador o codificación.")

# ═══════════════════════════════════════════════════════════
# MÉTRICAS
# ═══════════════════════════════════════════════════════════
info_cols = detectar_columnas(df)
res_df    = resumen_dataset(df)

m1,m2,m3,m4,m5,m6 = st.columns(6)
m1.metric("📋 Filas",      f"{res_df['filas']:,}")
m2.metric("📊 Columnas",   f"{res_df['columnas']}")
m3.metric("🔢 Numéricas",  f"{len(info_cols['numericas'])}")
m4.metric("🏷️ Categ.",     f"{len(info_cols['categoricas'])}")
m5.metric("📅 Fechas",     f"{len(info_cols['fechas'])}")
m6.metric("🔵 Binarias",   f"{len(info_cols['binarias'])}")

# ═══════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════
t1,t2,t3,t4,t5 = st.tabs([
    "🗺️  Explorar","🧮  Teorema de Bayes",
    "🤖  Naive Bayes","📈  Visualizaciones","💡  IA Insights",
])

# ── TAB 1 ────────────────────────────────────────────────
with t1:
    ca,cb=st.columns([3,2])
    with ca:
        st.markdown(section_hdr("Vista previa","📋"),unsafe_allow_html=True)
        nr=st.slider("Filas",5,min(100,len(df)),10,key="exp_r")
        st.dataframe(df.head(nr),use_container_width=True,height=280)
    with cb:
        st.markdown(section_hdr("Tipos detectados","🔍"),unsafe_allow_html=True)
        for (ico,nm,bc),cols_l in {
            ("🔢","Numéricas","o"):  info_cols["numericas"],
            ("🏷️","Categ.","c"):    info_cols["categoricas"],
            ("📅","Fechas","s"):    info_cols["fechas"],
            ("🔵","Binarias","g"):  info_cols["binarias"],
        }.items():
            if cols_l:
                bgs=" ".join(f'<span class="mb mb-{bc}">{c}</span>' for c in cols_l)
                st.markdown(
                    f'<div style="margin:10px 0;">'
                    f'<div style="font-family:Rajdhani,sans-serif;font-weight:700;font-size:10px;'
                    f'color:#444;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px;">'
                    f'{ico} {nm}</div><div>{bgs}</div></div>',unsafe_allow_html=True)
    st.markdown(divider(),unsafe_allow_html=True)
    if info_cols["numericas"]:
        st.markdown(section_hdr("Estadísticas","📊"),unsafe_allow_html=True)
        st.dataframe(df[info_cols["numericas"]].describe().round(3),use_container_width=True)
        st.markdown(section_hdr("Histogramas","📊"),unsafe_allow_html=True)
        nn=len(info_cols["numericas"]); cg=min(3,nn); rg=(nn+cg-1)//cg
        fig,axs=plt.subplots(rg,cg,figsize=(5*cg,3.5*rg),squeeze=False)
        fl=axs.flatten()
        for i,col in enumerate(info_cols["numericas"]):
            fl[i].hist(df[col].dropna(),bins=25,color=CHART_COLORS[i%10],edgecolor=MC["d1"],linewidth=0.4,alpha=0.9)
            fl[i].set_title(col); fl[i].set_xlabel("Valor"); fl[i].set_ylabel("Frecuencia")
        for j in range(nn,len(fl)): fl[j].set_visible(False)
        apply_mc_style(fig,fl[:nn]); plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    nulos=df.isnull().sum(); nulos=nulos[nulos>0]
    if len(nulos):
        st.markdown(section_hdr("Valores nulos","🕳️"),unsafe_allow_html=True)
        fig,ax=plt.subplots(figsize=(max(6,len(nulos)*1.5),3))
        ax.bar(nulos.index,nulos.values,color=MC["red"],edgecolor=MC["d1"],linewidth=0.4)
        ax.tick_params(axis="x",rotation=30)
        apply_mc_style(fig,ax); plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    else:
        st.markdown(ins("✅ Sin valores nulos — dataset limpio.","s"),unsafe_allow_html=True)

# ── TAB 2 ────────────────────────────────────────────────
with t2:
    st.markdown(section_hdr("Configuración","⚙️"),unsafe_allow_html=True)
    pos_t=(info_cols["binarias"]+info_cols["categoricas"]) or list(df.columns)
    b1,b2=st.columns(2)
    with b1: tc=st.selectbox("Variable objetivo:",pos_t,key="bt")
    with b2:
        fd=[c for c in df.columns if c!=tc and c not in info_cols["fechas"]]
        fc_s=st.selectbox("Variable de evidencia:",fd,key="bf")
    umb=None
    if fc_s in info_cols["numericas"]:
        cd=df[fc_s].dropna()
        umb=st.slider(f"Umbral '{fc_s}'",float(cd.min()),float(cd.max()),float(cd.median()),
                      step=float((cd.max()-cd.min())/100),key="bu")
    st.markdown(divider(),unsafe_allow_html=True)
    st.markdown(formula_box("P(A|B) = [ P(B|A) &times; P(A) ] / P(B)"),unsafe_allow_html=True)
    if st.button("⚡ Calcular Probabilidades Bayesianas",key="btn_b"):
        with st.spinner("Calculando..."):
            try:
                r=calcular_bayes(df,tc,fc_s,umb)
                c1,c2,c3,c4=st.columns(4)
                c1.metric("P(A) Prior",f"{r['P_A']:.4f}",f"{r['P_A']*100:.1f}%")
                c2.metric("P(B) Evidencia",f"{r['P_B']:.4f}")
                c3.metric("P(B|A) Verosim.",f"{r['P_B_dado_A']:.4f}")
                c4.metric("P(A|B) BAYES",f"{r['P_A_dado_B']:.4f}",f"x{r['factor_bayes']:.2f}")
                st.markdown(prob_table([
                    ("P(A) Prior",f"{r['P_A']:.6f}",f"{r['P_A']*100:.2f}%",False),
                    ("P(B) Evidencia",f"{r['P_B']:.6f}",f"{r['P_B']*100:.2f}%",False),
                    ("P(B|A) Verosimilitud",f"{r['P_B_dado_A']:.6f}",f"{r['P_B_dado_A']*100:.2f}%",False),
                    ("P(B|noA)",f"{r['P_B_dado_no_A']:.6f}",f"{r['P_B_dado_no_A']*100:.2f}%",False),
                    ("P(A|B) RESULTADO",f"{r['P_A_dado_B']:.6f}",f"{r['P_A_dado_B']*100:.2f}%",True),
                ]),unsafe_allow_html=True)
                d=r["delta"]
                msg=(f"🔴 Riesgo AUMENTA {d*100:.1f}pp (x{r['factor_bayes']:.2f})" if d>.05
                     else f"🟢 Riesgo REDUCE {abs(d)*100:.1f}pp" if d<-.05
                     else f"🟡 Efecto NEUTRO (delta={d*100:.1f}pp)")
                st.markdown(ins(msg,"d" if d>.05 else("s" if d<-.05 else "w")),unsafe_allow_html=True)
                st.markdown(progress_bar(r["P_A"],"Prior P(Fallo)"),unsafe_allow_html=True)
                st.markdown(progress_bar(r["P_A_dado_B"],"Posterior P(Fallo|Evidencia)"),unsafe_allow_html=True)
                fig,ax=plt.subplots(figsize=(7,4))
                vals=[r["P_A"],r["P_A_dado_B"]]
                bars=ax.bar(["P(Fallo)\nPrior","P(Fallo|Evidencia)\nPosterior"],vals,
                            color=[MC["diamond"],MC["red"] if d>0 else MC["em"]],
                            width=0.42,edgecolor=MC["d1"],linewidth=0)
                for bar,v in zip(bars,vals):
                    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.006,
                            f"{v*100:.2f}%",ha="center",color=MC["gold"],fontsize=11,fontweight="bold")
                ax.set_ylabel("Probabilidad"); ax.set_title(f"Actualización: {tc}")
                ax.set_ylim(0,min(1.15,max(vals)*1.55))
                ax.axhline(0.5,color="#444",linewidth=1,linestyle="--",alpha=0.5)
                apply_mc_style(fig,ax); plt.tight_layout(); st.pyplot(fig); plt.close(fig)
                if len(info_cols["numericas"])>1:
                    st.markdown(section_hdr("Impacto por variable","🔥"),unsafe_allow_html=True)
                    dm=calcular_probabilidades_multiples(df,tc,info_cols["numericas"])
                    if not dm.empty:
                        st.dataframe(dm.round(4),use_container_width=True)
                        fig,ax=plt.subplots(figsize=(9,3.5))
                        ax.bar(dm["Feature"],dm["Δ Riesgo"],
                               color=[MC["red"] if x>0 else MC["em"] for x in dm["Δ Riesgo"]],
                               edgecolor=MC["d1"],linewidth=0)
                        ax.axhline(0,color="#555",linewidth=1); ax.tick_params(axis="x",rotation=30)
                        apply_mc_style(fig,ax); plt.tight_layout(); st.pyplot(fig); plt.close(fig)
            except Exception as e: st.error(f"Error: {e}")

# ── TAB 3 ────────────────────────────────────────────────
with t3:
    st.markdown(section_hdr("Clasificador Naive Bayes","🤖"),unsafe_allow_html=True)
    pn=(info_cols["binarias"]+info_cols["categoricas"]) or list(df.columns)
    n1,n2=st.columns(2)
    with n1: tn=st.selectbox("Variable objetivo:",pn,key="nbt")
    with n2:
        fo=[c for c in (info_cols["numericas"]+info_cols["categoricas"]+info_cols["binarias"]) if c!=tn]
        fn=st.multiselect("Features:",fo,default=fo[:min(4,len(fo))],key="nbf")
    if not fn:
        st.markdown(ins("⚠️ Selecciona al menos 1 feature.","w"),unsafe_allow_html=True)
    elif st.button("🤖 Entrenar Clasificador",key="btn_nb"):
        with st.spinner("Entrenando..."):
            try:
                rnb=entrenar_naive_bayes(df,tn,fn)
                r1,r2,r3,r4=st.columns(4)
                r1.metric("🎯 Accuracy",f"{rnb['accuracy']*100:.1f}%")
                r2.metric("📡 Sensibilidad",f"{rnb['sensibilidad']*100:.1f}%")
                r3.metric("🛡️ Especific.",f"{rnb['especificidad']*100:.1f}%")
                r4.metric("⚖️ F1-Score",f"{rnb['f1']*100:.1f}%")
                st.markdown(
                    f'<div style="font-family:Rajdhani,sans-serif;font-size:12px;color:#444;'
                    f'text-align:center;margin:6px 0 16px;">'
                    f'Train:{rnb["n_train"]} | Test:{rnb["n_test"]} | '
                    f'TP={rnb["tp"]} FP={rnb["fp"]} FN={rnb["fn"]} TN={rnb["tn"]}</div>',
                    unsafe_allow_html=True)
                nc1,nc2=st.columns(2)
                with nc1:
                    st.markdown(section_hdr("Matriz confusión","🔲"),unsafe_allow_html=True)
                    fig,ax=plt.subplots(figsize=(5,4))
                    sns.heatmap(rnb["cm"],annot=True,fmt="d",cmap=mc_cmap("green"),
                                linewidths=2,linecolor=MC["d1"],
                                annot_kws={"size":18,"weight":"bold","color":"white"},
                                ax=ax,cbar=False,xticklabels=["No Fallo","Fallo"],
                                yticklabels=["No Fallo","Fallo"])
                    ax.set_title("Confusión"); ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
                    apply_mc_style(fig,ax); plt.tight_layout(); st.pyplot(fig); plt.close(fig)
                with nc2:
                    st.markdown(section_hdr("Importancia variables","📊"),unsafe_allow_html=True)
                    fi=rnb["feature_importance"]
                    if fi:
                        fis=sorted(fi.items(),key=lambda x:x[1],reverse=True)
                        fig,ax=plt.subplots(figsize=(5,4))
                        nms=[x[0] for x in fis]; vls=[x[1] for x in fis]
                        ax.barh(nms[::-1],vls[::-1],
                                color=[CHART_COLORS[i%10] for i in range(len(nms))][::-1],
                                edgecolor=MC["d1"],linewidth=0)
                        ax.set_title("Delta medias"); ax.set_xlabel("Diferencia")
                        apply_mc_style(fig,ax); plt.tight_layout(); st.pyplot(fig); plt.close(fig)
                fig,ax=plt.subplots(figsize=(9,3.5))
                yt=rnb["y_test"]; yp=rnb["y_prob"]
                ax.hist(yp[yt==0],bins=20,alpha=0.8,color=MC["diamond"],label="No Fallo",edgecolor=MC["d1"],linewidth=0.4)
                ax.hist(yp[yt==1],bins=20,alpha=0.8,color=MC["red"],label="Fallo",edgecolor=MC["d1"],linewidth=0.4)
                ax.axvline(0.5,color=MC["gold"],linewidth=2,linestyle="--",label="Umbral 0.5")
                ax.set_title("Distribución probabilidades"); ax.set_xlabel("P(Fallo)"); ax.set_ylabel("Frecuencia")
                ax.legend(fontsize=9,facecolor=MC["d2"],edgecolor=MC["b"],labelcolor=MC["text"])
                apply_mc_style(fig,ax); plt.tight_layout(); st.pyplot(fig); plt.close(fig)
                a=rnb["accuracy"]
                it=("s","✅ EXCELENTE") if a>=.85 else(("w","🟡 ACEPTABLE") if a>=.70 else("d","🔴 DÉBIL"))
                st.markdown(ins(f"{it[1]} — {a*100:.1f}% accuracy.",it[0]),unsafe_allow_html=True)
            except Exception as e: st.error(f"Error: {e}")

# ── TAB 4 ────────────────────────────────────────────────
with t4:
    cv=info_cols["numericas"] or list(df.columns)
    vc=st.selectbox("Variable principal:",cv,key="vm")
    dv=df[vc].dropna()
    v1,v2=st.columns(2)
    with v1:
        st.markdown(section_hdr("Distribución + KDE","📊"),unsafe_allow_html=True)
        fig,ax=plt.subplots(figsize=(6,4))
        ax.hist(dv,bins=28,color=MC["grass"],edgecolor=MC["d1"],linewidth=0.4,alpha=0.9,density=True)
        try:
            from scipy.stats import gaussian_kde
            kde=gaussian_kde(dv); xs=np.linspace(dv.min(),dv.max(),200)
            ax.plot(xs,kde(xs),color=MC["gold"],linewidth=2.5,label="KDE")
        except: pass
        ax.axvline(dv.mean(),color=MC["diamond"],linewidth=2,linestyle="--",label=f"Media:{dv.mean():.2f}")
        ax.axvline(dv.median(),color=MC["em"],linewidth=2,linestyle="-.",label=f"Mediana:{dv.median():.2f}")
        ax.legend(fontsize=7,facecolor=MC["d2"],edgecolor=MC["b"],labelcolor=MC["text"])
        ax.set_title(f"Distribución: {vc}"); ax.set_xlabel(vc); ax.set_ylabel("Densidad")
        apply_mc_style(fig,ax); plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    with v2:
        st.markdown(section_hdr("Boxplot","📦"),unsafe_allow_html=True)
        fig,ax=plt.subplots(figsize=(6,4))
        ax.boxplot(dv,patch_artist=True,widths=0.45,
                   boxprops=dict(facecolor=MC["gdk"],color=MC["grass"],linewidth=2),
                   whiskerprops=dict(color=MC["stone"],linewidth=1.5),
                   capprops=dict(color=MC["stone"],linewidth=2),
                   medianprops=dict(color=MC["gold"],linewidth=3),
                   flierprops=dict(marker="s",markerfacecolor=MC["red"],markersize=4,alpha=0.7,linestyle="none"))
        ax.set_title(f"Boxplot: {vc}"); ax.set_ylabel(vc); ax.set_xticks([])
        apply_mc_style(fig,ax); plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    if info_cols["fechas"]:
        st.markdown(section_hdr("Serie temporal","📅"),unsafe_allow_html=True)
        fs=st.selectbox("Fecha:",info_cols["fechas"],key="vf")
        try:
            dts=preparar_serie_temporal(df,fs,vc)
            fig,ax=plt.subplots(figsize=(12,4))
            ax.plot(dts.index,dts[vc],color=MC["em"],linewidth=1.5,alpha=0.9,zorder=3)
            ax.fill_between(dts.index,dts[vc],alpha=0.12,color=MC["grass"])
            mt=dts[vc].mean()
            ax.axhline(mt,color=MC["gold"],linewidth=1.5,linestyle="--",label=f"Media:{mt:.2f}")
            ax.set_title(f"Serie Temporal: {vc}"); ax.set_xlabel("Fecha"); ax.set_ylabel(vc)
            ax.legend(fontsize=8,facecolor=MC["d2"],edgecolor=MC["b"],labelcolor=MC["text"])
            apply_mc_style(fig,ax); plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        except Exception as e: st.markdown(ins(f"Error: {e}","w"),unsafe_allow_html=True)
    if len(info_cols["numericas"])>=2:
        st.markdown(section_hdr("Correlaciones","🔥"),unsafe_allow_html=True)
        cm_data=calcular_correlaciones(df,info_cols["numericas"]); nc=len(info_cols["numericas"])
        fig,ax=plt.subplots(figsize=(max(6,nc),max(5,nc-1)))
        sns.heatmap(cm_data,annot=True,fmt=".2f",cmap=mc_cmap("diverging"),
                    linewidths=1.5,linecolor=MC["d1"],annot_kws={"size":9,"color":"white"},
                    vmin=-1,vmax=1,ax=ax,cbar=True)
        ax.set_title("Correlación de Pearson")
        apply_mc_style(fig,ax); plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    if info_cols["categoricas"]:
        st.markdown(section_hdr("Categóricas","🏷️"),unsafe_allow_html=True)
        cat=st.selectbox("Variable:",info_cols["categoricas"],key="vc2")
        cts=df[cat].value_counts(); cc=CHART_COLORS[:len(cts)]
        fig,axs=plt.subplots(1,2,figsize=(12,4))
        axs[0].bar(cts.index.astype(str),cts.values,color=cc,edgecolor=MC["d1"],linewidth=0)
        axs[0].set_title(f"Conteo: {cat}"); axs[0].tick_params(axis="x",rotation=30)
        w,txts,ats=axs[1].pie(cts.values,labels=cts.index.astype(str),colors=cc,
                               autopct="%1.1f%%",startangle=90,pctdistance=0.82,
                               wedgeprops=dict(edgecolor=MC["d1"],linewidth=1.5))
        for t in txts: t.set_color(MC["text"]); t.set_fontsize(9)
        for at in ats: at.set_color("white"); at.set_fontsize(8); at.set_fontweight("bold")
        axs[1].set_title(f"Proporción: {cat}")
        apply_mc_style(fig,[axs[0],axs[1]]); plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    if info_cols["numericas"] and (info_cols["binarias"] or info_cols["categoricas"]):
        st.markdown(section_hdr("Probabilidad posterior","🔮"),unsafe_allow_html=True)
        pc1,pc2=st.columns(2)
        with pc1: tp=st.selectbox("Target:",info_cols["binarias"]+info_cols["categoricas"],key="pt")
        with pc2: fp=st.selectbox("Feature:",info_cols["numericas"],key="pf")
        try:
            dp=curva_posterior(df,tp,fp)
            if dp:
                fig,ax=plt.subplots(figsize=(10,4))
                ax.plot(dp["xs"],dp["posterior"],color=MC["red"],linewidth=2.5,label="P(Fallo|Evidencia)")
                ax.axhline(dp["P_A"],color=MC["gold"],linewidth=1.8,linestyle="--",label=f"P(Fallo)={dp['P_A']:.3f}")
                ax.fill_between(dp["xs"],dp["P_A"],dp["posterior"],where=(dp["posterior"]>dp["P_A"]),alpha=0.22,color=MC["red"])
                ax.fill_between(dp["xs"],dp["P_A"],dp["posterior"],where=(dp["posterior"]<=dp["P_A"]),alpha=0.22,color=MC["em"])
                ax.set_title(f"P({tp}|{fp}>umbral)"); ax.set_xlabel(f"Umbral {fp}"); ax.set_ylabel("P Posterior"); ax.set_ylim(0,1.05)
                ax.legend(fontsize=8,facecolor=MC["d2"],edgecolor=MC["b"],labelcolor=MC["text"])
                apply_mc_style(fig,ax); plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        except Exception as e: st.markdown(ins(f"Error: {e}","w"),unsafe_allow_html=True)

# ── TAB 5 ────────────────────────────────────────────────
with t5:
    st.markdown(section_hdr("Insights con IA","💡"),unsafe_allow_html=True)

    # ── Selector de proveedor ──────────────────────────────
    st.markdown("""
    <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:14px;">
      <div style="background:#0C0C12;border:1px solid #28283A;border-top:2px solid #4285F4;
        padding:14px 20px;border-radius:2px;flex:1;min-width:200px;">
        <div style="font-family:Rajdhani,sans-serif;font-weight:700;font-size:12px;
          color:#4285F4;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">
          ✦ Google Gemini — GRATIS</div>
        <div style="font-family:Rajdhani,sans-serif;font-size:12px;color:#7A7A8A;">
          60 consultas/min gratuitas<br>
          Key en: <a href="https://aistudio.google.com/apikey" target="_blank"
          style="color:#4DD9D9;">aistudio.google.com/apikey</a></div>
      </div>
      <div style="background:#0C0C12;border:1px solid #28283A;border-top:2px solid #FFAA00;
        padding:14px 20px;border-radius:2px;flex:1;min-width:200px;">
        <div style="font-family:Rajdhani,sans-serif;font-weight:700;font-size:12px;
          color:#FFAA00;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">
          ⛏ Claude (Anthropic) — De pago</div>
        <div style="font-family:Rajdhani,sans-serif;font-size:12px;color:#7A7A8A;">
          ~$0.003 por análisis<br>
          Key en: <a href="https://console.anthropic.com" target="_blank"
          style="color:#4DD9D9;">console.anthropic.com</a></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    pi=(info_cols["binarias"]+info_cols["categoricas"]) or list(df.columns)
    ai1,ai2,ai3 = st.columns([1.2, 2, 1])
    with ai1:
        proveedor = st.selectbox("Proveedor IA:",
            ["✦ Gemini (Gratis)", "⛏ Claude (Anthropic)"],
            label_visibility="collapsed", key="ia_prov")
    with ai2:
        ak = st.text_input("API Key:", type="password",
            placeholder="AIza... (Gemini)  ó  sk-ant-... (Claude)",
            label_visibility="collapsed", key="apik")
    with ai3:
        ti = st.selectbox("Variable objetivo:", pi,
            label_visibility="collapsed", key="iat")

    # ── Helper: construir prompt ───────────────────────────
    def _build_prompt():
        stats_txt = df[info_cols["numericas"]].describe().round(3).to_string() if info_cols["numericas"] else "N/A"
        return (
            f"Eres un experto en estadística bayesiana. Analiza este dataset y responde "
            f"estructurado en 5 secciones: RESUMEN, HALLAZGOS PRINCIPALES, VARIABLES DE RIESGO, "
            f"RECOMENDACIONES y PRÓXIMOS PASOS.\n\n"
            f"Dataset: {res_df['filas']} filas, {res_df['columnas']} columnas\n"
            f"Variable objetivo: {ti}\n"
            f"Variables numéricas: {', '.join(info_cols['numericas'])}\n"
            f"Variables categóricas: {', '.join(info_cols['categoricas'])}\n"
            f"Distribución '{ti}':\n{df[ti].value_counts().to_string()}\n"
            f"Estadísticas descriptivas:\n{stats_txt}"
        )

    def _show_result(texto, modelo_nombre):
        st.markdown(section_hdr(f"Análisis — {modelo_nombre}","🤖"), unsafe_allow_html=True)
        # Formatear secciones con colores
        html = ""
        for linea in texto.split("\n"):
            l = linea.strip()
            if any(l.startswith(s) for s in ["RESUMEN","HALLAZGOS","VARIABLES","RECOMENDACIONES","PRÓXIMOS","PROXIMOS"]):
                html += f'<div style="font-family:Rajdhani,sans-serif;font-weight:700;font-size:13px;color:#FFAA00;text-transform:uppercase;letter-spacing:2px;margin:18px 0 6px;border-bottom:1px solid #28283A;padding-bottom:4px;">{l}</div>'
            elif l.startswith("•") or l.startswith("-"):
                html += f'<div style="font-family:Rajdhani,sans-serif;font-size:14px;color:#E8E8E8;padding:3px 0 3px 16px;line-height:1.6;">{l}</div>'
            elif l:
                html += f'<div style="font-family:Rajdhani,sans-serif;font-size:14px;color:#AAAAAA;padding:2px 0;line-height:1.6;">{l}</div>'
        st.markdown(f'<div class="mc-card">{html}</div>', unsafe_allow_html=True)

    # ── Botones ────────────────────────────────────────────
    c_btn1, c_btn2 = st.columns([1, 1])
    with c_btn1:
        btn_ia = st.button("🤖 Generar Insights con IA", key="btn_ia")
    with c_btn2:
        btn_auto = st.button("📊 Análisis sin API (gratis)", key="btn_auto")

    # ── Análisis con IA ────────────────────────────────────
    if btn_ia:
        if not ak:
            st.markdown(ins("⚠️ Ingresa tu API Key primero.","w"), unsafe_allow_html=True)
        else:
            prompt = _build_prompt()
            if "Gemini" in proveedor:
                with st.spinner("Consultando Gemini..."):
                    try:
                        import urllib.request, json as _json
                        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite-preview:generateContent?key={ak}"
                        body = _json.dumps({"contents":[{"parts":[{"text": prompt}]}]}).encode()
                        req  = urllib.request.Request(url, data=body,
                               headers={"Content-Type":"application/json"}, method="POST")
                        with urllib.request.urlopen(req, timeout=30) as resp:
                            data = _json.loads(resp.read())
                        texto = data["candidates"][0]["content"]["parts"][0]["text"]
                        _show_result(texto, "Gemini 3.1 Flash Lite")
                    except Exception as e:
                        st.markdown(ins(f"❌ Error Gemini: {e}","d"), unsafe_allow_html=True)
                        st.markdown(ins("Verifica tu API Key en <a href='https://aistudio.google.com/apikey' target='_blank' style='color:#4DD9D9;'>aistudio.google.com/apikey</a>","w"), unsafe_allow_html=True)
            else:
                with st.spinner("Consultando Claude..."):
                    try:
                        import anthropic as ant
                        cl  = ant.Anthropic(api_key=ak)
                        msg = cl.messages.create(
                            model="claude-opus-4-5", max_tokens=1500,
                            messages=[{"role":"user","content": prompt}])
                        _show_result(msg.content[0].text, "Claude (Anthropic)")
                    except ImportError:
                        st.markdown(ins("Instala: pip install anthropic","d"), unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error Claude: {e}")

    # ── Análisis automático SIN API ────────────────────────
    if btn_auto:
        from sklearn.preprocessing import LabelEncoder
        ta  = ti if ti in df.columns else df.columns[0]
        da  = df[ta].value_counts()
        pe  = da.iloc[0] / len(df) if len(da) > 0 else 0
        ts  = df[ta].copy()
        if ts.dtype == "object":
            le = LabelEncoder(); ts = pd.Series(le.fit_transform(ts.astype(str)))

        # Correlaciones
        cr = {c: round(df[c].corr(ts), 4) for c in info_cols["numericas"] if c in df.columns}
        dc = pd.DataFrame(list(cr.items()), columns=["Variable","Correlación"]) if cr else pd.DataFrame()
        if not dc.empty:
            dc["Abs"] = dc["Correlación"].abs()
            dc = dc.sort_values("Abs", ascending=False).drop("Abs", axis=1)
            dc["Nivel"] = dc["Correlación"].apply(
                lambda x: "🔴 Alta" if abs(x)>.5 else "🟡 Media" if abs(x)>.2 else "🟢 Baja")

        # Mejor variable predictora
        mejor = dc.iloc[0]["Variable"] if not dc.empty else "N/A"
        mejor_corr = dc.iloc[0]["Correlación"] if not dc.empty else 0

        st.markdown(section_hdr("Análisis Automático","📊"), unsafe_allow_html=True)
        m1,m2,m3 = st.columns(3)
        m1.metric("Registros totales", f"{res_df['filas']:,}")
        m2.metric(f"P({ta}=más común)", f"{pe*100:.1f}%")
        m3.metric("Mejor predictor", mejor, f"r={mejor_corr:.3f}")

        st.markdown(ins(
            f"<b>Variable '{ta}':</b> {da.to_dict()}<br>"
            f"<b>P(evento principal):</b> {pe:.4f} ({pe*100:.2f}%)<br>"
            f"<b>Nulos en dataset:</b> {res_df['nulos']}", "s"
        ), unsafe_allow_html=True)

        if not dc.empty:
            st.markdown(section_hdr("Correlaciones con objetivo","🔗"), unsafe_allow_html=True)
            st.dataframe(dc, use_container_width=True)
            # Insight automático
            altas = dc[dc["Correlación"].abs() > 0.5]["Variable"].tolist()
            if altas:
                st.markdown(ins(f"⚡ Variables con correlación ALTA: <b>{', '.join(altas)}</b> — son las mejores candidatas para el análisis bayesiano.","s"), unsafe_allow_html=True)
            else:
                st.markdown(ins("🟡 Ninguna variable supera correlación 0.5 — prueba el Teorema de Bayes con cada variable para encontrar la más predictiva.","w"), unsafe_allow_html=True)

st.markdown(get_footer(),unsafe_allow_html=True)
