"""
minecraft_theme.py — BayesCraft v3.5
Partículas: CSS puro — SIN JavaScript, funciona siempre en Streamlit.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import numpy as np
import base64, shutil
from pathlib import Path

# ── Paleta ──────────────────────────────────────────────
MC = {
    "grass":"#5D8731","glt":"#7AB840","gdk":"#2D4518",
    "dirt":"#8B5E3C","stone":"#888888","slt":"#AAAAAA",
    "gold":"#FFAA00","gld":"#FFD060",
    "diamond":"#4DD9D9","em":"#1AE365","red":"#FF3030","lapis":"#1E5CCC",
    "d1":"#0C0C12","d2":"#141418","d3":"#1C1C22","d4":"#242430","panel":"#080810",
    "text":"#E8E8E8","dim":"#7A7A8A","b":"#28283A","blt":"#383848",
}
CHART_COLORS = ["#5D8731","#FFAA00","#4DD9D9","#FF3030","#1AE365",
                "#1E5CCC","#8B5E3C","#CC7722","#AA44CC","#888888"]
MIME_MAP = {
    ".png":"image/png",".jpg":"image/jpeg",".jpeg":"image/jpeg",
    ".gif":"image/gif",".webp":"image/webp",
    ".mp4":"video/mp4",".webm":"video/webm",
}

def _root(): return Path(__file__).parent

def prepare_static_files():
    md = _root()/"media"; sd = _root()/"static"
    md.mkdir(exist_ok=True); sd.mkdir(exist_ok=True)
    for f in md.iterdir():
        if f.suffix.lower() in MIME_MAP:
            try: shutil.copy2(f, sd/f.name)
            except: pass

def load_media_info():
    """
    Carga media/ como base64 — funciona en local Y en Streamlit Cloud.
    Videos > 50 MB se omiten para no saturar memoria.
    """
    md = _root()/"media"; md.mkdir(exist_ok=True); result=[]
    MAX_VIDEO_BYTES = 50 * 1024 * 1024  # 50 MB límite
    for f in sorted(md.iterdir()):
        suf=f.suffix.lower()
        if suf not in MIME_MAP or f.name.endswith(".txt"): continue
        mime=MIME_MAP[suf]; is_v=mime.startswith("video")
        try:
            size = f.stat().st_size
            if is_v and size > MAX_VIDEO_BYTES:
                # Video demasiado grande — usar static serving como fallback
                result.append({"name":f.name,"mime":mime,"is_video":True,
                               "url":f"/app/static/{f.name}","b64":None})
                continue
            b64 = base64.b64encode(f.read_bytes()).decode()
            result.append({"name":f.name,"mime":mime,"is_video":is_v,
                           "url":None,"b64":b64})
        except: pass
    return result

# ── matplotlib ──────────────────────────────────────────
def apply_mc_style(fig, axes=None):
    fig.patch.set_facecolor(MC["d2"])
    if axes is None: return
    if isinstance(axes, np.ndarray): ax_list=axes.flatten().tolist()
    elif isinstance(axes,(list,tuple)):
        flat=[]
        for item in axes:
            if isinstance(item,np.ndarray): flat.extend(item.flatten().tolist())
            elif hasattr(item,"get_facecolor"): flat.append(item)
            else:
                try: flat.extend(list(item))
                except: flat.append(item)
        ax_list=flat
    else: ax_list=[axes]
    for ax in ax_list:
        if ax is None or not hasattr(ax,"set_facecolor"): continue
        try:
            ax.set_facecolor(MC["d3"]); ax.tick_params(colors=MC["dim"],labelsize=7)
            ax.xaxis.label.set_color(MC["slt"]); ax.yaxis.label.set_color(MC["slt"])
            ax.title.set_color(MC["gold"]); ax.title.set_fontsize(9)
            for sp in ax.spines.values(): sp.set_edgecolor(MC["b"]); sp.set_linewidth(1.2)
            ax.grid(True,color="#1E1E2C",linewidth=0.7,linestyle="--",alpha=0.7); ax.set_axisbelow(True)
        except: pass

def mc_cmap(name="green"):
    maps={"green":[MC["d1"],MC["gdk"],MC["grass"],MC["gld"]],
          "diverging":[MC["lapis"],MC["d1"],MC["gold"]],"hot":[MC["d1"],MC["red"],MC["gold"]]}
    return LinearSegmentedColormap.from_list(f"mc_{name}",maps.get(name,maps["green"]))

# ═══════════════════════════════════════════════════════════════════
# CSS PRINCIPAL
# ═══════════════════════════════════════════════════════════════════
def get_css():
    return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323&family=Rajdhani:wght@400;600;700&display=swap');
:root{
  --d1:#0C0C12;--d2:#141418;--d3:#1C1C22;--panel:#080810;
  --grass:#5D8731;--glt:#7AB840;--gdk:#2D4518;
  --gold:#FFAA00;--dia:#4DD9D9;--em:#1AE365;--red:#FF3030;
  --text:#E8E8E8;--dim:#7A7A8A;--b:#28283A;
  --px:'Press Start 2P',monospace;--vt:'VT323',monospace;--bd:'Rajdhani',sans-serif;
  --nav-h:52px;
  --ng:0 0 8px rgba(93,135,49,.7),0 0 22px rgba(93,135,49,.3);
  --ng2:0 0 14px rgba(122,184,64,.95),0 0 36px rgba(93,135,49,.5);
  --ngo:0 0 8px rgba(255,170,0,.55),0 0 22px rgba(255,170,0,.25);
  --ngc:0 0 8px rgba(77,217,217,.55),0 0 22px rgba(77,217,217,.22);
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
html,body,[data-testid="stAppViewContainer"]{background:#0C0C12!important;}
.stApp{background:transparent!important;font-family:var(--bd)!important;color:var(--text)!important;}

/* Ocultar chrome Streamlit */
#MainMenu,footer,header,[data-testid="stToolbar"],[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],.stDeployButton,[data-testid="stHeader"],[data-testid="stDecoration"]{display:none!important;}
[data-testid="stSidebar"]{display:none!important;width:0!important;min-width:0!important;}
.main .block-container{padding:0!important;max-width:100%!important;}
section[data-testid="stMain"]{padding:0!important;}

/* ── PARTÍCULAS CSS (posición fija, z-index bajo) ────── */
@keyframes mcfall{
  from{transform:translateY(-70px) rotate(0deg);opacity:0;}
  8%{opacity:var(--a);}
  92%{opacity:var(--a);}
  to{transform:translateY(110vh) rotate(720deg);opacity:0;}
}
.mc-p{
  position:fixed;pointer-events:none;z-index:1;
  width:var(--s);height:var(--s);background:var(--c);
  left:var(--x);top:0;
  animation:mcfall var(--d) linear var(--dl) infinite;
}
.mc-p::after{content:'';position:absolute;top:0;left:0;width:100%;height:30%;background:rgba(255,255,255,.13);}
.mc-p::before{content:'';position:absolute;top:0;left:0;width:20%;height:100%;background:rgba(255,255,255,.08);}
.mc-star{
  position:fixed;pointer-events:none;z-index:1;
  width:1.5px;height:1.5px;border-radius:50%;
  background:rgba(255,255,255,.22);
}

/* ── TABS ─────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"]{background:var(--panel)!important;
  border-bottom:1px solid var(--b)!important;gap:0!important;padding:0 24px!important;}
.stTabs [data-baseweb="tab"]{font-family:var(--bd)!important;font-weight:700!important;
  font-size:13px!important;text-transform:uppercase!important;letter-spacing:1.5px!important;
  color:var(--dim)!important;background:transparent!important;border:none!important;
  border-bottom:3px solid transparent!important;padding:13px 20px!important;transition:all .2s!important;}
.stTabs [data-baseweb="tab"]:hover{color:var(--text)!important;}
.stTabs [aria-selected="true"]{color:#fff!important;border-bottom:3px solid var(--grass)!important;text-shadow:var(--ng)!important;}
.stTabs [data-baseweb="tab-panel"]{background:transparent!important;padding:24px!important;}

/* ── BOTONES ─────────────────────────────────────────── */
.stButton>button{
  font-family:var(--bd)!important;font-weight:700!important;font-size:14px!important;
  letter-spacing:2px!important;text-transform:uppercase!important;
  background:linear-gradient(180deg,#6aaa34 0%,#5D8731 45%,#4a6d27 100%)!important;
  color:#fff!important;border:none!important;border-radius:2px!important;
  padding:11px 20px!important;cursor:pointer!important;
  box-shadow:0 4px 0 var(--gdk),var(--ng),0 4px 14px rgba(0,0,0,.5)!important;
  text-shadow:0 1px 2px rgba(0,0,0,.6)!important;transition:all .15s!important;width:100%!important;}
.stButton>button:hover{
  background:linear-gradient(180deg,#7bc940 0%,#6aaa34 45%,#5D8731 100%)!important;
  transform:translateY(-2px)!important;box-shadow:0 6px 0 var(--gdk),var(--ng2),0 8px 20px rgba(0,0,0,.6)!important;}
.stButton>button:active{transform:translateY(2px)!important;box-shadow:0 2px 0 var(--gdk)!important;}

/* ── MÉTRICAS ────────────────────────────────────────── */
[data-testid="stMetric"]{background:var(--d3)!important;border:1px solid var(--b)!important;
  border-top:2px solid var(--grass)!important;padding:16px!important;border-radius:2px!important;
  box-shadow:0 0 12px rgba(93,135,49,.1)!important;}
[data-testid="stMetricLabel"]{font-family:var(--bd)!important;font-weight:700!important;
  font-size:10px!important;text-transform:uppercase!important;letter-spacing:1.5px!important;color:var(--dim)!important;}
[data-testid="stMetricValue"]{font-family:var(--vt)!important;font-size:38px!important;
  color:var(--gold)!important;line-height:1.1!important;}

/* ── FILE UPLOADER ───────────────────────────────────── */
[data-testid="stFileUploader"]{background:rgba(8,8,16,.72)!important;
  border:2px dashed rgba(93,135,49,.5)!important;border-radius:4px!important;
  padding:10px!important;backdrop-filter:blur(12px)!important;transition:all .25s!important;}
[data-testid="stFileUploader"]:hover{border-color:var(--glt)!important;box-shadow:var(--ng)!important;}
[data-testid="stFileUploader"] *{font-family:var(--bd)!important;font-size:13px!important;color:var(--dim)!important;}
[data-testid="stFileUploader"] button{background:rgba(93,135,49,.15)!important;
  color:var(--glt)!important;border:1px solid rgba(93,135,49,.6)!important;
  font-weight:700!important;padding:8px 20px!important;border-radius:2px!important;
  font-family:var(--bd)!important;box-shadow:var(--ng)!important;}

/* ── SELECTBOX ───────────────────────────────────────── */
.stSelectbox label,.stMultiSelect label,.stSlider label,.stTextInput label{
  font-family:var(--bd)!important;font-weight:700!important;font-size:10px!important;
  text-transform:uppercase!important;letter-spacing:1px!important;color:var(--dim)!important;}
.stSelectbox>div>div,.stMultiSelect>div>div{background:rgba(8,8,16,.8)!important;
  border:1px solid var(--b)!important;border-radius:2px!important;
  color:var(--text)!important;font-family:var(--bd)!important;font-size:13px!important;}
.stSelectbox>div>div:focus-within{border-color:var(--grass)!important;box-shadow:var(--ng)!important;}

/* ── DATAFRAME ───────────────────────────────────────── */
.stDataFrame{border:1px solid var(--b)!important;border-radius:2px!important;}
.stDataFrame thead th{background:var(--d3)!important;color:var(--dim)!important;
  font-family:var(--bd)!important;font-weight:700!important;font-size:10px!important;
  text-transform:uppercase!important;letter-spacing:1px!important;}
.stDataFrame tbody td{font-family:var(--bd)!important;font-size:13px!important;
  background:var(--d2)!important;color:var(--text)!important;border-color:var(--b)!important;}

/* Scrollbar */
::-webkit-scrollbar{width:7px;height:7px;}
::-webkit-scrollbar-track{background:var(--d1);}
::-webkit-scrollbar-thumb{background:#2a2a3a;border-radius:4px;}
::-webkit-scrollbar-thumb:hover{background:var(--grass);box-shadow:var(--ng);}
.stProgress>div>div>div{background:var(--grass)!important;box-shadow:var(--ng)!important;}

/* ── NAVBAR ──────────────────────────────────────────── */
.mc-nav{position:fixed;top:0;left:0;right:0;height:var(--nav-h);z-index:1000;
  background:rgba(8,8,16,.95);border-bottom:1px solid rgba(40,40,58,.8);
  backdrop-filter:blur(14px);display:flex;align-items:center;padding:0 28px;gap:20px;}
.mc-nav-logo{font-family:var(--px);font-size:11px;color:var(--gold);text-shadow:var(--ngo);white-space:nowrap;}
.mc-nav-ver{font-family:var(--vt);font-size:20px;color:#252535;margin-left:auto;}
.mc-nav-sep{font-family:Rajdhani,sans-serif;font-size:11px;color:#1e1e2e;text-transform:uppercase;
  letter-spacing:2px;border-left:1px solid #1a1a28;padding-left:16px;}

/* ── HERO ────────────────────────────────────────────── */
.mc-hero{position:relative;width:100%;height:calc(100vh - var(--nav-h));
  overflow:hidden;background:var(--d1);}
.mc-hero-slide{position:absolute;inset:0;background-size:cover;background-position:center;
  opacity:0;transition:opacity 1.4s ease;z-index:2;}
.mc-hero-slide video{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;}
.mc-hero-slide.on{opacity:1;}
.mc-hero-overlay{position:absolute;inset:0;z-index:3;
  background:linear-gradient(to bottom,rgba(8,8,16,.4) 0%,rgba(8,8,16,.08) 30%,rgba(8,8,16,.08) 55%,rgba(8,8,16,.97) 100%);}
.mc-hero-content{position:absolute;inset:0;z-index:4;
  display:flex;flex-direction:column;align-items:center;justify-content:center;padding:0 20px;}
.mc-hero-title{font-family:var(--px);font-size:clamp(22px,3.5vw,44px);
  color:var(--gold);text-shadow:4px 4px 0 #5a3a00,var(--ngo);
  letter-spacing:3px;margin-bottom:12px;text-align:center;line-height:1.4;}
.mc-hero-sub{font-family:var(--bd);font-weight:600;font-size:13px;
  color:rgba(255,255,255,.42);text-transform:uppercase;letter-spacing:5px;text-align:center;}
.mc-dots{position:absolute;bottom:28px;left:50%;transform:translateX(-50%);
  z-index:5;display:flex;gap:8px;}
.mc-dot{width:9px;height:9px;border-radius:50%;background:rgba(255,255,255,.2);
  cursor:pointer;border:none;padding:0;transition:all .3s;}
.mc-dot.on{background:var(--glt);transform:scale(1.5);box-shadow:var(--ng2);}

/* Franja pasto */
.mc-grass{width:100%;height:18px;
  background:linear-gradient(180deg,var(--glt) 0%,var(--grass) 45%,var(--gdk) 100%);
  box-shadow:0 0 22px rgba(122,184,64,.55),0 0 45px rgba(93,135,49,.25);}

/* Sticky bar */
.mc-sticky{position:fixed;top:var(--nav-h);left:0;right:0;z-index:998;
  background:linear-gradient(180deg,rgba(6,6,18,.97) 0%,rgba(6,6,18,.88) 70%,rgba(6,6,18,0) 100%);
  backdrop-filter:blur(14px);padding:10px 20px 28px;
  transform:translateY(-110%);transition:transform .35s cubic-bezier(.4,0,.2,1);pointer-events:none;}
.mc-sticky.visible{transform:translateY(0);pointer-events:all;}

/* Componentes */
.mc-card{background:var(--d3);border:1px solid var(--b);border-top:2px solid var(--grass);
  padding:20px 22px;margin:8px 0;border-radius:2px;box-shadow:0 0 14px rgba(93,135,49,.08);}
.mc-formula{background:var(--d1);border:1px solid #222230;border-left:4px solid var(--gold);
  box-shadow:var(--ngo);padding:16px 22px;margin:14px 0;font-family:var(--vt);font-size:24px;
  color:var(--gold);letter-spacing:1px;line-height:1.5;text-align:center;}
.mc-ins{background:rgba(93,135,49,.07);border-left:3px solid var(--grass);
  box-shadow:inset 3px 0 0 rgba(93,135,49,.35);
  padding:13px 17px;margin:10px 0;font-family:var(--bd);font-size:14px;color:var(--text);line-height:1.7;}
.mc-ins.d{background:rgba(255,48,48,.07);border-left-color:var(--red);box-shadow:inset 3px 0 0 rgba(255,48,48,.35);}
.mc-ins.w{background:rgba(255,170,0,.07);border-left-color:var(--gold);box-shadow:inset 3px 0 0 rgba(255,170,0,.35);}
.mc-ins.s{background:rgba(26,163,78,.07);border-left-color:var(--em);box-shadow:inset 3px 0 0 rgba(26,163,78,.35);}
.mb{display:inline-block;padding:3px 10px;border-radius:2px;font-family:var(--bd);
  font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin:2px;}
.mb-g{background:rgba(93,135,49,.14);border:1px solid var(--grass);color:var(--glt);box-shadow:var(--ng);}
.mb-o{background:rgba(255,170,0,.11);border:1px solid var(--gold);color:var(--gold);}
.mb-c{background:rgba(77,217,217,.11);border:1px solid var(--dia);color:var(--dia);box-shadow:var(--ngc);}
.mb-r{background:rgba(255,48,48,.11);border:1px solid var(--red);color:var(--red);}
.mb-s{background:rgba(136,136,136,.09);border:1px solid #555;color:#aaa;}
.mpt{width:100%;border-collapse:collapse;font-family:var(--bd);font-size:14px;margin:12px 0;}
.mpt th{background:var(--d2);color:var(--dim);font-weight:700;font-size:10px;
  text-transform:uppercase;letter-spacing:1px;padding:10px 16px;text-align:left;border-bottom:2px solid var(--b);}
.mpt td{padding:10px 16px;border-bottom:1px solid var(--b);color:var(--text);}
.mpt tr:hover td{background:rgba(255,255,255,.02);}
.mpt .hl{color:var(--gold);font-weight:700;font-size:16px;text-shadow:var(--ngo);}
.mcb{margin:7px 0;}
.mcb-lbl{font-family:var(--bd);font-size:11px;color:var(--dim);text-transform:uppercase;
  letter-spacing:1px;margin-bottom:5px;display:flex;justify-content:space-between;}
.mcb-lbl span{color:var(--gold);font-weight:700;}
.mcb-track{background:var(--d2);border:1px solid var(--b);height:17px;border-radius:2px;overflow:hidden;}
.mcb-fill{height:100%;background:linear-gradient(90deg,var(--gdk),var(--grass),var(--glt));
  position:relative;box-shadow:var(--ng);}
.mcb-fill::after{content:'';position:absolute;top:0;left:0;right:0;height:35%;background:rgba(255,255,255,.1);}
.mdiv{border:none;border-top:1px solid var(--b);margin:22px 0;}
.msec{display:flex;align-items:center;gap:14px;margin:22px 0 16px;padding-bottom:12px;border-bottom:1px solid var(--b);}
.msec-t{font-family:var(--bd);font-weight:700;font-size:13px;text-transform:uppercase;letter-spacing:2px;color:var(--text);}
.msec-l{flex:1;height:1px;background:var(--b);}
</style>
"""

# ═══════════════════════════════════════════════════════════════════
# PARTÍCULAS — 100% CSS, cero JavaScript
# ═══════════════════════════════════════════════════════════════════
def get_particles(dimmed: bool = False) -> str:
    """
    Bloques y estrellas en CSS puro con @keyframes.
    Siempre bien visibles — solo baja un poco cuando hay datos.
    """
    COLORS = [
        "#5D8731","#7AB840","#FFAA00","#4DD9D9","#1AE365",
        "#8B5E3C","#2D4518","#1E5CCC","#FF3030","#6aaa34",
        "#FFD060","#3a3a55","#1a1a28",
    ]
    # Con datos: opacidad reducida pero aun visible (0.65)
    opacity_wrap = "opacity:0.65;transition:opacity 1.2s;" if dimmed else "opacity:1;"

    html = f'<div style="position:fixed;inset:0;z-index:1;pointer-events:none;{opacity_wrap}">'

    # ── Estrellas ────────────────────────────────────────
    for i in range(60):
        x = (i * 137.508) % 100
        y = (i * 97.31)  % 68
        size = 1.5 + (i % 3) * 0.8   # 1.5–3.7 px
        html += (
            f'<div style="position:fixed;left:{x:.2f}%;top:{y:.2f}vh;'
            f'width:{size:.1f}px;height:{size:.1f}px;border-radius:50%;'
            f'background:rgba(255,255,255,0.35);pointer-events:none;"></div>'
        )

    # ── Bloques cayendo ──────────────────────────────────
    for i in range(35):
        x        = (i * 137.508 + 7) % 100
        size     = 10 + (i % 6) * 4          # 10–34 px — más grandes
        duration = 8 + (i % 8) * 2.5         # 8–25 s — más rápidos
        delay    = -(i * 1.3)
        color    = COLORS[i % len(COLORS)]
        opacity  = round(0.08 + (i % 5) * 0.04, 3)  # 0.08–0.24 — más visibles

        html += (
            f'<div class="mc-p" style="'
            f'--x:{x:.2f}%;--s:{size}px;--c:{color};'
            f'--d:{duration:.1f}s;--dl:{delay:.1f}s;--a:{opacity};'
            f'"></div>'
        )

    html += '</div>'
    return html

# ═══════════════════════════════════════════════════════════════════
# NAVBAR
# ═══════════════════════════════════════════════════════════════════
def get_navbar() -> str:
    return (
        '<div class="mc-nav">'
        '<span class="mc-nav-logo">&#9935; BAYESCRAFT</span>'
        '<span class="mc-nav-sep">Estadística Bayesiana</span>'
        '<span class="mc-nav-ver">v3.5</span>'
        '</div>'
        '<div style="height:52px;"></div>'
    )

# ═══════════════════════════════════════════════════════════════════
# HERO CON CARRUSEL
# ═══════════════════════════════════════════════════════════════════
def get_hero_html(media_info: list) -> str:
    slides = ""
    dots   = ""

    if media_info:
        for i, m in enumerate(media_info):
            on = " on" if i == 0 else ""
            if m["is_video"]:
                if m["b64"]:
                    # Base64 — funciona en local Y en Streamlit Cloud
                    src = f"data:{m['mime']};base64,{m['b64']}"
                else:
                    # Fallback static (solo local con enableStaticServing)
                    src = m["url"]
                slides += (
                    f'<div class="mc-hero-slide{on}" id="hs{i}">'
                    f'<video autoplay muted loop playsinline preload="auto">'
                    f'<source src="{src}" type="{m["mime"]}"></video></div>'
                )
            else:
                b64src = f"data:{m['mime']};base64,{m['b64']}"
                slides += (
                    f'<div class="mc-hero-slide{on}" id="hs{i}" style="'
                    f'background-image:url({b64src});'
                    f'background-size:cover;background-position:center;"></div>'
                )
            dots += f'<button class="mc-dot{on}" onclick="mcSlide({i})"></button>'
    else:
        slides = (
            '<div class="mc-hero-slide on" style="'
            'background:radial-gradient(ellipse at 30% 60%,rgba(45,69,24,.75) 0%,transparent 55%),'
            'radial-gradient(ellipse at 80% 30%,rgba(30,92,204,.14) 0%,transparent 50%),'
            'linear-gradient(155deg,#06060E 0%,#0C1208 45%,#070709 100%);"></div>'
        )
        dots = '<button class="mc-dot on"></button>'

    n    = len(media_info)
    auto = f"setInterval(function(){{mcSlide((mcCur+1)%{n});}},6000);" if n > 1 else ""

    # El JS del carrusel va DENTRO del string del hero, sin concatenación Python compleja
    js = (
        "<script>"
        "var mcCur=0;"
        "function mcSlide(n){"
        "var sl=document.querySelectorAll('.mc-hero-slide');"
        "var dt=document.querySelectorAll('.mc-dot');"
        "for(var i=0;i<sl.length;i++){"
        "sl[i].style.opacity=(i===n)?'1':'0';"
        "sl[i].className='mc-hero-slide'+(i===n?' on':'');}"
        "for(var i=0;i<dt.length;i++){dt[i].className='mc-dot'+(i===n?' on':'');}"
        "mcCur=n;}"
        "mcSlide(0);"
        + auto +
        "</script>"
    )

    return (
        '<div class="mc-hero" id="mcHero">'
        + slides +
        '<div class="mc-hero-overlay"></div>'
        '<div class="mc-hero-content">'
        '<div class="mc-hero-title">&#9935; BAYESCRAFT</div>'
        '<div class="mc-hero-sub">Analizador Bayesiano Estadístico</div>'
        '</div>'
        '<div class="mc-dots">' + dots + '</div>'
        '</div>'
        + js
    )

# ═══════════════════════════════════════════════════════════════════
# STICKY JS
# ═══════════════════════════════════════════════════════════════════
def get_sticky_js() -> str:
    return (
        "<script>"
        "(function(){"
        "function init(){"
        "var hero=document.getElementById('mcHero');"
        "var bar=document.getElementById('mcStickyBar');"
        "if(!hero||!bar)return;"
        "window.addEventListener('scroll',function(){"
        "bar.classList.toggle('visible',hero.getBoundingClientRect().bottom<80);});}"
        "document.readyState==='loading'"
        "?document.addEventListener('DOMContentLoaded',init):init();"
        "setTimeout(init,1200);"
        "})();"
        "</script>"
    )

# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════
def formula_box(f):
    return f'<div class="mc-formula">{f}</div>'

def ins(t, tp=""):
    cls = f"mc-ins {tp[0]}" if tp else "mc-ins"
    return f'<div class="{cls}">{t}</div>'

def divider():
    return '<hr class="mdiv">'

def section_hdr(title, icon=""):
    return f'<div class="msec"><span class="msec-t">{icon} {title}</span><div class="msec-l"></div></div>'

def prob_table(rows):
    h = '<table class="mpt"><thead><tr><th>Probabilidad</th><th>Valor</th><th>%</th></tr></thead><tbody>'
    for lbl, v, p, hl in rows:
        c = ' class="hl"' if hl else ''
        h += f'<tr><td>{lbl}</td><td{c}>{v}</td><td{c}>{p}</td></tr>'
    return h + '</tbody></table>'

def progress_bar(pct, label=""):
    w = min(100, max(0, pct * 100))
    return (
        f'<div class="mcb">'
        f'<div class="mcb-lbl">{label}<span>{w:.1f}%</span></div>'
        f'<div class="mcb-track"><div class="mcb-fill" style="width:{w}%"></div></div>'
        f'</div>'
    )

def badge(t, tp="s"):
    return f'<span class="mb mb-{tp}">{t}</span>'

def get_footer():
    return (
        '<div style="background:#05050A;border-top:1px solid #1a1a28;'
        'padding:14px 24px;text-align:center;margin-top:40px;">'
        '<span style="font-family:Rajdhani,sans-serif;font-size:11px;'
        'color:#1e1e2e;text-transform:uppercase;letter-spacing:2px;">'
        '&#9935; BayesCraft v3.5 — Python · Streamlit · Scikit-Learn · Bayes &#9935;'
        '</span></div>'
    )
