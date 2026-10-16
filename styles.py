"""Custom look: CSS and small HTML components.

Colors that must work in both light and dark themes are mixed from `currentColor`
(the theme's text color), so cards and borders adapt automatically when the viewer
switches theme.
"""

from __future__ import annotations

import html

import streamlit as st

GRADIENT = "linear-gradient(135deg, #b93a0c 0%, #ea580c 48%, #f59e0b 100%)"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,500,0,0&display=block');

.ms {{ font-family: 'Material Symbols Rounded'; font-weight: normal; font-style: normal; font-size: 1.3rem; line-height: 1;
  letter-spacing: normal; text-transform: none; white-space: nowrap; direction: ltr; font-feature-settings: 'liga';
  -webkit-font-smoothing: antialiased; display: inline-block; }}

:root {{
  --tint-1: color-mix(in srgb, currentColor 4%, transparent);
  --tint-2: color-mix(in srgb, currentColor 8%, transparent);
  --line: color-mix(in srgb, currentColor 13%, transparent);
  --muted: color-mix(in srgb, currentColor 62%, transparent);
  --shadow: 0 1px 2px rgba(0,0,0,.06), 0 8px 24px -8px rgba(15,23,42,.18);
  --shadow-hover: 0 2px 4px rgba(0,0,0,.08), 0 18px 40px -12px rgba(234,88,12,.35);
}}

.block-container {{ padding-top: 5rem; max-width: 1280px; }}
h1, h2, h3, h4 {{ font-family: 'Sora', sans-serif !important; letter-spacing: -0.02em; }}

/* ---------- hero banner ---------- */
.hero {{
  position: relative; overflow: hidden; border-radius: 24px; padding: 40px 44px;
  background: {GRADIENT}; color: #fff; box-shadow: 0 24px 60px -24px rgba(234,88,12,.65);
  margin-bottom: 28px;
}}
.hero::before, .hero::after {{
  content: ""; position: absolute; border-radius: 50%; filter: blur(2px);
  background: radial-gradient(circle, rgba(255,255,255,.28), rgba(255,255,255,0) 70%);
}}
.hero::before {{ width: 420px; height: 420px; right: -120px; top: -180px; }}
.hero::after  {{ width: 300px; height: 300px; right: 220px; bottom: -200px; opacity: .6; }}
.hero .eyebrow {{ font: 600 .78rem 'Inter', sans-serif; letter-spacing: .14em; text-transform: uppercase; opacity: .85; }}
.hero h1 {{ color: #fff !important; font-size: 2.6rem; font-weight: 800; line-height: 1.1; margin: 10px 0 12px; padding: 0;
  text-shadow: 0 2px 12px rgba(0,0,0,.18); }}
.hero p {{ font-size: 1.08rem; max-width: 640px; opacity: .92; margin: 0; }}
.hero .chips {{ margin-top: 22px; display: flex; gap: 8px; flex-wrap: wrap; position: relative; z-index: 1; }}
.hero .chip {{ background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.28); backdrop-filter: blur(6px);
  padding: 6px 14px; border-radius: 999px; font-size: .85rem; font-weight: 500; }}

/* ---------- page header ---------- */
.page-head {{ margin: 4px 0 22px; }}
.page-head h2 {{ font-size: 2rem; font-weight: 800; margin: 0; padding: 0;
  background: {GRADIENT}; -webkit-background-clip: text; background-clip: text; color: transparent !important; }}
.page-head p {{ color: var(--muted); margin: 6px 0 0; font-size: 1.02rem; }}

/* ---------- cards ---------- */
.card {{
  background: var(--tint-1); border: 1px solid var(--line); border-radius: 18px; padding: 20px 22px;
  box-shadow: var(--shadow); transition: transform .18s ease, box-shadow .18s ease; height: 100%;
}}
.card:hover {{ transform: translateY(-3px); box-shadow: var(--shadow-hover); }}
.kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 16px; margin: 6px 0 22px; }}
.kpi .icon {{ width: 40px; height: 40px; border-radius: 12px; display: grid; place-items: center; font-size: 1.15rem;
  background: {GRADIENT}; color: #fff; box-shadow: 0 8px 18px -8px rgba(234,88,12,.8); margin-bottom: 14px; }}
.kpi .label {{ font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); font-weight: 600; }}
.kpi {{ min-width: 0; }}
.kpi .value {{ font: 700 1.9rem 'Sora', sans-serif; letter-spacing: -0.02em; margin-top: 2px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.kpi .value.small {{ font-size: 1.3rem; padding: 6px 0 4px; }}
.kpi .sub {{ font-size: .85rem; color: var(--muted); margin-top: 2px; }}

.step {{ display: flex; flex-direction: column; gap: 6px; }}
.step .num {{ font: 800 .8rem 'Sora', sans-serif; color: #ea580c; letter-spacing: .1em; }}
.step h4 {{ margin: 0; padding: 0; font-size: 1.15rem; }}
.step p {{ margin: 0; color: var(--muted); font-size: .95rem; }}

/* ---------- model cards ---------- */
.model-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin: 18px 0 26px; }}
.model {{ position: relative; }}
.model .name {{ display: flex; align-items: center; gap: 10px; font: 600 1.02rem 'Sora', sans-serif; }}
.model .dot {{ width: 12px; height: 12px; border-radius: 50%; box-shadow: 0 0 0 4px color-mix(in srgb, currentColor 8%, transparent); }}
.model .score {{ font: 800 2.3rem 'Sora', sans-serif; letter-spacing: -0.03em; margin: 12px 0 0; }}
.model .score-label {{ font-size: .8rem; color: var(--muted); }}
.model .mini {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; margin-top: 16px; padding-top: 14px; border-top: 1px solid var(--line); }}
.model .mini div {{ font-size: .72rem; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }}
.model .mini b {{ display: block; font: 600 .98rem 'Inter', sans-serif; color: var(--fg, inherit); text-transform: none; letter-spacing: 0; }}
.badge {{ display: inline-block; margin-bottom: 12px; background: {GRADIENT}; color: #fff; font-size: .72rem; font-weight: 700;
  padding: 5px 11px; border-radius: 999px; letter-spacing: .04em; box-shadow: 0 6px 16px -6px rgba(234,88,12,.8); }}
.badge.ghost {{ visibility: hidden; }}
.model.winner {{ border: 1.5px solid #ea580c; box-shadow: 0 18px 44px -18px rgba(234,88,12,.6); }}

/* ---------- recommendation ---------- */
.reco {{ position: relative; border-radius: 22px; padding: 2px; background: {GRADIENT}; box-shadow: 0 24px 60px -28px rgba(234,88,12,.7); margin-bottom: 8px; }}
.reco-inner {{ border-radius: 20px; padding: 26px 30px; background: var(--reco-bg, rgba(10,15,31,.0)); backdrop-filter: blur(0); }}
.reco .eyebrow {{ font: 700 .75rem 'Inter', sans-serif; letter-spacing: .14em; text-transform: uppercase; color: #fff; opacity: .9; }}
.reco h3 {{ color: #fff !important; font-size: 1.9rem; margin: 6px 0 4px; padding: 0; }}
.reco .lead {{ color: rgba(255,255,255,.92); font-size: 1.02rem; margin: 0; }}
.reco ul {{ margin: 16px 0 0; padding-left: 0; list-style: none; display: grid; gap: 8px; }}
.reco li {{ color: #fff; background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.18); border-radius: 12px; padding: 10px 14px; font-size: .95rem; }}
.reco li b {{ color: #fff; }}

/* ---------- alerts as soft chips ---------- */
.note {{ display: flex; gap: 10px; align-items: flex-start; border-radius: 14px; padding: 12px 16px; margin: 0 0 10px;
  background: color-mix(in srgb, #f59e0b 12%, transparent); border: 1px solid color-mix(in srgb, #f59e0b 35%, transparent); font-size: .95rem; }}
.note.info {{ background: var(--tint-2); border-color: var(--line); }}

/* ---------- streamlit widgets ---------- */
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"], [data-testid^="stBaseButton-primary"] {{
  background: {GRADIENT} !important; border: 0 !important; color: #fff !important; font-weight: 600;
  box-shadow: 0 10px 24px -10px rgba(234,88,12,.9); transition: transform .15s ease, box-shadow .15s ease;
}}
.stButton > button[kind="primary"]:hover, [data-testid^="stBaseButton-primary"]:hover {{ transform: translateY(-1px); box-shadow: 0 14px 30px -10px rgba(234,88,12,1); }}
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid var(--line); }}
.stTabs [data-baseweb="tab"] {{ border-radius: 10px 10px 0 0; padding: 8px 16px; font-weight: 600; }}
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) {{ border-radius: 18px; }}
[data-testid="stExpander"] details {{ border-radius: 14px; }}
[data-testid="stPlotlyChart"] {{ border: 1px solid var(--line); border-radius: 18px; padding: 8px 10px; background: var(--tint-1); }}

/* ---------- sidebar profile ---------- */
.profile {{ display: flex; gap: 12px; align-items: center; padding: 14px; border-radius: 16px; background: var(--tint-2); border: 1px solid var(--line); }}
.avatar {{ width: 42px; height: 42px; border-radius: 14px; display: grid; place-items: center; color: #fff; font: 700 1rem 'Sora', sans-serif; background: {GRADIENT}; }}
.profile .who {{ font-weight: 600; line-height: 1.2; }}
.profile .mail {{ font-size: .8rem; color: var(--muted); word-break: break-all; }}
.side-label {{ font-size: .72rem; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); font-weight: 700; margin: 18px 0 6px; }}

/* ---------- sign-in ---------- */
.auth-panel {{ border-radius: 26px; padding: 44px 40px; color: #fff; background: {GRADIENT}; position: relative; overflow: hidden;
  box-shadow: 0 30px 70px -30px rgba(234,88,12,.8); min-height: 520px; }}
.auth-panel::after {{ content: ""; position: absolute; width: 380px; height: 380px; right: -140px; bottom: -140px; border-radius: 50%;
  background: radial-gradient(circle, rgba(255,255,255,.3), rgba(255,255,255,0) 70%); }}
.auth-panel .logo {{ font: 800 1.1rem 'Sora', sans-serif; display: flex; align-items: center; gap: 10px; }}
.auth-panel .logo span {{ width: 34px; height: 34px; border-radius: 10px; background: rgba(255,255,255,.2); display: grid; place-items: center; }}
.auth-panel h1 {{ color: #fff !important; font-size: 2.4rem; font-weight: 800; line-height: 1.12; margin: 46px 0 14px; padding: 0; }}
.auth-panel p {{ opacity: .9; font-size: 1.02rem; margin: 0 0 26px; }}
.auth-panel .feat {{ display: flex; gap: 12px; align-items: center; margin: 12px 0; font-size: .98rem; position: relative; z-index: 1; }}
.auth-panel .feat i {{ font-style: normal; width: 30px; height: 30px; border-radius: 9px; background: rgba(255,255,255,.18); display: grid; place-items: center; }}
.auth-title {{ font: 800 1.8rem 'Sora', sans-serif; letter-spacing: -0.02em; margin: 8px 0 2px; }}
.auth-sub {{ color: var(--muted); margin: 0 0 18px; }}

@media (max-width: 640px) {{
  .hero {{ padding: 28px 22px; }} .hero h1 {{ font-size: 1.9rem; }}
  .auth-panel {{ min-height: 0; padding: 30px 24px; }} .auth-panel h1 {{ font-size: 1.8rem; margin-top: 24px; }}
}}
</style>
"""


def inject() -> None:
    st.html(CSS)


def esc(text) -> str:
    return html.escape(str(text))


def hero(eyebrow: str, title: str, subtitle: str, chips: list[str] | None = None) -> None:
    chip_html = "".join(f'<span class="chip">{esc(c)}</span>' for c in chips or [])
    st.html(
        f'<div class="hero"><div class="eyebrow">{esc(eyebrow)}</div><h1>{esc(title)}</h1>'
        f'<p>{esc(subtitle)}</p><div class="chips">{chip_html}</div></div>'
    )


def page_header(title: str, subtitle: str) -> None:
    st.html(f'<div class="page-head"><h2>{esc(title)}</h2><p>{esc(subtitle)}</p></div>')


def kpis(items: list[tuple[str, str, str, str]]) -> None:
    """items: (Material Symbols icon name, label, value, sub)."""
    cards = "".join(
        f'<div class="card kpi"><div class="icon">{icon(sym)}</div><div class="label">{esc(label)}</div>'
        f'<div class="value{" small" if len(str(value)) > 9 else ""}" title="{esc(value)}">{esc(value)}</div>'
        f'<div class="sub">{esc(sub)}</div></div>'
        for sym, label, value, sub in items
    )
    st.html(f'<div class="kpi-grid">{cards}</div>')


def icon(name: str) -> str:
    """A Material Symbols icon, e.g. icon("folder_open")."""
    return f'<span class="ms">{esc(name)}</span>'


def note(text_html: str, kind: str = "warn") -> None:
    sym = icon("warning" if kind == "warn" else "lightbulb")
    cls = "note" if kind == "warn" else "note info"
    st.html(f'<div class="{cls}">{sym}<div>{text_html}</div></div>')


def step_card(num: str, title: str, text: str) -> None:
    st.html(f'<div class="card step"><div class="num">{esc(num)}</div><h4>{esc(title)}</h4><p>{esc(text)}</p></div>')


def model_cards(cards: list[dict]) -> None:
    """Each dict: name, color, score, score_label, mini (list of (label, value)), winner (bool)."""
    out = []
    for c in cards:
        mini = "".join(f"<div>{esc(lbl)}<b>{esc(val)}</b></div>" for lbl, val in c["mini"])
        badge = f'<span class="badge{"" if c["winner"] else " ghost"}">RECOMMENDED</span>'
        out.append(
            f'<div class="card model{" winner" if c["winner"] else ""}">{badge}'
            f'<div class="name"><span class="dot" style="background:{c["color"]}"></span>{esc(c["name"])}</div>'
            f'<div class="score">{esc(c["score"])}</div><div class="score-label">{esc(c["score_label"])}</div>'
            f'<div class="mini">{mini}</div></div>'
        )
    st.html(f'<div class="model-grid">{"".join(out)}</div>')


def recommendation(model: str, lead: str, points_html: list[str]) -> None:
    items = "".join(f"<li>{p}</li>" for p in points_html)
    st.html(
        f'<div class="reco"><div class="reco-inner"><div class="eyebrow">Recommended model</div>'
        f'<h3>{esc(model)}</h3><p class="lead">{lead}</p><ul>{items}</ul></div></div>'
    )


def sidebar_profile(name: str, email: str) -> None:
    initials = "".join(part[0] for part in name.split()[:2]).upper() or "?"
    st.html(
        f'<div class="profile"><div class="avatar">{esc(initials)}</div>'
        f'<div><div class="who">{esc(name)}</div><div class="mail">{esc(email)}</div></div></div>'
    )


def auth_panel() -> None:
    st.html(
        """
        <div class="auth-panel">
          <div class="logo"><span><span class="ms">science</span></span>Model Comparison Lab</div>
          <h1>Find the best model for your data, with the reasons why.</h1>
          <p>Upload a CSV, choose what to predict, and let three machine-learning models compete
             under cross-validation.</p>
          <div class="feat"><i><span class="ms">monitoring</span></i>Automatic data profiling &amp; missing-value checks</div>
          <div class="feat"><i><span class="ms">tune</span></i>Tuned Logistic Regression, Random Forest &amp; Gradient Boosting</div>
          <div class="feat"><i><span class="ms">query_stats</span></i>ROC curves, confusion matrices &amp; feature importance</div>
          <div class="feat"><i><span class="ms">workspace_premium</span></i>A clear recommendation backed by statistics</div>
        </div>
        """
    )
