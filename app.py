from __future__ import annotations

from datetime import datetime, timezone
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard_core import (
    SERIES_SPECS,
    calculate_snapshot,
    format_number,
    latest_date,
    load_live_data,
    make_demo_data,
    risk_label,
)

st.set_page_config(
    page_title="داشبورد اقتصاد کلان و مدیریت ریسک",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
html, body, [class*="css"] { direction: rtl; text-align: right; }
[data-testid="stSidebar"] { direction: rtl; }
.block-container { padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1500px; }
.dashboard-title { font-size: 2.0rem; font-weight: 800; margin-bottom: .2rem; }
.dashboard-subtitle { color: #64748b; margin-bottom: 1rem; }
.kpi-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:12px 0 18px; }
.kpi-card { border:1px solid rgba(148,163,184,.24); border-radius:16px; padding:16px; background:linear-gradient(145deg,rgba(255,255,255,.95),rgba(248,250,252,.9)); box-shadow:0 6px 22px rgba(15,23,42,.05); min-height:120px; }
.kpi-label { color:#64748b; font-size:.88rem; }
.kpi-value { font-size:1.65rem; font-weight:800; margin-top:7px; color:#0f172a; }
.kpi-note { color:#475569; font-size:.8rem; margin-top:7px; line-height:1.7; }
.question-card { border-right:4px solid #334155; border-radius:12px; padding:14px 16px; background:rgba(248,250,252,.92); margin:8px 0; line-height:1.9; }
.section-label { font-size:1.1rem; font-weight:800; margin-top:10px; }
.source-note { color:#64748b; font-size:.78rem; line-height:1.8; }
.badge { display:inline-block; padding:5px 10px; border-radius:999px; font-size:.8rem; font-weight:700; }
.badge-green { background:#dcfce7; color:#166534; }
.badge-yellow { background:#fef3c7; color:#92400e; }
.badge-red { background:#fee2e2; color:#991b1b; }
.badge-gray { background:#e2e8f0; color:#334155; }
@media (max-width: 900px) { .kpi-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } }
@media (max-width: 560px) { .kpi-grid { grid-template-columns:1fr; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def get_data():
    data, status = load_live_data()
    return data, status


def safe_data(use_demo: bool):
    if use_demo:
        demo = make_demo_data()
        status = pd.DataFrame({
            "key": list(demo.keys()),
            "indicator": list(demo.keys()),
            "source": "داده نمایشی مصنوعی",
            "latest_observation": [s.index.max() for s in demo.values()],
            "status": "نمایشی",
            "error": "",
            "fetched_at_utc": datetime.now(timezone.utc),
        })
        return demo, status, True

    data, status = get_data()
    critical = {"cpi", "gdp", "unemployment", "policy_rate", "yield_curve"}
    if len(critical.intersection(data)) < 3:
        # The app remains usable if an upstream provider temporarily blocks requests.
        return make_demo_data(), status, True
    return data, status, False


def badge_for_score(score: float) -> str:
    _, label = risk_label(score)
    klass = {"سبز": "badge-green", "زرد": "badge-yellow", "قرمز": "badge-red"}.get(label, "badge-gray")
    text = "نامشخص" if pd.isna(score) else f"{score:.0f}/100 — {label}"
    return f'<span class="badge {klass}">{text}</span>'


def kpi_card(label: str, value: str, note: str, badge_html: str = "") -> str:
    return f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div>{badge_html}</div>
      <div class="kpi-note">{note}</div>
    </div>
    """


def line_chart(series_map: dict[str, pd.Series], title: str, years: int, y_title: str = "", zero_line: bool = False, normalize: bool = False):
    fig = go.Figure()
    for name, s in series_map.items():
        if s is None or s.dropna().empty:
            continue
        x = s.dropna().sort_index()
        cutoff = x.index.max() - pd.DateOffset(years=years)
        x = x.loc[x.index >= cutoff]
        if normalize and not x.empty:
            x = x / x.iloc[0] * 100
        fig.add_trace(go.Scatter(x=x.index, y=x.values, mode="lines", name=name, line=dict(width=2.2)))
    if zero_line:
        fig.add_hline(y=0, line_dash="dash", opacity=.55)
    fig.update_layout(
        title=title,
        height=390,
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        xaxis_title="",
        yaxis_title=y_title,
        template="plotly_white",
    )
    return fig


def pct_series(s: pd.Series, periods: int) -> pd.Series:
    return s.dropna().pct_change(periods) * 100


def render_header(status: pd.DataFrame, fallback: bool):
    successful = status.loc[status["status"] == "موفق"] if not status.empty else pd.DataFrame()
    last_obs = successful["latest_observation"].max() if not successful.empty else pd.NaT
    source_text = "حالت نمایشی" if fallback else "داده زنده منابع رسمی"
    st.markdown('<div class="dashboard-title">داشبورد اقتصاد کلان و مدیریت ریسک</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="dashboard-subtitle">{source_text} • آخرین مشاهده قابل‌دسترس: {last_obs.date() if pd.notna(last_obs) else "نامشخص"} • محاسبات هنگام بازشدن صفحه به‌روزرسانی می‌شوند.</div>',
        unsafe_allow_html=True,
    )
    if fallback:
        st.warning("اتصال زنده به منابع در این اجرا کافی نبود؛ برای جلوگیری از صفحه خالی، داده نمایشی برچسب‌خورده نمایش داده شده است. پس از استقرار روی اینترنت، منابع زنده دوباره خوانده می‌شوند.")


def main():
    with st.sidebar:
        st.header("تنظیمات")
        years = st.select_slider("بازه نمودار", options=[3, 5, 10, 15, 20], value=10, format_func=lambda x: f"{x} سال")
        demo_default = os.getenv("MACRO_DASHBOARD_DEMO", "0") == "1"
        demo_mode = st.toggle("حالت نمایشی", value=demo_default, help="فقط برای پیش‌نمایش بدون اینترنت")
        if st.button("به‌روزرسانی اکنون", width="stretch"):
            st.cache_data.clear()
            st.rerun()
        st.divider()
        st.caption("مدل امتیازدهی، تحلیلی و سناریومحور است؛ خروجی آن پیش‌بینی قطعی رکود، تورم یا بازار نیست.")

    with st.spinner("دریافت و پردازش داده‌ها..."):
        data, status, fallback = safe_data(demo_mode)
        snapshot = calculate_snapshot(data)

    render_header(status, fallback)
    scores = snapshot["scores"]
    m = snapshot["metrics"]

    cards = "".join([
        kpi_card("وضعیت چرخه", snapshot["cycle"], "ترکیب GDP، تولید صنعتی و تغییر بازار کار", badge_for_score(scores["growth"])),
        kpi_card("شرایط پولی", snapshot["monetary_stance"], f"نرخ حقیقی تقریبی: {format_number(m['real_policy_rate'], 1, '٪')}", badge_for_score(scores["inflation"])),
        kpi_card("امتیاز کل ریسک", format_number(scores["overall"], 0, "/100"), "میانگین وزنی رشد، تورم، مالی و عرضه", badge_for_score(scores["overall"])),
        kpi_card("ریسک اصلی ۶ماهه", snapshot["biggest_risk"], snapshot["narratives"]["confidence"], badge_for_score(max(v for k, v in scores.items() if k != "overall" and not pd.isna(v)))),
    ])
    st.markdown(f'<div class="kpi-grid">{cards}</div>', unsafe_allow_html=True)

    tabs = st.tabs(["خلاصه مدیریتی", "رشد و تورم", "پول و اعتبار", "بازارها", "ژئوپلیتیک و زنجیره تأمین", "اثر بر کسب‌وکار", "داده و روش"])

    with tabs[0]:
        st.markdown("### سه سؤال طلایی این هفته")
        for question, answer in [
            ("اقتصاد جهانی در حال گرم‌تر شدن است یا سردتر شدن؟", snapshot["narratives"]["temperature"]),
            ("پول در حال ارزان‌تر شدن است یا گران‌تر؟", snapshot["narratives"]["money"]),
            ("بزرگ‌ترین ریسک شش ماه آینده چیست؟", snapshot["narratives"]["risk"]),
        ]:
            st.markdown(f'<div class="question-card"><strong>{question}</strong><br>{answer}</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([1.05, 1])
        with col1:
            st.markdown("### چراغ راهنمای اقتصاد")
            st.dataframe(snapshot["traffic"], hide_index=True, width="stretch", height=430)
        with col2:
            pillar = pd.DataFrame({
                "حوزه": ["رشد", "تورم", "شرایط مالی", "عرضه و ژئوپلیتیک"],
                "امتیاز": [scores["growth"], scores["inflation"], scores["financial"], scores["supply"]],
            }).dropna()
            fig = go.Figure(go.Bar(x=pillar["امتیاز"], y=pillar["حوزه"], orientation="h", text=pillar["امتیاز"].round(0), textposition="auto"))
            fig.update_layout(title="چهار ستون ریسک", xaxis_range=[0, 100], height=420, margin=dict(l=20, r=20, t=55, b=20), template="plotly_white")
            st.plotly_chart(fig, width="stretch")

    with tabs[1]:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("رشد حقیقی GDP", format_number(m["gdp_yoy"], 1, "٪"))
        c2.metric("تولید صنعتی", format_number(m["ip_yoy"], 1, "٪"))
        c3.metric("تورم کل", format_number(m["cpi_yoy"], 1, "٪"))
        c4.metric("بیکاری", format_number(m["unemployment"], 1, "٪"), delta=format_number(m["unemployment_3m"], 1, " واحد درصد / ۳ماه"))

        if "gdp" in data and "industrial_production" in data:
            st.plotly_chart(line_chart({"GDP حقیقی (رشد سالانه)": pct_series(data["gdp"], 4), "تولید صنعتی (رشد سالانه)": pct_series(data["industrial_production"], 12)}, "رشد اقتصادی", years, "درصد"), width="stretch")
        if "cpi" in data:
            inflation_map = {"تورم کل": pct_series(data["cpi"], 12)}
            if "core_cpi" in data:
                inflation_map["تورم هسته"] = pct_series(data["core_cpi"], 12)
            st.plotly_chart(line_chart(inflation_map, "تورم سالانه", years, "درصد"), width="stretch")
        if "unemployment" in data:
            st.plotly_chart(line_chart({"نرخ بیکاری": data["unemployment"]}, "بازار کار", years, "درصد"), width="stretch")

    with tabs[2]:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("نرخ سیاستی", format_number(m["policy_rate"], 2, "٪"))
        c2.metric("نرخ حقیقی تقریبی", format_number(m["real_policy_rate"], 1, "٪"))
        c3.metric("شیب منحنی بازده", format_number(m["yield_curve"], 2, " واحد درصد"))
        c4.metric("رشد اعتبار بانکی", format_number(m["credit_yoy"], 1, "٪"))

        rate_map = {}
        if "policy_rate" in data: rate_map["نرخ سیاستی"] = data["policy_rate"]
        if "real_yield" in data: rate_map["بازده حقیقی ۱۰ساله"] = data["real_yield"]
        if rate_map:
            st.plotly_chart(line_chart(rate_map, "هزینه پول و نرخ حقیقی", years, "درصد", zero_line=True), width="stretch")
        if "yield_curve" in data:
            st.plotly_chart(line_chart({"۱۰ساله منهای ۳ماهه": data["yield_curve"]}, "منحنی بازده", years, "واحد درصد", zero_line=True), width="stretch")
        liquidity = {}
        if "m2" in data: liquidity["رشد M2"] = pct_series(data["m2"], 12)
        if "bank_credit" in data: liquidity["رشد اعتبار بانکی"] = pct_series(data["bank_credit"], 52)
        if liquidity:
            st.plotly_chart(line_chart(liquidity, "نقدینگی و اعتبار — رشد سالانه", years, "درصد", zero_line=True), width="stretch")
        if "financial_stress" in data:
            st.plotly_chart(line_chart({"تنش مالی": data["financial_stress"]}, "شاخص تنش مالی", years, "انحراف معیار", zero_line=True), width="stretch")

    with tabs[3]:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("نفت برنت / ۳ماه", format_number(m["oil_3m"], 1, "٪"))
        c2.metric("مس / ۳ماه", format_number(m["copper_3m"], 1, "٪"))
        c3.metric("طلا / ۳ماه", format_number(m["gold_3m"], 1, "٪"))
        c4.metric("دلار / ۳ماه", format_number(m["dollar_3m"], 1, "٪"))
        market_map = {}
        for key, title in [("brent", "نفت"), ("copper", "مس"), ("gold", "طلا"), ("dollar", "دلار")]:
            if key in data: market_map[title] = data[key]
        if market_map:
            st.plotly_chart(line_chart(market_map, "عملکرد نسبی بازارها — مبنا ۱۰۰", years, "شاخص", normalize=True), width="stretch")
        st.info("قیمت‌ها از آخرین مشاهده منتشرشده منابع دریافت می‌شوند و برای معامله لحظه‌ای طراحی نشده‌اند.")

    with tabs[4]:
        c1, c2, c3 = st.columns(3)
        c1.metric("فشار زنجیره تأمین", format_number(m["gscpi"], 2), help="بالاتر از صفر یعنی فشار بالاتر از میانگین تاریخی")
        c2.metric("صدک ریسک ژئوپلیتیک", format_number(m["gpr_percentile"], 0, "%"))
        c3.metric("صدک عدم‌قطعیت جهانی", format_number(m["uncertainty_percentile"], 0, "%"))
        if "gscpi" in data:
            st.plotly_chart(line_chart({"GSCPI": data["gscpi"]}, "فشار زنجیره تأمین جهانی", years, "انحراف معیار", zero_line=True), width="stretch")
        risk_map = {}
        if "gpr" in data: risk_map["ریسک ژئوپلیتیک AI-GPR"] = data["gpr"]
        if "policy_uncertainty" in data: risk_map["عدم‌قطعیت سیاست اقتصادی"] = data["policy_uncertainty"]
        if risk_map:
            st.plotly_chart(line_chart(risk_map, "ریسک و عدم‌قطعیت", years, "شاخص", normalize=True), width="stretch")
        st.caption("AI-GPR یک شاخص خبری مبتنی بر مدل زبانی است؛ بنابراین مکمل تحلیل رویدادهاست، نه جایگزین ارزیابی انسانی و حقوقی تحریم‌ها.")

    with tabs[5]:
        st.markdown("### ترجمه سیگنال‌های کلان به اقدام تجاری")
        st.dataframe(snapshot["business"], hide_index=True, width="stretch")
        st.markdown("### اقدام عملی پیشنهادی")
        overall = scores["overall"]
        if pd.isna(overall):
            st.warning("پوشش داده برای توصیه عملی کافی نیست.")
        elif overall > 66:
            st.error("سناریوی دفاعی: نقدینگی بالاتر، موجودی ایمن برای اقلام بحرانی، تأمین‌کننده جایگزین، کوتاه‌کردن اعتبار قیمت و کنترل تعهدات ارزی.")
        elif overall > 33:
            st.warning("سناریوی احتیاط: خرید مرحله‌ای، قرارداد حمل انعطاف‌پذیر، سناریوی نرخ ارز و بازبینی هفتگی شاخص‌های هشدار.")
        else:
            st.success("سناریوی متعادل: حفظ کنترل ریسک، اما امکان برنامه‌ریزی رشد و سرمایه‌گذاری مرحله‌ای بیشتر است.")

    with tabs[6]:
        st.markdown("### وضعیت اتصال منابع")
        view_status = status.copy()
        if "fetched_at_utc" in view_status:
            view_status["fetched_at_utc"] = pd.to_datetime(view_status["fetched_at_utc"]).dt.strftime("%Y-%m-%d %H:%M UTC")
        if "latest_observation" in view_status:
            view_status["latest_observation"] = pd.to_datetime(view_status["latest_observation"], errors="coerce").dt.strftime("%Y-%m-%d")
        st.dataframe(view_status[[c for c in ["indicator", "source", "latest_observation", "status", "error"] if c in view_status]], hide_index=True, width="stretch")

        st.markdown("### روش محاسبه")
        st.markdown("""
- امتیازها از **۰ تا ۱۰۰** هستند؛ عدد بالاتر یعنی ریسک بیشتر.
- چهار ستون وزن برابر دارند: رشد، تورم، شرایط مالی، و عرضه/ژئوپلیتیک.
- داده‌های اقتصادی با تناوب انتشار خودشان به‌روز می‌شوند؛ بازشدن روزانه داشبورد، GDP فصلی را روزانه نمی‌کند.
- نرخ حقیقی تقریبی برابر نرخ سیاستی منهای تورم سالانه است و جایگزین برآوردهای دقیق‌تر نرخ حقیقی تعادلی نیست.
- آستانه‌ها فرض تحلیلی اولیه‌اند و باید بعداً با حساسیت کسب‌وکار، کشور هدف و داده‌های داخلی شرکت کالیبره شوند.
        """)
        st.markdown("### منابع اصلی")
        st.markdown("FRED / Federal Reserve Bank of St. Louis، New York Fed GSCPI، و AI-GPR از Iacoviello & Tong. نام سری‌های FRED در جدول اتصال نمایش داده می‌شود.")


if __name__ == "__main__":
    main()
