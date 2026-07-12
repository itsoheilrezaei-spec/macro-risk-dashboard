from __future__ import annotations

from datetime import datetime, timezone
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard_core import (
    calculate_snapshot,
    format_number,
    load_live_data,
    make_demo_data,
    risk_label,
)

st.set_page_config(
    page_title="داشبورد اقتصاد کلان و مدیریت ریسک",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# فقط برای راست‌چین‌سازی و بهبود نمایش موبایل؛ محتوای کارت‌ها با اجزای بومی
# Streamlit ساخته می‌شود تا کد HTML به‌صورت متن نمایش داده نشود.
st.markdown(
    """
    <style>
    html, body { direction: rtl; text-align: right; }
    [data-testid="stSidebar"] { direction: rtl; }
    [data-testid="stMetric"] { direction: rtl; text-align: right; }
    [data-testid="stDataFrame"] { direction: rtl; }
    .block-container {
        padding-top: 0.8rem;
        padding-bottom: 2.5rem;
        max-width: 1450px;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px;
    }
    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.75rem;
            padding-right: 0.75rem;
            padding-top: 0.35rem;
        }
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 0.55rem !important;
        }
        [data-testid="column"] {
            min-width: 100% !important;
            width: 100% !important;
            flex: 1 1 100% !important;
        }
        h1 { font-size: 1.65rem !important; line-height: 1.45 !important; }
        h2 { font-size: 1.35rem !important; }
        h3 { font-size: 1.12rem !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def get_data():
    return load_live_data()


def safe_data(use_demo: bool):
    if use_demo:
        demo = make_demo_data()
        status = pd.DataFrame(
            {
                "key": list(demo.keys()),
                "indicator": list(demo.keys()),
                "source": "داده نمایشی مصنوعی",
                "latest_observation": [s.index.max() for s in demo.values()],
                "status": "نمایشی",
                "error": "",
                "fetched_at_utc": datetime.now(timezone.utc),
            }
        )
        return demo, status, True

    data, status = get_data()
    critical = {"cpi", "gdp", "unemployment", "policy_rate", "yield_curve"}
    if len(critical.intersection(data)) < 3:
        return make_demo_data(), status, True
    return data, status, False


def score_text(score: float) -> str:
    icon, label = risk_label(score)
    if pd.isna(score):
        return "⚪ داده ناکافی"
    return f"{icon} {score:.0f}/100 — {label}"


def latest_valid_score(scores: dict[str, float]) -> float:
    values = [
        value
        for key, value in scores.items()
        if key != "overall" and value is not None and not pd.isna(value)
    ]
    return max(values) if values else np.nan


def render_summary_card(title: str, value: str, note: str, score: float):
    with st.container(border=True):
        st.caption(title)
        st.markdown(f"### {value}")
        st.write(score_text(score))
        st.caption(note)


def line_chart(
    series_map: dict[str, pd.Series],
    title: str,
    years: int,
    y_title: str = "",
    zero_line: bool = False,
    normalize: bool = False,
):
    fig = go.Figure()
    for name, series in series_map.items():
        if series is None or series.dropna().empty:
            continue
        x = series.dropna().sort_index()
        cutoff = x.index.max() - pd.DateOffset(years=years)
        x = x.loc[x.index >= cutoff]
        if normalize and not x.empty:
            x = x / x.iloc[0] * 100
        fig.add_trace(
            go.Scatter(
                x=x.index,
                y=x.values,
                mode="lines",
                name=name,
                line=dict(width=2.2),
            )
        )
    if zero_line:
        fig.add_hline(y=0, line_dash="dash", opacity=0.55)
    fig.update_layout(
        title=title,
        height=390,
        margin=dict(l=12, r=12, t=55, b=18),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hovermode="x unified",
        xaxis_title="",
        yaxis_title=y_title,
    )
    return fig


def pct_series(series: pd.Series, periods: int) -> pd.Series:
    return series.dropna().pct_change(periods) * 100


def render_header(status: pd.DataFrame, fallback: bool):
    successful = (
        status.loc[status["status"] == "موفق"]
        if not status.empty and "status" in status.columns
        else pd.DataFrame()
    )
    last_obs = (
        successful["latest_observation"].max()
        if not successful.empty and "latest_observation" in successful.columns
        else pd.NaT
    )
    source_text = "حالت نمایشی / جایگزین" if fallback else "داده زنده منابع رسمی"
    st.title("داشبورد اقتصاد کلان و مدیریت ریسک")
    last_text = str(last_obs.date()) if pd.notna(last_obs) else "نامشخص"
    st.caption(
        f"{source_text} • آخرین مشاهده قابل‌دسترس: {last_text} • "
        "محاسبات هنگام بازشدن صفحه به‌روزرسانی می‌شوند."
    )
    if fallback:
        st.warning(
            "در این اجرا داده زنده کافی دریافت نشد و بخشی از داشبورد با داده نمایشی "
            "برچسب‌خورده نمایش داده می‌شود. برای دیدن علت، بخش «داده و روش» را باز کنید."
        )


def render_executive(snapshot: dict):
    scores = snapshot["scores"]
    st.subheader("سه سؤال طلایی این هفته")
    for question, answer in [
        (
            "اقتصاد جهانی در حال گرم‌تر شدن است یا سردتر شدن؟",
            snapshot["narratives"]["temperature"],
        ),
        (
            "پول در حال ارزان‌تر شدن است یا گران‌تر؟",
            snapshot["narratives"]["money"],
        ),
        (
            "بزرگ‌ترین ریسک شش ماه آینده چیست؟",
            snapshot["narratives"]["risk"],
        ),
    ]:
        with st.container(border=True):
            st.markdown(f"**{question}**")
            st.write(answer)

    st.subheader("چراغ راهنمای اقتصاد")
    st.dataframe(snapshot["traffic"], hide_index=True, width="stretch", height=430)

    pillar = pd.DataFrame(
        {
            "حوزه": ["رشد", "تورم", "شرایط مالی", "عرضه و ژئوپلیتیک"],
            "امتیاز": [
                scores["growth"],
                scores["inflation"],
                scores["financial"],
                scores["supply"],
            ],
        }
    ).dropna()
    fig = go.Figure(
        go.Bar(
            x=pillar["امتیاز"],
            y=pillar["حوزه"],
            orientation="h",
            text=pillar["امتیاز"].round(0),
            textposition="auto",
        )
    )
    fig.update_layout(
        title="چهار ستون ریسک",
        xaxis_range=[0, 100],
        height=380,
        margin=dict(l=12, r=12, t=55, b=18),
    )
    st.plotly_chart(fig, width="stretch")


def render_growth(data: dict[str, pd.Series], metrics: dict, years: int):
    cols = st.columns(4)
    cols[0].metric("رشد حقیقی GDP", format_number(metrics["gdp_yoy"], 1, "٪"))
    cols[1].metric("تولید صنعتی", format_number(metrics["ip_yoy"], 1, "٪"))
    cols[2].metric("تورم کل", format_number(metrics["cpi_yoy"], 1, "٪"))
    cols[3].metric(
        "بیکاری",
        format_number(metrics["unemployment"], 1, "٪"),
        delta=format_number(metrics["unemployment_3m"], 1, " واحد درصد / ۳ماه"),
    )

    if "gdp" in data and "industrial_production" in data:
        st.plotly_chart(
            line_chart(
                {
                    "GDP حقیقی (رشد سالانه)": pct_series(data["gdp"], 4),
                    "تولید صنعتی (رشد سالانه)": pct_series(
                        data["industrial_production"], 12
                    ),
                },
                "رشد اقتصادی",
                years,
                "درصد",
            ),
            width="stretch",
        )
    if "cpi" in data:
        inflation_map = {"تورم کل": pct_series(data["cpi"], 12)}
        if "core_cpi" in data:
            inflation_map["تورم هسته"] = pct_series(data["core_cpi"], 12)
        st.plotly_chart(
            line_chart(inflation_map, "تورم سالانه", years, "درصد"),
            width="stretch",
        )
    if "unemployment" in data:
        st.plotly_chart(
            line_chart(
                {"نرخ بیکاری": data["unemployment"]},
                "بازار کار",
                years,
                "درصد",
            ),
            width="stretch",
        )


def render_money(data: dict[str, pd.Series], metrics: dict, years: int):
    cols = st.columns(4)
    cols[0].metric("نرخ سیاستی", format_number(metrics["policy_rate"], 2, "٪"))
    cols[1].metric(
        "نرخ حقیقی تقریبی", format_number(metrics["real_policy_rate"], 1, "٪")
    )
    cols[2].metric(
        "شیب منحنی بازده", format_number(metrics["yield_curve"], 2, " واحد درصد")
    )
    cols[3].metric(
        "رشد اعتبار بانکی", format_number(metrics["credit_yoy"], 1, "٪")
    )

    rate_map: dict[str, pd.Series] = {}
    if "policy_rate" in data:
        rate_map["نرخ سیاستی"] = data["policy_rate"]
    if "real_yield" in data:
        rate_map["بازده حقیقی ۱۰ساله"] = data["real_yield"]
    if rate_map:
        st.plotly_chart(
            line_chart(
                rate_map,
                "هزینه پول و نرخ حقیقی",
                years,
                "درصد",
                zero_line=True,
            ),
            width="stretch",
        )
    if "yield_curve" in data:
        st.plotly_chart(
            line_chart(
                {"۱۰ساله منهای ۳ماهه": data["yield_curve"]},
                "منحنی بازده",
                years,
                "واحد درصد",
                zero_line=True,
            ),
            width="stretch",
        )
    liquidity: dict[str, pd.Series] = {}
    if "m2" in data:
        liquidity["رشد M2"] = pct_series(data["m2"], 12)
    if "bank_credit" in data:
        liquidity["رشد اعتبار بانکی"] = pct_series(data["bank_credit"], 52)
    if liquidity:
        st.plotly_chart(
            line_chart(
                liquidity,
                "نقدینگی و اعتبار — رشد سالانه",
                years,
                "درصد",
                zero_line=True,
            ),
            width="stretch",
        )
    if "financial_stress" in data:
        st.plotly_chart(
            line_chart(
                {"تنش مالی": data["financial_stress"]},
                "شاخص تنش مالی",
                years,
                "انحراف معیار",
                zero_line=True,
            ),
            width="stretch",
        )


def render_markets(data: dict[str, pd.Series], metrics: dict, years: int):
    cols = st.columns(4)
    cols[0].metric("نفت برنت / ۳ماه", format_number(metrics["oil_3m"], 1, "٪"))
    cols[1].metric("مس / ۳ماه", format_number(metrics["copper_3m"], 1, "٪"))
    cols[2].metric("طلا / ۳ماه", format_number(metrics["gold_3m"], 1, "٪"))
    cols[3].metric("دلار / ۳ماه", format_number(metrics["dollar_3m"], 1, "٪"))

    market_map: dict[str, pd.Series] = {}
    for key, title in [
        ("brent", "نفت"),
        ("copper", "مس"),
        ("gold", "طلا"),
        ("dollar", "دلار"),
    ]:
        if key in data:
            market_map[title] = data[key]
    if market_map:
        st.plotly_chart(
            line_chart(
                market_map,
                "عملکرد نسبی بازارها — مبنا ۱۰۰",
                years,
                "شاخص",
                normalize=True,
            ),
            width="stretch",
        )
    st.info(
        "قیمت‌ها از آخرین مشاهده منتشرشده منابع دریافت می‌شوند و برای معامله لحظه‌ای طراحی نشده‌اند."
    )


def render_supply(data: dict[str, pd.Series], metrics: dict, years: int):
    cols = st.columns(3)
    cols[0].metric(
        "فشار زنجیره تأمین",
        format_number(metrics["gscpi"], 2),
        help="بالاتر از صفر یعنی فشار بالاتر از میانگین تاریخی",
    )
    cols[1].metric(
        "صدک ریسک ژئوپلیتیک", format_number(metrics["gpr_percentile"], 0, "%")
    )
    cols[2].metric(
        "صدک عدم‌قطعیت جهانی",
        format_number(metrics["uncertainty_percentile"], 0, "%"),
    )

    if "gscpi" in data:
        st.plotly_chart(
            line_chart(
                {"GSCPI": data["gscpi"]},
                "فشار زنجیره تأمین جهانی",
                years,
                "انحراف معیار",
                zero_line=True,
            ),
            width="stretch",
        )
    risk_map: dict[str, pd.Series] = {}
    if "gpr" in data:
        risk_map["ریسک ژئوپلیتیک AI-GPR"] = data["gpr"]
    if "policy_uncertainty" in data:
        risk_map["عدم‌قطعیت سیاست اقتصادی"] = data["policy_uncertainty"]
    if risk_map:
        st.plotly_chart(
            line_chart(
                risk_map,
                "ریسک و عدم‌قطعیت",
                years,
                "شاخص",
                normalize=True,
            ),
            width="stretch",
        )
    st.caption(
        "AI-GPR یک شاخص خبری مبتنی بر مدل زبانی است؛ مکمل تحلیل رویدادهاست و جایگزین ارزیابی انسانی و حقوقی تحریم‌ها نیست."
    )


def render_business(snapshot: dict):
    st.subheader("ترجمه سیگنال‌های کلان به اقدام تجاری")
    st.dataframe(snapshot["business"], hide_index=True, width="stretch")
    st.subheader("اقدام عملی پیشنهادی")
    overall = snapshot["scores"]["overall"]
    if pd.isna(overall):
        st.warning("پوشش داده برای توصیه عملی کافی نیست.")
    elif overall > 66:
        st.error(
            "سناریوی دفاعی: نقدینگی بالاتر، موجودی ایمن برای اقلام بحرانی، "
            "تأمین‌کننده جایگزین، کوتاه‌کردن اعتبار قیمت و کنترل تعهدات ارزی."
        )
    elif overall > 33:
        st.warning(
            "سناریوی احتیاط: خرید مرحله‌ای، قرارداد حمل انعطاف‌پذیر، "
            "سناریوی نرخ ارز و بازبینی هفتگی شاخص‌های هشدار."
        )
    else:
        st.success(
            "سناریوی متعادل: حفظ کنترل ریسک، همراه با امکان برنامه‌ریزی رشد "
            "و سرمایه‌گذاری مرحله‌ای بیشتر."
        )


def render_method(status: pd.DataFrame):
    st.subheader("وضعیت اتصال منابع")
    view_status = status.copy()
    if "fetched_at_utc" in view_status:
        view_status["fetched_at_utc"] = pd.to_datetime(
            view_status["fetched_at_utc"]
        ).dt.strftime("%Y-%m-%d %H:%M UTC")
    if "latest_observation" in view_status:
        view_status["latest_observation"] = pd.to_datetime(
            view_status["latest_observation"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")
    columns = [
        col
        for col in ["indicator", "source", "latest_observation", "status", "error"]
        if col in view_status
    ]
    st.dataframe(view_status[columns], hide_index=True, width="stretch")

    st.subheader("روش محاسبه")
    st.markdown(
        """
- امتیازها از **۰ تا ۱۰۰** هستند؛ عدد بالاتر یعنی ریسک بیشتر.
- چهار ستون وزن برابر دارند: رشد، تورم، شرایط مالی، و عرضه/ژئوپلیتیک.
- داده‌های اقتصادی مطابق تناوب انتشار خودشان به‌روز می‌شوند؛ بازشدن روزانه داشبورد، GDP فصلی را روزانه نمی‌کند.
- نرخ حقیقی تقریبی برابر نرخ سیاستی منهای تورم سالانه است و جایگزین برآورد دقیق نرخ حقیقی تعادلی نیست.
- آستانه‌ها فرض تحلیلی اولیه‌اند و باید با حساسیت کسب‌وکار و کشور هدف کالیبره شوند.
        """
    )
    st.subheader("منابع اصلی")
    st.write(
        "FRED / Federal Reserve Bank of St. Louis، New York Fed GSCPI و AI-GPR. "
        "نام سری‌های دریافت‌شده در جدول اتصال نمایش داده می‌شود."
    )


def main():
    with st.sidebar:
        st.header("تنظیمات")
        years = st.select_slider(
            "بازه نمودار",
            options=[3, 5, 10, 15, 20],
            value=10,
            format_func=lambda x: f"{x} سال",
        )
        demo_default = os.getenv("MACRO_DASHBOARD_DEMO", "0") == "1"
        demo_mode = st.toggle(
            "حالت نمایشی",
            value=demo_default,
            help="فقط برای پیش‌نمایش بدون اینترنت",
        )
        if st.button("به‌روزرسانی اکنون", width="stretch"):
            st.cache_data.clear()
            st.rerun()
        st.divider()
        st.caption(
            "مدل امتیازدهی تحلیلی و سناریومحور است؛ خروجی آن پیش‌بینی قطعی رکود، تورم یا بازار نیست."
        )

    with st.spinner("دریافت و پردازش داده‌ها..."):
        data, status, fallback = safe_data(demo_mode)
        snapshot = calculate_snapshot(data)

    render_header(status, fallback)
    scores = snapshot["scores"]
    metrics = snapshot["metrics"]

    cards = st.columns(4)
    with cards[0]:
        render_summary_card(
            "وضعیت چرخه",
            snapshot["cycle"],
            "ترکیب GDP، تولید صنعتی و تغییر بازار کار",
            scores["growth"],
        )
    with cards[1]:
        render_summary_card(
            "شرایط پولی",
            snapshot["monetary_stance"],
            f"نرخ حقیقی تقریبی: {format_number(metrics['real_policy_rate'], 1, '٪')}",
            scores["inflation"],
        )
    with cards[2]:
        render_summary_card(
            "امتیاز کل ریسک",
            format_number(scores["overall"], 0, "/100"),
            "میانگین وزنی رشد، تورم، مالی و عرضه",
            scores["overall"],
        )
    with cards[3]:
        render_summary_card(
            "ریسک اصلی ۶ماهه",
            snapshot["biggest_risk"],
            snapshot["narratives"]["confidence"],
            latest_valid_score(scores),
        )

    page = st.selectbox(
        "بخش داشبورد",
        [
            "خلاصه مدیریتی",
            "رشد و تورم",
            "پول و اعتبار",
            "بازارها",
            "ژئوپلیتیک و زنجیره تأمین",
            "اثر بر کسب‌وکار",
            "داده و روش",
        ],
        index=0,
    )

    if page == "خلاصه مدیریتی":
        render_executive(snapshot)
    elif page == "رشد و تورم":
        render_growth(data, metrics, years)
    elif page == "پول و اعتبار":
        render_money(data, metrics, years)
    elif page == "بازارها":
        render_markets(data, metrics, years)
    elif page == "ژئوپلیتیک و زنجیره تأمین":
        render_supply(data, metrics, years)
    elif page == "اثر بر کسب‌وکار":
        render_business(snapshot)
    else:
        render_method(status)


if __name__ == "__main__":
    main()
