from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO, StringIO
from typing import Dict, Iterable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
GSCPI_XLSX = "https://www.newyorkfed.org/medialibrary/research/interactives/gscpi/downloads/gscpi_data.xlsx"
AI_GPR_MONTHLY = "https://www.matteoiacoviello.com/ai_gpr_files/ai_gpr_data_monthly.csv"


@dataclass(frozen=True)
class SeriesSpec:
    key: str
    ids: tuple[str, ...]
    title_fa: str
    unit_fa: str
    frequency: str
    source: str = "FRED"


SERIES_SPECS: Dict[str, SeriesSpec] = {
    "policy_rate": SeriesSpec("policy_rate", ("FEDFUNDS",), "نرخ بهره سیاستی آمریکا", "درصد", "monthly"),
    "cpi": SeriesSpec("cpi", ("CPIAUCSL",), "شاخص قیمت مصرف‌کننده", "شاخص", "monthly"),
    "core_cpi": SeriesSpec("core_cpi", ("CPILFESL",), "تورم هسته", "شاخص", "monthly"),
    "gdp": SeriesSpec("gdp", ("GDPC1",), "تولید ناخالص داخلی حقیقی", "میلیارد دلار حقیقی", "quarterly"),
    "unemployment": SeriesSpec("unemployment", ("UNRATE",), "نرخ بیکاری", "درصد", "monthly"),
    "yield_curve": SeriesSpec("yield_curve", ("T10Y3M",), "شیب منحنی بازده ۱۰ساله منهای ۳ماهه", "واحد درصد", "daily"),
    "m2": SeriesSpec("m2", ("M2SL",), "نقدینگی M2", "میلیارد دلار", "monthly"),
    "bank_credit": SeriesSpec("bank_credit", ("TOTBKCR",), "اعتبار بانکی", "میلیارد دلار", "weekly"),
    "industrial_production": SeriesSpec("industrial_production", ("INDPRO",), "تولید صنعتی", "شاخص", "monthly"),
    "brent": SeriesSpec("brent", ("DCOILBRENTEU",), "نفت برنت", "دلار/بشکه", "daily"),
    "copper": SeriesSpec("copper", ("PCOPPUSDM",), "قیمت جهانی مس", "دلار/تن", "monthly"),
    "gold": SeriesSpec("gold", ("GOLDAMGBD228NLBM", "GOLDPMGBD228NLBM"), "طلا", "دلار/اونس", "daily"),
    "dollar": SeriesSpec("dollar", ("DTWEXBGS",), "شاخص گسترده دلار", "شاخص", "daily"),
    "real_yield": SeriesSpec("real_yield", ("DFII10",), "بازده حقیقی ۱۰ساله", "درصد", "daily"),
    "financial_stress": SeriesSpec("financial_stress", ("STLFSI4",), "شاخص تنش مالی سنت‌لوئیس", "انحراف معیار", "weekly"),
    "policy_uncertainty": SeriesSpec("policy_uncertainty", ("GEPUCURRENT",), "عدم‌قطعیت سیاست اقتصادی جهانی", "شاخص", "monthly"),
}


class DataFetchError(RuntimeError):
    pass


def _http_get(url: str, timeout: int = 18) -> bytes:
    headers = {"User-Agent": "MacroRiskDashboard/1.0 (+personal analytical dashboard)"}
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.content


def clean_series(df: pd.DataFrame, value_col: Optional[str] = None) -> pd.Series:
    if df.empty:
        raise DataFetchError("empty dataset")
    date_candidates = [c for c in df.columns if str(c).lower() in {"date", "observation_date", "time", "month"}]
    date_col = date_candidates[0] if date_candidates else df.columns[0]
    if value_col is None:
        numeric_candidates = [c for c in df.columns if c != date_col]
        if not numeric_candidates:
            raise DataFetchError("value column not found")
        value_col = numeric_candidates[0]
    dates = pd.to_datetime(df[date_col], errors="coerce")
    values = pd.to_numeric(df[value_col].replace(".", np.nan), errors="coerce")
    s = pd.Series(values.values, index=dates, name=str(value_col)).dropna()
    s = s[~s.index.isna()].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    if s.empty:
        raise DataFetchError("no usable observations")
    return s


def fetch_fred(spec: SeriesSpec) -> tuple[pd.Series, str]:
    errors: list[str] = []
    for series_id in spec.ids:
        try:
            raw = _http_get(FRED_CSV.format(series_id=series_id))
            df = pd.read_csv(BytesIO(raw))
            value_col = series_id if series_id in df.columns else None
            return clean_series(df, value_col), series_id
        except Exception as exc:  # pragma: no cover - network dependent
            errors.append(f"{series_id}: {exc}")
    raise DataFetchError(" | ".join(errors))


def fetch_gscpi() -> pd.Series:
    raw = _http_get(GSCPI_XLSX)
    workbook = pd.ExcelFile(BytesIO(raw))
    for sheet in workbook.sheet_names:
        df = pd.read_excel(workbook, sheet_name=sheet)
        if df.empty:
            continue
        normalized = {str(c).strip().lower(): c for c in df.columns}
        date_col = next((normalized[k] for k in normalized if "date" in k or "month" in k), None)
        value_col = next((normalized[k] for k in normalized if "gscpi" in k), None)
        if date_col is not None and value_col is not None:
            return clean_series(df[[date_col, value_col]], value_col)
    # Fallback: infer first date-like and first numeric column.
    for sheet in workbook.sheet_names:
        df = pd.read_excel(workbook, sheet_name=sheet)
        try:
            return clean_series(df)
        except Exception:
            continue
    raise DataFetchError("GSCPI columns not found")


def fetch_ai_gpr() -> pd.Series:
    raw = _http_get(AI_GPR_MONTHLY)
    df = pd.read_csv(BytesIO(raw))
    cols = {str(c).strip().lower(): c for c in df.columns}
    date_col = next((cols[k] for k in cols if k in {"date", "month"} or "date" in k), df.columns[0])
    value_col = next(
        (cols[k] for k in cols if k in {"gpr_ai", "ai_gpr", "gpr ai"}),
        None,
    )
    if value_col is None:
        candidates = [c for c in df.columns if c != date_col and "gpr" in str(c).lower()]
        value_col = candidates[0] if candidates else df.columns[1]
    return clean_series(df[[date_col, value_col]], value_col)


def load_live_data() -> tuple[dict[str, pd.Series], pd.DataFrame]:
    """Fetch independent sources concurrently so a slow provider does not block the dashboard."""
    data: dict[str, pd.Series] = {}
    status_rows: list[dict] = []
    fetched_at = datetime.now(timezone.utc)

    def fred_job(key: str, spec: SeriesSpec):
        s, actual_id = fetch_fred(spec)
        return key, spec.title_fa, f"FRED:{actual_id}", s

    def external_job(key: str, title: str, source: str, fn):
        return key, title, source, fn()

    jobs = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for key, spec in SERIES_SPECS.items():
            jobs[pool.submit(fred_job, key, spec)] = (key, spec.title_fa, "FRED")
        external = {
            "gscpi": ("فشار زنجیره تأمین جهانی", "NY Fed GSCPI", fetch_gscpi),
            "gpr": ("ریسک ژئوپلیتیک AI-GPR", "Iacoviello & Tong", fetch_ai_gpr),
        }
        for key, (title, source, fn) in external.items():
            jobs[pool.submit(external_job, key, title, source, fn)] = (key, title, source)

        for future in as_completed(jobs):
            key, title, source_hint = jobs[future]
            try:
                result_key, result_title, source, series = future.result()
                data[result_key] = series
                status_rows.append({
                    "key": result_key,
                    "indicator": result_title,
                    "source": source,
                    "latest_observation": series.index.max(),
                    "status": "موفق",
                    "error": "",
                    "fetched_at_utc": fetched_at,
                })
            except Exception as exc:
                status_rows.append({
                    "key": key,
                    "indicator": title,
                    "source": source_hint,
                    "latest_observation": pd.NaT,
                    "status": "ناموفق",
                    "error": str(exc)[:240],
                    "fetched_at_utc": fetched_at,
                })

    status = pd.DataFrame(status_rows)
    if not status.empty:
        status = status.sort_values(["status", "indicator"], ascending=[False, True]).reset_index(drop=True)
    return data, status


def clip(value: float, low: float = 0.0, high: float = 100.0) -> float:
    if value is None or pd.isna(value):
        return np.nan
    return float(min(high, max(low, value)))


def latest(s: Optional[pd.Series]) -> float:
    if s is None or s.dropna().empty:
        return np.nan
    return float(s.dropna().iloc[-1])


def latest_date(s: Optional[pd.Series]) -> Optional[pd.Timestamp]:
    if s is None or s.dropna().empty:
        return None
    return pd.Timestamp(s.dropna().index[-1])


def yoy(s: Optional[pd.Series], periods: int) -> float:
    if s is None:
        return np.nan
    x = s.dropna()
    if len(x) <= periods:
        return np.nan
    return float((x.iloc[-1] / x.iloc[-1 - periods] - 1.0) * 100.0)


def change_over_months(s: Optional[pd.Series], months: int, pct: bool = True) -> float:
    if s is None or s.dropna().empty:
        return np.nan
    x = s.dropna().sort_index()
    end_date = x.index[-1]
    target = end_date - pd.DateOffset(months=months)
    before = x.loc[x.index <= target]
    if before.empty:
        return np.nan
    old = float(before.iloc[-1])
    new = float(x.iloc[-1])
    if pct:
        if old == 0:
            return np.nan
        return (new / old - 1.0) * 100.0
    return new - old


def annualized_3m_inflation(s: Optional[pd.Series]) -> float:
    if s is None:
        return np.nan
    x = s.dropna()
    if len(x) < 4:
        return np.nan
    return float(((x.iloc[-1] / x.iloc[-4]) ** 4 - 1.0) * 100.0)


def percentile_rank(s: Optional[pd.Series], years: int = 10) -> float:
    if s is None or s.dropna().empty:
        return np.nan
    x = s.dropna().sort_index()
    cutoff = x.index[-1] - pd.DateOffset(years=years)
    hist = x.loc[x.index >= cutoff]
    if len(hist) < 10:
        hist = x
    if len(hist) < 2:
        return np.nan
    val = hist.iloc[-1]
    return float((hist <= val).mean() * 100.0)


def risk_label(score: float) -> tuple[str, str]:
    if pd.isna(score):
        return "⚪", "داده ناکافی"
    if score <= 33:
        return "🟢", "سبز"
    if score <= 66:
        return "🟡", "زرد"
    return "🔴", "قرمز"


def _weighted_mean(items: Iterable[tuple[float, float]]) -> float:
    valid = [(v, w) for v, w in items if v is not None and not pd.isna(v)]
    if not valid:
        return np.nan
    return float(sum(v * w for v, w in valid) / sum(w for _, w in valid))


def calculate_snapshot(data: dict[str, pd.Series]) -> dict:
    cpi_yoy = yoy(data.get("cpi"), 12)
    core_yoy = yoy(data.get("core_cpi"), 12)
    cpi_3m = annualized_3m_inflation(data.get("cpi"))
    gdp_yoy = yoy(data.get("gdp"), 4)
    ip_yoy = yoy(data.get("industrial_production"), 12)
    m2_yoy = yoy(data.get("m2"), 12)
    credit_yoy = yoy(data.get("bank_credit"), 52)
    unemployment = latest(data.get("unemployment"))
    unemployment_3m = change_over_months(data.get("unemployment"), 3, pct=False)
    policy_rate = latest(data.get("policy_rate"))
    yield_curve = latest(data.get("yield_curve"))
    real_yield = latest(data.get("real_yield"))
    stress = latest(data.get("financial_stress"))
    oil_3m = change_over_months(data.get("brent"), 3, pct=True)
    copper_3m = change_over_months(data.get("copper"), 3, pct=True)
    gold_3m = change_over_months(data.get("gold"), 3, pct=True)
    dollar_3m = change_over_months(data.get("dollar"), 3, pct=True)
    gscpi = latest(data.get("gscpi"))
    gpr_pct = percentile_rank(data.get("gpr"), 10)
    uncertainty_pct = percentile_rank(data.get("policy_uncertainty"), 10)

    # Risk transforms: 0 = calm/constructive, 100 = high risk.
    gdp_risk = clip(50 - 18 * gdp_yoy) if not pd.isna(gdp_yoy) else np.nan
    ip_risk = clip(45 - 12 * ip_yoy) if not pd.isna(ip_yoy) else np.nan
    unemployment_risk = clip(25 + 95 * unemployment_3m) if not pd.isna(unemployment_3m) else np.nan
    growth_score = _weighted_mean([(gdp_risk, 0.4), (ip_risk, 0.35), (unemployment_risk, 0.25)])

    inflation_level_risk = clip(abs(cpi_yoy - 2.0) * 18 + max(cpi_yoy - 3.0, 0) * 12) if not pd.isna(cpi_yoy) else np.nan
    core_risk = clip(abs(core_yoy - 2.0) * 15 + max(core_yoy - 3.0, 0) * 12) if not pd.isna(core_yoy) else np.nan
    momentum_risk = clip(20 + max(cpi_3m - 2.5, 0) * 14) if not pd.isna(cpi_3m) else np.nan
    inflation_score = _weighted_mean([(inflation_level_risk, 0.4), (core_risk, 0.35), (momentum_risk, 0.25)])

    curve_risk = clip(35 - 45 * yield_curve) if not pd.isna(yield_curve) else np.nan
    real_yield_risk = clip(25 + 18 * max(real_yield - 1.0, 0)) if not pd.isna(real_yield) else np.nan
    stress_risk = clip(45 + 20 * stress) if not pd.isna(stress) else np.nan
    credit_risk = clip(50 - 8 * credit_yoy) if not pd.isna(credit_yoy) else np.nan
    financial_score = _weighted_mean([(curve_risk, 0.3), (real_yield_risk, 0.2), (stress_risk, 0.25), (credit_risk, 0.25)])

    oil_risk = clip(25 + max(oil_3m, 0) * 2.2) if not pd.isna(oil_3m) else np.nan
    gscpi_risk = clip(50 + 22 * gscpi) if not pd.isna(gscpi) else np.nan
    gpr_risk = gpr_pct
    uncertainty_risk = uncertainty_pct
    supply_score = _weighted_mean([(oil_risk, 0.25), (gscpi_risk, 0.30), (gpr_risk, 0.25), (uncertainty_risk, 0.20)])

    overall = _weighted_mean([(growth_score, 0.25), (inflation_score, 0.25), (financial_score, 0.25), (supply_score, 0.25)])

    # Cycle and monetary stance.
    if not pd.isna(gdp_yoy) and not pd.isna(ip_yoy) and not pd.isna(unemployment_3m):
        if gdp_yoy < 0 or (ip_yoy < -2 and unemployment_3m > 0.35):
            cycle = "انقباض / خطر رکود"
        elif gdp_yoy >= 2 and ip_yoy > 0 and unemployment_3m <= 0.2:
            cycle = "رشد / توسعه"
        elif gdp_yoy > 0 and (ip_yoy < 0 or unemployment_3m > 0.2):
            cycle = "کاهش شتاب رشد"
        else:
            cycle = "بازیابی / وضعیت مختلط"
    else:
        cycle = "داده ناکافی"

    real_policy_rate = policy_rate - cpi_yoy if not pd.isna(policy_rate) and not pd.isna(cpi_yoy) else np.nan
    if pd.isna(real_policy_rate):
        monetary_stance = "داده ناکافی"
    elif real_policy_rate > 1.25:
        monetary_stance = "انقباضی"
    elif real_policy_rate < -0.5:
        monetary_stance = "انبساطی"
    else:
        monetary_stance = "خنثی تا محدودکننده"

    scores = {
        "growth": growth_score,
        "inflation": inflation_score,
        "financial": financial_score,
        "supply": supply_score,
        "overall": overall,
    }
    highest_key = max((k for k in ("growth", "inflation", "financial", "supply") if not pd.isna(scores[k])), key=lambda k: scores[k], default=None)
    risk_names = {
        "growth": "افت رشد و بازار کار",
        "inflation": "ماندگاری یا بازگشت تورم",
        "financial": "سخت‌تر شدن اعتبار و شرایط مالی",
        "supply": "شوک انرژی، زنجیره تأمین یا ژئوپلیتیک",
    }
    biggest_risk = risk_names.get(highest_key, "داده ناکافی")

    metrics = {
        "policy_rate": policy_rate,
        "real_policy_rate": real_policy_rate,
        "cpi_yoy": cpi_yoy,
        "core_yoy": core_yoy,
        "cpi_3m_ann": cpi_3m,
        "gdp_yoy": gdp_yoy,
        "ip_yoy": ip_yoy,
        "unemployment": unemployment,
        "unemployment_3m": unemployment_3m,
        "yield_curve": yield_curve,
        "m2_yoy": m2_yoy,
        "credit_yoy": credit_yoy,
        "real_yield": real_yield,
        "financial_stress": stress,
        "oil_3m": oil_3m,
        "copper_3m": copper_3m,
        "gold_3m": gold_3m,
        "dollar_3m": dollar_3m,
        "gscpi": gscpi,
        "gpr_percentile": gpr_pct,
        "uncertainty_percentile": uncertainty_pct,
    }

    traffic = build_traffic_table(metrics, scores)
    narratives = build_narratives(cycle, monetary_stance, biggest_risk, metrics, scores)
    business = build_business_impacts(metrics, scores)
    return {
        "metrics": metrics,
        "scores": scores,
        "cycle": cycle,
        "monetary_stance": monetary_stance,
        "biggest_risk": biggest_risk,
        "traffic": traffic,
        "narratives": narratives,
        "business": business,
    }


def build_traffic_table(metrics: dict, scores: dict) -> pd.DataFrame:
    indicator_scores = {
        "رشد GDP": clip(50 - 18 * metrics["gdp_yoy"]) if not pd.isna(metrics["gdp_yoy"]) else np.nan,
        "تورم کل": clip(abs(metrics["cpi_yoy"] - 2) * 18 + max(metrics["cpi_yoy"] - 3, 0) * 12) if not pd.isna(metrics["cpi_yoy"]) else np.nan,
        "بازار کار": clip(25 + 95 * metrics["unemployment_3m"]) if not pd.isna(metrics["unemployment_3m"]) else np.nan,
        "منحنی بازده": clip(35 - 45 * metrics["yield_curve"]) if not pd.isna(metrics["yield_curve"]) else np.nan,
        "نقدینگی M2": clip(45 - 5 * metrics["m2_yoy"]) if not pd.isna(metrics["m2_yoy"]) else np.nan,
        "اعتبار بانکی": clip(50 - 8 * metrics["credit_yoy"]) if not pd.isna(metrics["credit_yoy"]) else np.nan,
        "نفت": clip(25 + max(metrics["oil_3m"], 0) * 2.2) if not pd.isna(metrics["oil_3m"]) else np.nan,
        "مس": clip(50 - 2.0 * metrics["copper_3m"]) if not pd.isna(metrics["copper_3m"]) else np.nan,
        "دلار": clip(35 + 3.0 * max(metrics["dollar_3m"], 0)) if not pd.isna(metrics["dollar_3m"]) else np.nan,
        "زنجیره تأمین": clip(50 + 22 * metrics["gscpi"]) if not pd.isna(metrics["gscpi"]) else np.nan,
        "ژئوپلیتیک": metrics["gpr_percentile"],
    }
    rows = []
    for name, score in indicator_scores.items():
        icon, label = risk_label(score)
        rows.append({"شاخص": name, "چراغ": icon, "وضعیت": label, "امتیاز ریسک": None if pd.isna(score) else round(score), "جهت": _direction_text(name, metrics)})
    return pd.DataFrame(rows)


def _direction_text(name: str, m: dict) -> str:
    mapping = {
        "رشد GDP": m.get("gdp_yoy"),
        "تورم کل": m.get("cpi_3m_ann"),
        "بازار کار": m.get("unemployment_3m"),
        "منحنی بازده": m.get("yield_curve"),
        "نقدینگی M2": m.get("m2_yoy"),
        "اعتبار بانکی": m.get("credit_yoy"),
        "نفت": m.get("oil_3m"),
        "مس": m.get("copper_3m"),
        "دلار": m.get("dollar_3m"),
        "زنجیره تأمین": m.get("gscpi"),
        "ژئوپلیتیک": m.get("gpr_percentile"),
    }
    v = mapping.get(name)
    if v is None or pd.isna(v):
        return "نامشخص"
    if name in {"بازار کار", "تورم کل", "دلار", "نفت", "زنجیره تأمین", "ژئوپلیتیک"}:
        return "فشار بیشتر" if v > 0 else "فشار کمتر"
    return "رو به بهبود" if v > 0 else "رو به تضعیف"


def build_narratives(cycle: str, stance: str, biggest_risk: str, m: dict, scores: dict) -> dict[str, str]:
    if "رشد" in cycle:
        temperature = "اقتصاد هنوز در فاز رشد است، اما باید کیفیت رشد و بازار کار هم‌زمان کنترل شود."
    elif "کاهش" in cycle:
        temperature = "اقتصاد در حال سردتر شدن است و شتاب فعالیت کاهش یافته است."
    elif "انقباض" in cycle:
        temperature = "ترکیب رشد و بازار کار هشدار انقباضی می‌دهد."
    else:
        temperature = "سیگنال‌های رشد مختلط‌اند و برای نتیجه قطعی، انتشارهای بعدی لازم است."

    money = {
        "انقباضی": "پول همچنان گران است؛ نرخ حقیقی مثبت و شرایط مالی محدودکننده‌اند.",
        "انبساطی": "شرایط پولی نسبتاً آسان است و هزینه واقعی پول پایین است.",
        "خنثی تا محدودکننده": "شرایط پولی نه کاملاً انبساطی است و نه به‌شدت انقباضی؛ جهت بعدی تورم تعیین‌کننده است.",
    }.get(stance, "برای ارزیابی هزینه پول، داده کافی وجود ندارد.")

    confidence = sum(not pd.isna(v) for v in m.values()) / max(1, len(m))
    conf_text = "بالا" if confidence >= 0.8 else "متوسط" if confidence >= 0.6 else "پایین"
    return {
        "temperature": temperature,
        "money": money,
        "risk": f"مهم‌ترین ریسک شش‌ماهه در مدل فعلی: {biggest_risk}.",
        "confidence": f"درجه پوشش داده و اطمینان محاسباتی: {conf_text} ({confidence:.0%}).",
    }


def build_business_impacts(m: dict, scores: dict) -> pd.DataFrame:
    impacts = []

    dollar = m.get("dollar_3m")
    if not pd.isna(dollar):
        if dollar > 3:
            impacts.append(("واردات و ارز", "فشار افزایشی", "تقویت دلار معمولاً هزینه واردات و تأمین مالی دلاری را بالا می‌برد.", "افزایش پوشش ارزی و کوتاه‌کردن اعتبار قیمت فروش"))
        elif dollar < -3:
            impacts.append(("واردات و ارز", "فرصت نسبی", "تضعیف دلار بخشی از فشار خرید خارجی را کاهش می‌دهد.", "بازبینی زمان خرید و تثبیت قیمت تأمین‌کننده"))
        else:
            impacts.append(("واردات و ارز", "خنثی", "حرکت سه‌ماهه دلار محدود بوده است.", "حفظ پوشش عادی"))

    oil = m.get("oil_3m")
    if not pd.isna(oil):
        impacts.append(("انرژی و حمل", "فشار افزایشی" if oil > 10 else "قابل کنترل", "رشد نفت می‌تواند کرایه، سوخت و هزینه مواد انرژی‌بر را بالا ببرد." if oil > 10 else "تغییر نفت در محدوده‌ای است که فعلاً شوک بزرگ هزینه‌ای نشان نمی‌دهد.", "مذاکره قرارداد حمل و بودجه سناریویی"))

    copper = m.get("copper_3m")
    if not pd.isna(copper):
        impacts.append(("صنعت و قطعات", "تقاضای قوی / هزینه بالاتر" if copper > 8 else "تقاضای ضعیف‌تر" if copper < -8 else "متعادل", "مس هم‌زمان پیام تقاضای صنعتی و فشار هزینه مواد اولیه را منتقل می‌کند.", "کنترل قیمت خرید و موجودی مواد فلزی"))

    financial = scores.get("financial")
    if not pd.isna(financial):
        impacts.append(("تأمین مالی", "پرریسک" if financial > 66 else "احتیاط" if financial > 33 else "سالم", "امتیاز شرایط مالی از منحنی بازده، بازده حقیقی، تنش مالی و اعتبار بانکی ساخته شده است.", "حفظ نقدینگی و تنوع‌بخشی به منابع تأمین مالی"))

    supply = scores.get("supply")
    if not pd.isna(supply):
        impacts.append(("زنجیره تأمین", "پرریسک" if supply > 66 else "احتیاط" if supply > 33 else "سالم", "ریسک عرضه از نفت، فشار زنجیره تأمین، عدم‌قطعیت و ژئوپلیتیک ترکیب می‌شود.", "افزایش موجودی اقلام بحرانی و تأمین‌کننده جایگزین"))

    return pd.DataFrame(impacts, columns=["حوزه", "وضعیت", "تفسیر", "اقدام پیشنهادی"])


def format_number(value: float, decimals: int = 1, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:,.{decimals}f}{suffix}"


def make_demo_data(end: str = "2026-06-30") -> dict[str, pd.Series]:
    """Deterministic synthetic fallback for local preview/tests only."""
    rng = np.random.default_rng(42)
    end_ts = pd.Timestamp(end)
    monthly = pd.date_range(end=end_ts, periods=180, freq="MS")
    quarterly = pd.date_range(end=end_ts, periods=60, freq="QS")
    daily = pd.date_range(end=end_ts, periods=1300, freq="B")
    weekly = pd.date_range(end=end_ts, periods=260, freq="W-FRI")

    cpi = pd.Series(240 * np.exp(np.cumsum(0.002 + rng.normal(0, 0.0006, len(monthly)))), index=monthly)
    core = pd.Series(245 * np.exp(np.cumsum(0.0021 + rng.normal(0, 0.0004, len(monthly)))), index=monthly)
    gdp = pd.Series(18000 * np.exp(np.cumsum(0.005 + rng.normal(0, 0.004, len(quarterly)))), index=quarterly)
    ip = pd.Series(100 + np.cumsum(rng.normal(0.08, 0.35, len(monthly))), index=monthly)
    unemp = pd.Series(np.clip(4.2 + np.cumsum(rng.normal(0, 0.035, len(monthly))), 3.1, 8.0), index=monthly)
    fed = pd.Series(np.clip(2.5 + np.sin(np.linspace(0, 10, len(monthly))) * 1.5, 0, 6), index=monthly)
    m2 = pd.Series(14000 * np.exp(np.cumsum(0.004 + rng.normal(0, 0.002, len(monthly)))), index=monthly)
    credit = pd.Series(15000 * np.exp(np.cumsum(0.0012 + rng.normal(0, 0.001, len(weekly)))), index=weekly)
    ycurve = pd.Series(np.sin(np.linspace(0, 14, len(daily))) * 1.2 + rng.normal(0, 0.12, len(daily)), index=daily)
    brent = pd.Series(np.maximum(25, 75 + np.cumsum(rng.normal(0, 0.8, len(daily)))), index=daily)
    copper = pd.Series(np.maximum(4000, 8500 + np.cumsum(rng.normal(5, 70, len(monthly)))), index=monthly)
    gold = pd.Series(np.maximum(800, 1800 + np.cumsum(rng.normal(0.7, 8, len(daily)))), index=daily)
    dollar = pd.Series(100 + np.cumsum(rng.normal(0, 0.12, len(daily))), index=daily)
    real_yield = pd.Series(1.2 + np.sin(np.linspace(0, 8, len(daily))) * 0.8, index=daily)
    stress = pd.Series(rng.normal(-0.2, 0.45, len(weekly)), index=weekly)
    uncertainty = pd.Series(np.maximum(50, 180 + rng.normal(0, 45, len(monthly))), index=monthly)
    gscpi = pd.Series(rng.normal(0, 0.65, len(monthly)), index=monthly)
    gpr = pd.Series(np.maximum(20, 100 + rng.normal(0, 28, len(monthly))), index=monthly)

    return {
        "policy_rate": fed,
        "cpi": cpi,
        "core_cpi": core,
        "gdp": gdp,
        "unemployment": unemp,
        "yield_curve": ycurve,
        "m2": m2,
        "bank_credit": credit,
        "industrial_production": ip,
        "brent": brent,
        "copper": copper,
        "gold": gold,
        "dollar": dollar,
        "real_yield": real_yield,
        "financial_stress": stress,
        "policy_uncertainty": uncertainty,
        "gscpi": gscpi,
        "gpr": gpr,
    }
