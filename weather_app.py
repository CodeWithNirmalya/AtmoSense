"""
╔══════════════════════════════════════════════════════════════╗
║          AtmoSense — Streamlit Weather Forecast App          ║
║   Polynomial Regression · Open-Meteo APIs · Interactive UI  ║
╚══════════════════════════════════════════════════════════════╝

HOW TO RUN:
    pip install streamlit requests numpy pandas scikit-learn plotly
    streamlit run weather_app.py
"""

# ─── Imports ──────────────────────────────────────────────────────────────────
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AtmoSense · Weather Intelligence",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: #0a0e1a;
        color: #e8edf5;
    }
    .stApp { background: #0a0e1a; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #111828 !important;
        border-right: 1px solid rgba(255,255,255,0.07);
    }
    section[data-testid="stSidebar"] * { color: #e8edf5 !important; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3bf0c4, #4da8f5) !important;
        color: #0a0e1a !important;
        font-family: 'Bebas Neue', sans-serif !important;
        font-size: 18px !important;
        letter-spacing: 3px !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 14px 32px !important;
        width: 100% !important;
        transition: opacity 0.2s !important;
    }
    .stButton > button:hover { opacity: 0.9 !important; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #111828;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 20px !important;
    }
    [data-testid="metric-container"] label {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 10px !important;
        letter-spacing: 3px !important;
        color: #6b7a96 !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-family: 'Bebas Neue', sans-serif !important;
        font-size: 48px !important;
        color: #3bf0c4 !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricDelta"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 12px !important;
    }

    /* Selectbox, radio */
    .stSelectbox > div, .stRadio > div {
        background: #111828 !important;
        border-color: rgba(255,255,255,0.07) !important;
        border-radius: 10px !important;
    }

    /* Divider */
    hr { border-color: rgba(255,255,255,0.07) !important; }

    /* Info / success / warning boxes */
    .stAlert { border-radius: 10px !important; }

    /* Spinner */
    .stSpinner > div { border-top-color: #3bf0c4 !important; }

    /* Hide Streamlit default header/footer */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Custom HTML blocks ── */
    .atmo-header {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 56px;
        letter-spacing: 5px;
        background: linear-gradient(135deg, #3bf0c4, #4da8f5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1;
        margin-bottom: 4px;
    }
    .atmo-sub {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        letter-spacing: 3px;
        color: #6b7a96;
        text-transform: uppercase;
        margin-bottom: 32px;
    }
    .section-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        letter-spacing: 3px;
        color: #6b7a96;
        text-transform: uppercase;
        margin: 24px 0 10px;
    }
    .card {
        background: #111828;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 12px;
    }
    .card-accent { border-color: rgba(59,240,196,0.25); }
    .card-blue   { border-color: rgba(77,168,245,0.25); }
    .big-temp {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 72px;
        line-height: 1;
        letter-spacing: -1px;
    }
    .temp-green { color: #3bf0c4; }
    .temp-blue  { color: #4da8f5; }
    .temp-warn  { color: #f5a623; }
    .card-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        letter-spacing: 3px;
        color: #6b7a96;
        margin-bottom: 8px;
    }
    .badge {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        letter-spacing: 2px;
        padding: 4px 12px;
        border-radius: 20px;
        border: 1px solid rgba(59,240,196,0.3);
        color: #3bf0c4;
        background: rgba(59,240,196,0.05);
    }
    .day-strip {
        display: flex;
        gap: 8px;
        overflow-x: auto;
        padding-bottom: 8px;
    }
    .day-tile {
        min-width: 80px;
        background: #111828;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 12px 10px;
        text-align: center;
        flex-shrink: 0;
    }
    .day-tile.highlight { border-color: rgba(59,240,196,0.35); background: rgba(59,240,196,0.04); }
    .day-name { font-family: 'JetBrains Mono', monospace; font-size: 9px; letter-spacing: 2px; color: #6b7a96; margin-bottom: 6px; }
    .day-icon { font-size: 22px; margin-bottom: 6px; }
    .day-hi   { font-weight: 500; font-size: 15px; color: #e8edf5; }
    .day-lo   { font-size: 12px; font-weight: 300; color: #6b7a96; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  BACKEND — Open-Meteo API functions  (same logic as your original script)
# ══════════════════════════════════════════════════════════════════════════════

GEOCODE_URL  = "https://geocoding-api.open-meteo.com/v1/search"
HIST_URL     = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


@st.cache_data(ttl=3600, show_spinner=False)
def geocode_city(name: str) -> dict:
    """Converts a city name → lat, lon, timezone via Open-Meteo geocoding."""
    params = {"name": name, "count": 1, "language": "en", "format": "json"}
    r = requests.get(GEOCODE_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not data.get("results"):
        raise ValueError(f"Could not geocode city '{name}'. Try a different spelling.")
    res = data["results"][0]
    return {
        "name":      res.get("name"),
        "latitude":  res["latitude"],
        "longitude": res["longitude"],
        "timezone":  res.get("timezone", "auto"),
        "country":   res.get("country"),
        "admin1":    res.get("admin1"),
    }


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_history(lat: float, lon: float, start_date: date, end_date: date, timezone: str = "auto") -> pd.DataFrame:
    """Download daily historical max/min temperatures, returns cleaned DataFrame."""
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start_date.isoformat(),
        "end_date":   end_date.isoformat(),
        "daily":      ["temperature_2m_max", "temperature_2m_min"],
        "timezone":   timezone,
        "models":     "era5",
    }
    r = requests.get(HIST_URL, params=params, timeout=60)
    r.raise_for_status()
    daily = r.json().get("daily", {})
    if not daily or "time" not in daily:
        raise RuntimeError("Historical data not available for the requested range.")

    df = pd.DataFrame(daily)
    for col in ["temperature_2m_max", "temperature_2m_min"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["temp_mean"] = (df["temperature_2m_max"] + df["temperature_2m_min"]) / 2.0
    df["date"]      = pd.to_datetime(df["time"])
    df = df.dropna(subset=["temp_mean"]).reset_index(drop=True)
    return df[["date", "temperature_2m_min", "temperature_2m_max", "temp_mean"]]


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_forecast(lat: float, lon: float, timezone: str = "auto") -> pd.DataFrame:
    """Fetch the next 7 days of official forecast, returns DataFrame."""
    params = {
        "latitude":     lat,
        "longitude":    lon,
        "daily":        ["temperature_2m_max", "temperature_2m_min"],
        "forecast_days": 7,
        "timezone":     timezone,
    }
    r = requests.get(FORECAST_URL, params=params, timeout=30)
    r.raise_for_status()
    daily = r.json().get("daily", {})
    if not daily or "time" not in daily:
        return pd.DataFrame(columns=["date", "temperature_2m_min", "temperature_2m_max", "temp_mean"])
    df = pd.DataFrame(daily)
    df["date"]             = pd.to_datetime(df["time"])
    df["temperature_2m_max"] = pd.to_numeric(df["temperature_2m_max"], errors="coerce")
    df["temperature_2m_min"] = pd.to_numeric(df["temperature_2m_min"], errors="coerce")
    df["temp_mean"]        = (df["temperature_2m_max"] + df["temperature_2m_min"]) / 2.0
    return df[["date", "temperature_2m_min", "temperature_2m_max", "temp_mean"]]


def build_xy(df: pd.DataFrame):
    """Prepare (X, y, base_date) for sklearn regression."""
    df = df.sort_values("date").reset_index(drop=True)
    base = df["date"].min()
    df["x"] = (df["date"] - base).dt.days.astype(int)
    X = df[["x"]].values
    y = df["temp_mean"].values.astype(float)
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return df.loc[mask].reset_index(drop=True), X[mask], y[mask], base


def fit_poly_regression(X: np.ndarray, y: np.ndarray, degree: int = 3) -> Pipeline:
    """Build and fit a sklearn polynomial regression pipeline."""
    model = Pipeline([
        ("poly",   PolynomialFeatures(degree=degree, include_bias=False)),
        ("linreg", LinearRegression()),
    ])
    model.fit(X, y)
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def temp_icon(max_c: float) -> str:
    if max_c > 38: return "🌡️"
    if max_c > 32: return "☀️"
    if max_c > 26: return "🌤️"
    if max_c > 20: return "⛅"
    return "🌧️"


def build_plotly_chart(hist_df: pd.DataFrame, fc_df: pd.DataFrame, model: Pipeline, base_date: pd.Timestamp) -> go.Figure:
    """Create a Plotly chart overlaying historical data, model curve, and forecast."""
    # Model curve (sampled every 2 days across full range)
    n_hist = len(hist_df)
    n_fc   = len(fc_df)
    xs_curve = np.arange(0, n_hist + n_fc, 2).reshape(-1, 1)
    ys_curve = model.predict(xs_curve)
    dates_curve = [base_date + pd.Timedelta(days=int(x)) for x in xs_curve.flatten()]

    fig = go.Figure()

    # Historical shaded area
    fig.add_trace(go.Scatter(
        x=hist_df["date"], y=hist_df["temperature_2m_max"],
        name="Hist Max", mode="lines",
        line=dict(color="rgba(59,240,196,0.15)", width=0),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=hist_df["date"], y=hist_df["temperature_2m_min"],
        name="Hist Min", mode="lines",
        line=dict(color="rgba(59,240,196,0.15)", width=0),
        fill="tonexty",
        fillcolor="rgba(59,240,196,0.06)",
        showlegend=False,
    ))

    # Historical mean
    fig.add_trace(go.Scatter(
        x=hist_df["date"], y=hist_df["temp_mean"],
        name="Historical Mean",
        mode="lines",
        line=dict(color="rgba(59,240,196,0.7)", width=1.5),
    ))

    # Model curve
    fig.add_trace(go.Scatter(
        x=dates_curve, y=ys_curve,
        name="Model Curve (deg 3)",
        mode="lines",
        line=dict(color="rgba(245,166,35,0.85)", width=2, dash="dot"),
    ))

    # 7-day forecast
    if not fc_df.empty:
        fig.add_trace(go.Scatter(
            x=fc_df["date"], y=fc_df["temp_mean"],
            name="7-Day Forecast",
            mode="lines+markers",
            line=dict(color="rgba(77,168,245,0.9)", width=2, dash="dash"),
            marker=dict(size=6, color="#4da8f5"),
        ))

    fig.update_layout(
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#111828",
        font=dict(family="JetBrains Mono", color="#6b7a96", size=11),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(
            orientation="h", y=1.08, x=0,
            font=dict(size=10),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            zerolinecolor="rgba(255,255,255,0.05)",
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            zerolinecolor="rgba(255,255,255,0.05)",
            title="°C",
        ),
        hovermode="x unified",
        height=320,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

CITIES = {
    "🌊 Kolkata":   "Kolkata",
    "🏛️ Delhi":     "Delhi",
    "🌊 Mumbai":    "Mumbai",
    "☀️ Chennai":   "Chennai",
    "🌿 Bengaluru": "Bengaluru",
    "🏰 Hyderabad": "Hyderabad",
}

with st.sidebar:
    st.markdown('<div class="atmo-header">AtmoSense</div>', unsafe_allow_html=True)
    st.markdown('<div class="atmo-sub">Weather Intelligence</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">// Select City</div>', unsafe_allow_html=True)
    city_label = st.radio("", list(CITIES.keys()), label_visibility="collapsed")
    city_name  = CITIES[city_label]

    st.markdown('<div class="section-label">// Model Settings</div>', unsafe_allow_html=True)
    poly_degree  = st.slider("Polynomial Degree", min_value=1, max_value=5, value=3, help="Higher = more flexible curve")
    history_days = st.slider("Training Days", min_value=30, max_value=180, value=120, step=10, help="Days of ERA5 history to train on")

    st.markdown("---")
    analyze_btn = st.button("⟶  ANALYZE WEATHER")

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:1px;color:#6b7a96;line-height:1.8;">
    DATA SOURCES<br>
    · Open-Meteo ERA5 Archive<br>
    · Open-Meteo NWP Forecast<br><br>
    MODEL<br>
    · Polynomial Regression<br>
    · scikit-learn Pipeline<br><br>
    BUILT WITH ♥ IN PYTHON
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PANEL — Header
# ══════════════════════════════════════════════════════════════════════════════

col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown('<div class="atmo-header">AtmoSense</div>', unsafe_allow_html=True)
    st.markdown('<div class="atmo-sub">Polynomial Regression · Weather Intelligence · Open-Meteo APIs</div>', unsafe_allow_html=True)
with col_h2:
    st.markdown(f"""
    <div style="text-align:right;font-family:'JetBrains Mono',monospace;font-size:11px;color:#6b7a96;padding-top:20px;letter-spacing:1px;">
    <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#3bf0c4;margin-right:6px;vertical-align:middle;"></span>LIVE DATA<br>
    {date.today().strftime('%d %b %Y').upper()}
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PANEL — Run when button clicked
# ══════════════════════════════════════════════════════════════════════════════

if analyze_btn:
    today    = date.today()
    tomorrow = today + timedelta(days=1)

    # ── Step 1: Geocode ──────────────────────────────────────────────────────
    with st.spinner(f"📡 Geocoding {city_name}..."):
        try:
            place = geocode_city(city_name)
        except Exception as e:
            st.error(f"⚠️ Geocoding failed: {e}")
            st.stop()

    lat, lon, tz = place["latitude"], place["longitude"], place["timezone"]
    parts = [p for p in [place.get("name"), place.get("admin1"), place.get("country")] if p]

    # ── Step 2: Fetch History ────────────────────────────────────────────────
    start    = today - timedelta(days=history_days)
    hist_end = today - timedelta(days=3)           # ERA5 has a ~3-day lag

    with st.spinner(f"📚 Fetching {history_days}-day historical data (ERA5)..."):
        try:
            hist_df = fetch_history(lat, lon, start, hist_end, tz)
        except Exception as e:
            st.error(f"⚠️ Historical data error: {e}")
            st.stop()

    if hist_df.empty or len(hist_df) < 5:
        st.error("❌ Not enough historical samples. Try a different city.")
        st.stop()

    # ── Step 3: Fetch Forecast ───────────────────────────────────────────────
    with st.spinner("🔭 Fetching 7-day official forecast..."):
        try:
            fc_df = fetch_forecast(lat, lon, tz)
        except Exception as e:
            st.warning(f"⚠️ Forecast fetch failed: {e}. Proceeding without comparison.")
            fc_df = pd.DataFrame(columns=["date", "temperature_2m_min", "temperature_2m_max", "temp_mean"])

    # ── Step 4: Fit Model ────────────────────────────────────────────────────
    with st.spinner(f"🔬 Fitting Polynomial Regression (degree={poly_degree})..."):
        hist_df, X, y, base_date = build_xy(hist_df)
        if len(y) < 5:
            st.error("❌ Too few clean samples to fit the model.")
            st.stop()
        model = fit_poly_regression(X, y, degree=poly_degree)

    # ── Step 5: Predict Tomorrow ─────────────────────────────────────────────
    x_tomorrow = np.array([[(pd.Timestamp(tomorrow) - base_date).days]])
    y_pred     = float(model.predict(x_tomorrow)[0])

    # Get official forecast value for tomorrow
    fc_val = None
    if not fc_df.empty:
        row = fc_df[fc_df["date"].dt.date == tomorrow]
        if not row.empty:
            fc_val = float(row["temp_mean"].iloc[0])

    # ════════════════════════════════════════════════════════════════════════
    #  RENDER RESULTS
    # ════════════════════════════════════════════════════════════════════════

    # Location Header
    loc_str   = ", ".join(parts)
    coord_str = f"{lat:.3f}°N  {lon:.3f}°E"
    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;">
      <div>
        <div style="font-family:'Bebas Neue',sans-serif;font-size:36px;letter-spacing:2px;color:#e8edf5;">{loc_str}</div>
        <div style="font-size:13px;color:#6b7a96;">Tomorrow · {tomorrow.strftime('%A, %d %B %Y')}</div>
      </div>
      <div class="badge">{coord_str}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tomorrow's Prediction Cards ──────────────────────────────────────────
    st.markdown('<div class="section-label">// Tomorrow\'s Prediction</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="card card-accent">
            <div class="card-label">// MODEL PREDICTION · POLY DEG {poly_degree}</div>
            <div class="big-temp temp-green">{y_pred:.1f}<span style="font-size:24px;vertical-align:super;font-weight:300;font-family:'DM Sans',sans-serif">°C</span></div>
            <div style="font-size:13px;color:#6b7a96;margin-top:8px;">Mean temperature · regression model</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        if fc_val is not None:
            st.markdown(f"""
            <div class="card card-blue">
                <div class="card-label">// OPEN-METEO OFFICIAL FORECAST</div>
                <div class="big-temp temp-blue">{fc_val:.1f}<span style="font-size:24px;vertical-align:super;font-weight:300;font-family:'DM Sans',sans-serif">°C</span></div>
                <div style="font-size:13px;color:#6b7a96;margin-top:8px;">Professional NWP model · official</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card">
                <div class="card-label">// OPEN-METEO OFFICIAL FORECAST</div>
                <div class="big-temp" style="color:#6b7a96;">N/A</div>
                <div style="font-size:13px;color:#6b7a96;margin-top:8px;">Forecast unavailable</div>
            </div>
            """, unsafe_allow_html=True)

    with col3:
        if fc_val is not None:
            diff      = y_pred - fc_val
            abs_diff  = abs(diff)
            sign      = "+" if diff >= 0 else ""
            if abs_diff < 0.5:   assess_icon, assess_txt, diff_color = "✅", "EXCELLENT MATCH", "#3bf0c4"
            elif abs_diff < 2.0: assess_icon, assess_txt, diff_color = "🟡", "GOOD AGREEMENT",  "#f5a623"
            else:                assess_icon, assess_txt, diff_color = "🔴", "HIGH DELTA",       "#f55f5f"

            accuracy = max(0, 100 - abs_diff * 15)
            st.markdown(f"""
            <div class="card">
                <div class="card-label">// DELTA (MODEL − FORECAST)</div>
                <div class="big-temp" style="color:{diff_color};">{sign}{diff:.2f}<span style="font-size:24px;vertical-align:super;font-weight:300;font-family:'DM Sans',sans-serif">°C</span></div>
                <div style="margin-top:10px;">
                    <div style="height:4px;background:#1a2235;border-radius:2px;overflow:hidden;">
                        <div style="height:100%;width:{accuracy:.0f}%;background:linear-gradient(90deg,#3bf0c4,#4da8f5);border-radius:2px;"></div>
                    </div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#6b7a96;letter-spacing:1px;margin-top:6px;">
                        {assess_icon} {assess_txt} · {accuracy:.0f}% AGREEMENT
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card">
                <div class="card-label">// DELTA</div>
                <div class="big-temp" style="color:#6b7a96;">—</div>
                <div style="font-size:13px;color:#6b7a96;margin-top:8px;">No forecast to compare</div>
            </div>
            """, unsafe_allow_html=True)

    # ── 7-Day Forecast Strip ─────────────────────────────────────────────────
    st.markdown('<div class="section-label">// 7-Day Forecast Strip</div>', unsafe_allow_html=True)

    if not fc_df.empty:
        day_tiles_html = '<div class="day-strip">'
        days_abbr = ['SUN', 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT']
        for _, row in fc_df.iterrows():
            is_tom = row["date"].date() == tomorrow
            label  = "TMW" if is_tom else days_abbr[row["date"].dayofweek]
            hi     = row["temperature_2m_max"]
            lo     = row["temperature_2m_min"]
            ico    = temp_icon(hi) if pd.notna(hi) else "❓"
            hl_cls = "highlight" if is_tom else ""
            day_tiles_html += f"""
            <div class="day-tile {hl_cls}">
                <div class="day-name">{label}</div>
                <div class="day-icon">{ico}</div>
                <div class="day-hi">{hi:.0f}°</div>
                <div class="day-lo">{lo:.0f}°</div>
            </div>"""
        day_tiles_html += '</div>'
        st.markdown(day_tiles_html, unsafe_allow_html=True)
    else:
        st.info("7-day forecast data unavailable.")

    # ── Historical + Model Chart ─────────────────────────────────────────────
    st.markdown('<div class="section-label">// Historical Temperature Trend · Model Curve</div>', unsafe_allow_html=True)
    fig = build_plotly_chart(hist_df, fc_df, model, base_date)
    st.plotly_chart(fig, use_container_width=True)

    # ── Raw Data Table (collapsible) ─────────────────────────────────────────
    with st.expander("📊 Raw Historical Data Table"):
        st.dataframe(
            hist_df[["date", "temperature_2m_min", "temperature_2m_max", "temp_mean"]]
            .rename(columns={"temperature_2m_min": "Min °C", "temperature_2m_max": "Max °C", "temp_mean": "Mean °C"})
            .set_index("date")
            .style.format("{:.2f}"),
            height=300,
        )

    # ── Model Coefficients ───────────────────────────────────────────────────
    with st.expander("🔬 Model Details"):
        linreg   = model.named_steps["linreg"]
        poly     = model.named_steps["poly"]
        feat_names = poly.get_feature_names_out(["x"])
        coef_df  = pd.DataFrame({"Feature": feat_names, "Coefficient": linreg.coef_})
        coef_df  = pd.concat([
            pd.DataFrame({"Feature": ["intercept"], "Coefficient": [linreg.intercept_]}),
            coef_df
        ], ignore_index=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div class="card" style="margin-top:0">
                <div class="card-label">// MODEL METADATA</div>
                <table style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#e8edf5;line-height:2;width:100%">
                    <tr><td style="color:#6b7a96">Algorithm</td><td>Polynomial Regression</td></tr>
                    <tr><td style="color:#6b7a96">Degree</td><td>{poly_degree}</td></tr>
                    <tr><td style="color:#6b7a96">Training Rows</td><td>{len(y)}</td></tr>
                    <tr><td style="color:#6b7a96">Training Days</td><td>{history_days}</td></tr>
                    <tr><td style="color:#6b7a96">Base Date</td><td>{base_date.date()}</td></tr>
                    <tr><td style="color:#6b7a96">Predicted °C</td><td style="color:#3bf0c4">{y_pred:.4f}</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.dataframe(coef_df.style.format({"Coefficient": "{:.6f}"}), height=200)

else:
    # ── Landing / Idle state ─────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:80px 40px;">
        <div style="font-size:80px;margin-bottom:24px;">🌡️</div>
        <div style="font-family:'Bebas Neue',sans-serif;font-size:32px;letter-spacing:3px;color:#6b7a96;">
            SELECT A CITY AND CLICK ANALYZE
        </div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:2px;color:#6b7a96;margin-top:12px;opacity:0.6;">
            POWERED BY ERA5 HISTORICAL DATA · OPEN-METEO FORECAST · POLYNOMIAL REGRESSION
        </div>
    </div>
    """, unsafe_allow_html=True)
