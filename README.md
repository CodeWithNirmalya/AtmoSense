<div align="center">

<!-- BANNER -->
<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:1a6b4a,50:2e86c1,100:3bf0c4&height=200&section=header&text=AtmoSense&fontSize=72&fontColor=ffffff&fontAlignY=38&desc=Intelligent%20Weather%20Forecasting%20%C2%B7%20Polynomial%20Regression%20%C2%B7%20ERA5%20%C2%B7%20Open-Meteo&descAlignY=60&descSize=16&animation=fadeIn"/>

<!-- BADGES -->
<p>
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Open--Meteo-ERA5-00BFFF?style=for-the-badge&logo=cloudfoundry&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge"/>
</p>

<p>
  <img src="https://img.shields.io/badge/Status-Active-3bf0c4?style=flat-square"/>
  <img src="https://img.shields.io/badge/Version-2.0.0-4da8f5?style=flat-square"/>
  <img src="https://img.shields.io/badge/Made%20with-%E2%99%A5%20by%20Nirmalya%20Raja-e05252?style=flat-square"/>
</p>

<br/>

> **AtmoSense** combines a *polynomial regression model* trained on 120 days of ERA5 reanalysis data with professional NWP forecasts — delivering tomorrow's temperature prediction alongside a real-time accuracy comparison, all wrapped in a beautiful dual-mode interface.

<br/>

<!-- DEMO SCREENSHOT PLACEHOLDER -->
<!--
<img src="assets/demo_light.png" width="49%" alt="Light Mode"/>
<img src="assets/demo_dark.png" width="49%" alt="Dark Mode"/>
-->

</div>

---

## ✨ Highlights

| | Feature | Description |
|---|---|---|
| 🧮 | **Polynomial Regression** | Degree 1–5 curve fitted via scikit-learn on 120 days of ERA5 daily min/max temps |
| 🛰️ | **ERA5 Reanalysis** | Gold-standard ECMWF atmospheric reanalysis data via Open-Meteo archive API |
| 🎯 | **NWP Comparison** | Model predictions benchmarked live against Open-Meteo professional forecasts |
| 🌡️ | **Comfort Index** | Unique thermal comfort score (0–100) with a visual spectrum gauge |
| 🔬 | **Advanced Mode** | Raw data tables, model coefficients, configurable hyperparameters & trend charts |
| 🌗 | **Light / Dark Theme** | Editorial off-white light mode + deep atmospheric dark mode |
| 🌐 | **HTML Demo** | Zero-install interactive demo — just open `atmosense_v2.html` in any browser |

---

## 🗂️ Project Structure

```
atmosense/
│
├── 📄 weather_backend.py      ← Core logic: geocoding, ERA5 fetch, regression
├── 📄 weather_app.py          ← Streamlit UI (Normal + Advanced user modes)
├── 📄 atmosense_v2.html       ← Standalone HTML demo (no Python required)
├── 📄 requirements.txt        ← All pip dependencies
└── 📄 README.md
```

---

## 🚀 Quick Start

### 1 · Clone the repository

```bash
git clone https://github.com/<your-username>/atmosense.git
cd atmosense
```

### 2 · Install dependencies

```bash
pip install -r requirements.txt
```

> **Python 3.9+** required. It is recommended to use a virtual environment:
> ```bash
> python -m venv venv
> source venv/bin/activate      # macOS / Linux
> venv\Scripts\activate         # Windows
> pip install -r requirements.txt
> ```

### 3 · Run the Streamlit app

```bash
streamlit run weather_app.py
```

The app opens automatically at **http://localhost:8501** ✅

### 4 · Or open the HTML demo instantly

```bash
# No Python, no server — just open in browser
open atmosense_v2.html          # macOS
start atmosense_v2.html         # Windows
xdg-open atmosense_v2.html      # Linux
```

---

## 🧠 How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AtmoSense Pipeline                          │
└─────────────────────────────────────────────────────────────────────┘

  City Name
      │
      ▼
  ┌───────────────────────┐
  │  Open-Meteo Geocoding │  ─── lat, lon, timezone
  └───────────────────────┘
              │
      ┌───────┴────────┐
      ▼                ▼
  ┌──────────┐    ┌──────────────┐
  │  ERA5    │    │  NWP 7-day   │
  │ Archive  │    │  Forecast    │
  │ 120 days │    │  Open-Meteo  │
  └────┬─────┘    └──────┬───────┘
       │                 │
       ▼                 │
  ┌──────────────────┐   │
  │ Polynomial Fit   │   │
  │  deg 1–5 curve   │   │
  │  scikit-learn    │   │
  └────┬─────────────┘   │
       │                 │
       ▼                 ▼
  ┌──────────────────────────────┐
  │  Prediction vs NWP Δ Delta   │
  │  Comfort Index · 7-day strip │
  │  Trend Chart · Coefficients  │
  └──────────────────────────────┘
```

### Backend Functions

| Function | Description |
|---|---|
| `geocode_city(name)` | Resolves city name → latitude, longitude, timezone |
| `fetch_history(lat, lon, start, end, tz)` | Downloads ERA5 daily max/min temps, cleans NaN rows |
| `fetch_forecast(lat, lon, tz)` | Fetches 7-day NWP forecast from Open-Meteo |
| `build_xy(df)` | Converts dates → integer day-index features for regression |
| `fit_poly_regression(X, y, degree)` | Builds `PolynomialFeatures → LinearRegression` sklearn pipeline |

---

## 🖥️ Interface Modes

### 🌤 Normal User Mode
Clean consumer-facing weather dashboard with:
- **Hero temperature cards** — model prediction vs official forecast
- **Stat row** — tomorrow's max/min, 7-day high/low, accuracy estimate
- **7-day forecast strip** — colour-coded day tiles with weather icons
- **Comfort Index** — thermal comfort spectrum gauge (unique to AtmoSense)

### 🔬 Advanced User Mode
Data science dashboard with:
- **Hyperparameter sliders** — polynomial degree (1–5), training window (30–180 days)
- **Interactive Plotly chart** — ERA5 historical range, model curve overlay, forecast
- **Model coefficients table** — β₀ intercept and all polynomial terms
- **Raw data table** — last 14 days of ERA5 daily records with temp colouring
- **Delta analysis** — agreement score, assessment badge, progress bar

---

## 📦 Dependencies

```txt
streamlit>=1.30.0
requests>=2.31.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
plotly>=5.18.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🌆 Supported Cities

| City | State | Coordinates |
|---|---|---|
| 🌊 Kolkata | West Bengal | 22.57°N · 88.36°E |
| 🏛️ Delhi | Delhi NCT | 28.66°N · 77.22°E |
| 🌊 Mumbai | Maharashtra | 19.07°N · 72.88°E |
| ☀️ Chennai | Tamil Nadu | 13.08°N · 80.27°E |
| 🌿 Bengaluru | Karnataka | 12.97°N · 77.59°E |
| 🏰 Hyderabad | Telangana | 17.38°N · 78.47°E |

---

## ⚡ CLI Reference

```bash
# Run on default port 8501
streamlit run weather_app.py

# Run on a custom port
streamlit run weather_app.py --server.port 8502

# Run and expose on local network (share with teammates)
streamlit run weather_app.py --server.address 0.0.0.0

# Stop the app
Ctrl + C
```

---

## 🔧 Troubleshooting

<details>
<summary><b>❌ "streamlit: command not found"</b></summary>

```bash
pip install streamlit
# OR use the module form
python -m streamlit run weather_app.py
```
</details>

<details>
<summary><b>❌ "ModuleNotFoundError: No module named 'sklearn'"</b></summary>

```bash
pip install scikit-learn
```
</details>

<details>
<summary><b>❌ Port 8501 already in use</b></summary>

```bash
streamlit run weather_app.py --server.port 8502
```
</details>

<details>
<summary><b>❌ Browser doesn't open automatically</b></summary>

Navigate manually to `http://localhost:8501` in your browser.
</details>

<details>
<summary><b>❌ Historical data unavailable / API timeout</b></summary>

The ERA5 archive has a ~3-day lag. Dates are automatically adjusted. If you see a timeout, wait 30 seconds and retry — Open-Meteo is a free public API with occasional rate limits.
</details>

---

## 🗺️ Roadmap

- [ ] Add hourly temperature granularity
- [ ] Humidity, wind speed, and precipitation fields
- [ ] Multiple regression model comparison (Ridge, Lasso, SVR)
- [ ] Exportable PDF forecast report
- [ ] City search (free text geocoding)
- [ ] Deploy to Streamlit Cloud with one-click button

---

## 📜 License

This project is released under the **MIT License** — free to use, modify, and distribute with attribution.

---

## 🙏 Acknowledgements

| Resource | Usage |
|---|---|
| [Open-Meteo](https://open-meteo.com) | Free weather & archive APIs — no API key required |
| [ECMWF ERA5](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5) | Atmospheric reanalysis dataset |
| [scikit-learn](https://scikit-learn.org) | Polynomial regression pipeline |
| [Streamlit](https://streamlit.io) | Web app framework |
| [Plotly](https://plotly.com) | Interactive visualisation |
| [Chart.js](https://chartjs.org) | HTML demo charting |

---

<div align="center">

<br/>

**Built with focus, curiosity, and a love for atmospheric science.**

<br/>

```
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   "The atmosphere is not a province — it is a planet."  ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
```

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:3bf0c4,50:2e86c1,100:1a6b4a&height=120&section=footer"/>

**Designed & developed with ♥ by [Nirmalya Raja](https://github.com/nirmalyaraja)**

*Powered by Open-Meteo Free APIs · ERA5 Reanalysis · scikit-learn · v2.0 · 2025*

</div>
