# NBA Analytics Explorer

Interactive NBA analytics app featuring head-to-head FG% comparisons, shot charts, player trends, team stats, and league leaders. Built with Python, Streamlit, and the NBA Stats API.

**[Try the live app →](#)** *(link updated after deployment)*

---

## Head-to-Head FG% Comparison

Smoothed heatmap showing where each player shoots better, colored by team colors with volume-weighted opacity. Auto-generated headlines and zone splits (paint, mid-range, 3PT).

- Compare any two players across any season back to 1996-97
- Cross-era matchups supported (e.g. Jordan '96 vs. Curry '16)
- Pre-cached data for instant loading; live API fallback for uncached players

---

## Features

| Tab | What it does |
|-----|-------------|
| **Head-to-Head** | Side-by-side FG% heatmap with team colors and zone splits |
| **Shot Chart** | Individual player shot density heatmap on half-court |
| **Player Trends** | Game-by-game stat trends with rolling averages and career arcs |
| **Teams & Standings** | League-wide team stats, filterable by conference/division |
| **League Leaders** | Top-N leaderboards for any stat category |

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### Pre-fetch shot data (recommended before first run)

```bash
# Current season — all active players (~8-10 min)
python -m scripts.prefetch_shots

# Historical legends — ~30 players across key seasons (~15-20 min)
python -m scripts.prefetch_historical
```

### Run the app

```bash
streamlit run app.py
```

---

## Project Structure

```
DataModeling/
├── app.py                        # Streamlit app (main entry point)
├── requirements.txt
├── .streamlit/config.toml        # Streamlit theme config
├── data/
│   ├── shot_cache/               # Pre-fetched Parquet files (player shots)
│   ├── consumer_expenditure_bls.csv
│   └── nba_birthplaces_by_country.csv
├── scripts/
│   ├── prefetch_shots.py         # Cache current-season active players
│   └── prefetch_historical.py    # Cache historical legends
└── src/
    ├── cache.py                  # Shot data cache layer (Parquet + API fallback)
    ├── config.py                 # Paths and settings
    ├── shot_chart_comparison.py  # Head-to-head FG% comparison engine
    ├── nba_shot_chart.py         # Individual shot chart heatmap
    ├── geo_bubble_map.py         # NBA birthplaces bubble map
    ├── stock_dashboard.py        # S&P 500 + VIX dashboard
    └── spending_treemap.py       # BLS-style spending treemap
```

---

## Data Sources

- **NBA:** [nba_api](https://github.com/swar/nba_api) (NBA Stats API)
- **Stocks:** [Yahoo Finance](https://finance.yahoo.com/) via yfinance
- **Economic data (optional):** [FRED](https://fred.stlouisfed.org/) (API key required)
- **Consumer spending:** BLS Consumer Expenditure Survey–style summary

---

## Author

Built by **Alex Smith** — [alexsmith.finance](https://alexsmith.finance) · [LinkedIn](https://www.linkedin.com/in/alexwesleysmith/)
