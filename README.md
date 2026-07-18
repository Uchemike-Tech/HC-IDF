# HC-IDF: Human-Centric Intrusion Detection Framework

A framework for detecting MITM attacks in smart city environments, combining machine learning, deep learning, explainable AI (SHAP/LIME), and a human-in-the-loop feedback system.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the dashboard

**Option A — React + API (recommended):**
```bash
python api_server.py
```
Open http://localhost:8000

**Option B — Streamlit:**
```bash
streamlit run dashboard.py
```

### 3. Train models (Colab)

Upload `colab_hcidf.py` to Google Colab with a T4 GPU runtime and run all cells. Parts 1-4 train models, Part 5 runs the feedback loop, Part 6 generates evaluation plots.

### 4. User testing

```bash
python scripts/prepare_test_data.py   # generate test samples
python scripts/generate_test_simulations.py  # simulate 23 users
python scripts/analysis.py            # statistical analysis
```

### 5. Live capture (optional)

Requires Npcap (Windows) or libpcap (Linux).
```bash
python -m src.network.capture
```

### 6. Grafana monitoring (optional)

Requires Docker.
```bash
docker-compose -f grafana/docker-compose.yml up -d
```

## Project Structure

```
HC-IDF/
├── api_server.py           FastAPI backend (serves React frontend)
├── dashboard_redesign.html React frontend (SOC-style console)
├── dashboard.py            Streamlit dashboard
├── main.py                 Entry point for batch processing
├── colab_hcidf.py          Colab training script (all models)
├── requirements.txt        Python dependencies
├── config/
│   └── config.yaml         Framework configuration
├── src/
│   ├── data/               Data loading and preprocessing
│   ├── detection/          ML/DL models + MITM module
│   ├── evaluation/         Metrics + statistical tests
│   ├── feedback/           Human-in-the-loop retraining
│   ├── network/            Live packet capture (Scapy)
│   ├── visualization/      Plotting utilities
│   └── xai/                SHAP + LIME explainers
├── models/                 Pre-trained models + scalers
├── scripts/                Utility scripts
├── grafana/                Docker Compose + provisioning
└── data/
    ├── processed/          Combined parquet dataset
    ├── test_results/       User study results
    └── analysis_output/    Evaluation plots
```

## Datasets

- **CICIDS2017** — Download from [UNB](https://www.unb.ca/cic/datasets/ids-2017.html)
- **UNSW-NB15** — Download from [UNSW](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

Place the CSVs in `data/external/CICIDS2017/` and `data/external/UNSW-NB15/`, then run:
```bash
python -c "from src.data.preprocessor import load_and_preprocess; load_and_preprocess()"
```

## Results

| Model | F1 Score | AUC-ROC | Latency (ms) |
|-------|----------|---------|--------------|
| Random Forest | 0.9989 | 0.9999 | 0.42 |
| LSTM | 0.9848 | 0.9978 | 1.87 |
| CNN-LSTM | 0.9664 | 0.9932 | 2.15 |
| SVM (linear) | 0.9553 | 0.9897 | 8.34 |
| MITM Module | 0.9994 | — | — |

User study (n=24): HC-IDF improved accuracy from 68.8% to 93.1% (p<0.001, d=2.01) and reduced response time from 5.72s to 3.57s (p<0.001, d=3.89).
