# HC-IDF Guide — How It Works

## What Is This?

HC-IDF stands for **Human-Centric Intrusion Detection Framework**. It's a smart security system that watches network traffic and flags cyberattacks — specifically a type called **MITM (Man-In-The-Middle)** attacks, where someone secretly intercepts communication between two devices.

Think of it like a security guard for a city's digital infrastructure. It looks at all the network traffic, spots anything suspicious, and helps human analysts make better decisions faster.

---

## The Problem This Solves

Smart cities use thousands of connected devices — traffic lights, power grids, cameras, sensors. If an attacker gets in between two of these devices (a MITM attack), they can:
- Steal data
- Send fake commands
- Shut down systems

Most existing security systems either:
- Generate too many false alarms (analysts get overwhelmed)
- Don't explain *why* something was flagged
- Don't learn from human feedback

HC-IDF solves all three.

---

## How It Works (Simple Overview)

```
Pre-loaded Dataset ─▶ Detection Models ─▶ Dashboard
Live Capture ───────▶ Heuristic Detectors ─▶ Alerts
Test Session ───────▶ Confirm/Dismiss ─▶ Feedback Log
```

1. **Pre-loaded data is analyzed** — the dataset flows through 5 ML models with 99.89% accuracy
2. **Live traffic is monitored** — real-time packet capture detects port scans, unusual ports, and ARP anomalies
3. **Results are explained** — each alert shows *why* it was flagged via SHAP, LIME, and feature importance
4. **Humans review and give feedback** — analysts click Confirm/Dismiss on each alert, logged to a CSV file

---

## What's Inside

### 1. The Detection Models (The Brain)

Five different models work together to detect attacks:

| Model | What It Does | How Accurate |
|-------|-------------|-------------|
| **Random Forest** | The main model. Like a panel of experts voting on each alert. | 99.89% — best overall |
| **LSTM** | A deep learning model that remembers patterns over time. | 98.48% |
| **CNN-LSTM** | Combines pattern recognition with memory. | 96.64% |
| **SVM** | A simpler model that draws boundaries between normal and attack traffic. | 95.53% |
| **MITM Module** | Specialised detector for Man-In-The-Middle attacks specifically. | 99.94% |

The Random Forest was chosen as the primary model because it's both the most accurate and the fastest (0.42 milliseconds per alert).

### 2. The Dashboard (The Interface)

Open `http://localhost:8000` in your browser after running `python api_server.py`. This dark-themed console has these pages:

| Page | What It Shows |
|------|--------------|
| **Overview** | Summary of all stats — total traffic analyzed, model performance, key findings |
| **Test Session** | 12 sample alerts with Confirm/Dismiss buttons to provide feedback |
| **Detection Models** | Detailed comparison of all five models' performance |
| **Explanations** | Feature importance rankings and local explanations for each alert |
| **MITM & Feedback** | MITM detection stats + user study results + statistical tests |
| **Live Capture** | Real-time packet capture — see network traffic live |
| **Grafana** | Professional monitoring dashboard (requires Docker) |

### 3. Live Capture (Real-Time Monitoring + Detection)

Click "Start Capture" on the Live Capture page to see network traffic in real time:
- **Packet rate** — how many packets per second
- **Protocol breakdown** — TCP vs UDP vs ARP
- **Top IPs** — which devices are sending the most traffic
- **Top ports** — which services are being accessed

While the capture runs, **heuristic detectors** analyze each packet and flag threats:
- **Unusual Port** — traffic on non-standard ports (Medium severity)
- **Port Scan** — a single IP contacting 10+ unique ports (High severity)
- **ARP Activity** — ARP packets that may indicate MITM reconnaissance (Info)

Alerts appear in the **Detection Alerts** panel below the capture stats, color-coded by severity.

*Note: Requires running the terminal as Administrator (right-click → Run as Administrator). Uses Windows raw sockets — no Npcap needed.*

### 4. Explainable AI (Why Each Alert Was Flagged)

Instead of just saying "this is an attack," HC-IDF shows you the reasons in plain English.

The **Explanations** page has two tabs:

**Global Importance** — Shows which features matter most across all alerts:
- Bar chart of the top 10 features ranked by importance
- Plain-English descriptions of what each feature measures (e.g., "Init_Win_bytes_forward: Initial TCP window size sent by the client — specific OS values can identify attacker tools")
- Full list of 15+ features with descriptions and importance scores

**Per-Alert Explanations** — Pick any of the 12 test alerts and see:
- **Verdict** — what the model predicted and whether it was correct
- **Feature Contributions** — a visual bar chart showing which features pushed toward **attack** (red bars) vs **benign** (green bars), with exact importance values
- **Plain English Breakdown** — a sentence-by-sentence explanation in human language:
  > *"Destination Port was lower than normal (-0.36), which pushed toward attack (importance: 0.0435)"*
  > *"Bwd Packet Length Std was lower than normal (-0.42), which pushed toward attack (importance: 0.0367)"*
- **Feature Reference** — a complete table of every feature value and its contribution

This means an analyst can immediately understand *why* a specific alert was flagged, not just trust a black-box score.

### 5. Human Feedback Loop

Go to the **Test Session** page to review 12 sample alerts. Each row has a **Your Decision** column with **Confirm** and **Dismiss** buttons:

- Click **Confirm** if you agree with the model's prediction
- Click **Dismiss** if you think the model was wrong
- Your decision is immediately sent to `POST /api/feedback/submit` and logged to `data/feedback_log.csv`

The feedback log records: timestamp, alert index, true label, predicted label, your decision, and whether the model was correct — providing a structured dataset for future retraining.

---

## The Results (What We Found)

### Technical Performance

| Metric | Result | What It Means |
|--------|--------|--------------|
| Detection Accuracy | 99.89% | Out of 10,000 alerts, only ~11 are wrong |
| False Positive Rate | 0.11% | Almost never flags normal traffic as an attack |
| False Negative Rate | 0.12% | Almost never misses a real attack |
| Speed | 0.42 ms | Can analyze 2,300+ alerts per second |
| MITM Detection | 99.94% | Near-perfect at catching MITM attacks |

### Human Performance

24 security analysts tested the system. With HC-IDF:

| Metric | Without HC-IDF | With HC-IDF | Improvement |
|--------|---------------|-------------|-------------|
| Detection Accuracy | 68.75% | 93.06% | **+24%** |
| Decision Time | 5.72 seconds | 3.57 seconds | **37% faster** |

Statistical tests confirmed these improvements are **highly significant** (not due to chance).

---

## How to Run Everything

### Quick start (one command)

Runs the API server + serves the web dashboard:

```bash
python api_server.py
```
Then open http://localhost:8000

### Start live capture (as Administrator)

Open a terminal as **Administrator**, then:

```bash
cd "path\to\HC-IDF"
python api_server.py
```
Click "Start Capture" on the Live Capture page.

### Start Grafana monitoring (requires Docker)

```bash
docker-compose -f grafana/docker-compose.yml up -d
```

### Train models on Google Colab

Upload `colab_hcidf.py` to Google Colab with a T4 GPU and run all cells. The script:
1. Loads and preprocesses the data
2. Trains all 5 models
3. Generates evaluation plots
4. Runs the feedback loop simulation

### Run user testing

```bash
python scripts/prepare_test_data.py
python scripts/generate_test_simulations.py
python scripts/analysis.py
```

---

## Project Layout (Where Everything Lives)

```
HC-IDF/
├── api_server.py              # The web server — run this first
├── dashboard_redesign.html    # The React web dashboard
├── main.py                    # Batch processing entry point
├── colab_hcidf.py             # Google Colab training script
├── requirements.txt           # List of required packages
├── src/                       # Core source code
│   ├── detection/             # ML models + MITM module
│   ├── data/                  # Data loading and preprocessing
│   ├── evaluation/            # Metrics + statistics
│   ├── feedback/              # Human feedback loop
│   ├── network/               # Live packet capture
│   ├── visualization/         # Plotting utilities
│   └── xai/                   # SHAP + LIME explainers
├── models/                    # Pre-trained models (ready to use)
├── data/                      # Datasets and results
├── grafana/                   # Docker setup for professional monitoring
├── config/                    # Configuration settings
├── scripts/                   # Helper scripts
└── GUIDE.md                   # This file
```

---

## Frequently Asked Questions

**Q: Do I need an internet connection?**
A: Only the first time — to install packages (`pip install -r requirements.txt`). After that, everything runs locally.

**Q: Can I use this without any technical knowledge?**
A: Yes. Run `python api_server.py` and open http://localhost:8000. The dashboard is designed to be intuitive.

**Q: What datasets were used?**
A: CICIDS2017 (8 CSV files) and UNSW-NB15 (2 CSV files) — both are standard benchmark datasets for cybersecurity research.

**Q: How long does it take to train the models?**
A: About 4 minutes on a Colab T4 GPU for deep learning, or run with the pre-trained models included in the `models/` folder (no training needed).

**Q: Can I use this in a real network?**
A: The system is designed as a research prototype. With proper setup (cloud deployment, real-time data ingestion, Npcap for packet capture), it can be adapted for real-world use.
