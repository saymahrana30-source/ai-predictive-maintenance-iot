# Project Guide

A beginner-friendly walkthrough of the project: what it does, why each step exists, and how to check that each step works.

---

## 1. Project explanation

### In simple words
Imagine a motor in a factory. Before it breaks, it usually gets hotter and shakes more. Nobody can watch every motor every hour, but sensors can. This project teaches an AI to recognise the pattern that comes just before a breakdown, so the team can repair the machine before it stops.

### In technical words
This is **supervised binary classification on multivariate time-series data**. For every machine and every hour, the model receives summary statistics of recent sensor readings (temperature, vibration, current) and estimates the probability that a failure happens within the next 24 hours. The model is evaluated on a time-based hold-out set with imbalance-aware metrics.

### The full workflow

| Stage | What happens | Where in the code |
|---|---|---|
| Sensor data collection | Temperature, vibration and current arrive hourly (simulated here; in a real plant they come from PLCs or IoT gateways) | `src/simulate_data.py` |
| Preprocessing | Impossible values are removed and gaps are filled | `src/preprocess.py` |
| Feature engineering | 6-hour averages, variability and trends are computed per machine | `src/features.py` |
| Model training | A Logistic Regression baseline and a Random Forest are trained | `src/train.py` |
| Prediction | Each hour gets a failure probability | `src/predict.py` |
| Alert generation | Probabilities become NORMAL, WARNING or CRITICAL | `src/alerts.py` |
| Visualization | Charts show sensor trends, predictions and model behaviour | `src/visualize.py` |

### How companies use predictive maintenance
- **Predict machine failures:** spot wear days before a breakdown
- **Reduce downtime:** schedule repairs in planned stops instead of emergency shutdowns
- **Save cost:** avoid damage that a late repair would cause, and avoid servicing healthy machines too early
- **Improve efficiency:** keep production lines, turbines and aircraft systems available for longer

These benefits apply across manufacturing plants, factories, power plants, the automotive industry and aviation.

---

## 2. Tech stack options

| | Option A: easiest | Option B: intermediate (chosen) | Option C: advanced |
|---|---|---|---|
| Tools | pandas, scikit-learn, Matplotlib | Option A + Flask, joblib, pytest | Option B + TensorFlow/Keras, MQTT, Docker, cloud hosting |
| Dataset | One simulated CSV file | Simulated multi-machine time series | Public data such as NASA C-MAPSS, plus live streams |
| Models | Logistic Regression | Logistic Regression and Random Forest | LSTM, GRU, Isolation Forest |
| Difficulty | Beginner | Beginner to intermediate | Advanced |
| GPU needed | No | No | Helpful for LSTMs, not required |
| Expected output | A notebook with accuracy and a few plots | A modular project with an API, tests and charts | A deployed streaming system with a dashboard |

**Selected: Option B.** It gives easy execution (one command), strong GitHub proof (an API, tests, charts and results), and practical understanding of the full pipeline. It also follows the brief's preference for Python, pandas, NumPy, scikit-learn, Matplotlib and simple classification models. No GPU is needed.

---

## 3. Architecture

### Module responsibilities

| Module | Input | Output | Responsibility |
|---|---|---|---|
| `config.py` | nothing | settings | Every value you might want to change |
| `simulate_data.py` | settings | hourly sensor CSV | Creates virtual machines and failures |
| `preprocess.py` | raw CSV | clean table | Removes bad values, fills gaps |
| `features.py` | clean table | features and labels | Summarises recent history, defines the target |
| `train.py` | training data | saved model | Trains both models on past data only |
| `evaluate.py` | predictions | metrics | Measures accuracy, precision, recall, F1 and ROC-AUC |
| `predict.py` | model and new readings | probabilities and alerts | Makes predictions using the same features as training |
| `alerts.py` | probability | NORMAL / WARNING / CRITICAL | Turns numbers into actions |
| `visualize.py` | data and predictions | PNG charts | Makes the results visible |
| `api.py` | JSON request | JSON prediction | Serves the model over HTTP |
| `main.py` | nothing | everything above | Runs the full pipeline |

### Data flow
Raw readings → cleaned readings → features plus labels → trained models → predictions → alerts, metrics, charts and API responses.

---

## 4. Implementation phases

### Phase 1: Setup
- **What to do:** install Python 3.10 to 3.12, create a virtual environment, and install `requirements.txt`
- **Why:** an isolated environment keeps this project's libraries separate from everything else
- **Expected output:** `python -c "import pandas, sklearn, flask"` runs without errors
- **Common mistakes:** installing packages outside the virtual environment; forgetting to activate it
- **How to verify:** the terminal prompt shows `(.venv)` and the import command works

### Phase 2: Dataset loading
- **What to do:** run `python main.py` and check the dataset summary
- **Why:** you need to know what your data looks like before modelling it
- **Expected output:** 28,800 readings from 8 machines and 64 failures
- **Common mistakes:** running from the wrong folder, so `data/` is not found
- **How to verify:** open `data/raw/iot_sensor_data.csv` and check the columns

### Phase 3: Data cleaning
- **What to do:** review the cleaning step, which removes impossible values and fills gaps
- **Why:** real sensors send errors, and models fail silently on bad input
- **Expected output:** "874 missing or impossible sensor values found" followed by "Gaps left: 0"
- **Common mistakes:** dropping rows (this breaks the time order); filling gaps across different machines
- **How to verify:** `python -m pytest -q` passes `test_preprocess_repairs_bad_and_missing_values`

### Phase 4: Feature engineering
- **What to do:** review `features.py` and the list of 12 features
- **Why:** rolling averages and trends reveal wear that single readings hide
- **Expected output:** 12 features per reading, and about 5.3% of hours labelled "failure within 24h"
- **Common mistakes:** using future values in a feature (data leakage); letting one machine's history affect another
- **How to verify:** `test_rolling_features_only_use_their_own_machine` and `test_labels_never_cross_between_machines` pass

### Phase 5: Model building
- **What to do:** review the time-based split and the two models
- **Why:** a model must be tested on data from after its training period, as it would be in real life
- **Expected output:** 23,032 training rows and 5,760 test rows, with both models trained
- **Common mistakes:** shuffling time-series data randomly; forgetting `class_weight` for rare failures; saving the model without its feature order
- **How to verify:** `models/predictive_maintenance_model.joblib` exists and loads correctly

### Phase 6: Evaluation
- **What to do:** read the metrics table and the confusion matrix
- **Why:** accuracy alone is misleading when failures are rare
- **Expected output:** `outputs/metrics.json`, with the Random Forest at F1 0.76 and precision 0.81
- **Common mistakes:** judging by accuracy; evaluating on training data
- **How to verify:** you can explain why precision and recall matter more than accuracy here

### Phase 7: Failure prediction
- **What to do:** start the API and send `docs/sample_request.json`
- **Why:** a model is only useful once other systems can ask it questions
- **Expected output:** `"alert_level": "CRITICAL"` with a probability near 0.98
- **Common mistakes:** the API computes features differently from training; forgetting to start the server first
- **How to verify:** the healthy-window request returns NORMAL, and a request with too few readings returns a helpful 400 error

### Phase 8: Visualization
- **What to do:** open the four PNG files in `screenshots/`
- **Why:** charts make the results understandable to non-technical people, including interviewers
- **Expected output:** sensor history, failure probability, confusion matrix and feature importance charts
- **Common mistakes:** unlabelled axes; charts that are blank or cut off
- **How to verify:** each chart has a title, labelled axes and a legend

### Phase 9: GitHub publishing
- **What to do:** follow `docs/GITHUB_GUIDE.md`
- **Why:** a public repository is your proof of work
- **Expected output:** a public repo with a clear README
- **Common mistakes:** committing `.venv/` or large model files; having no README; never testing the clone
- **How to verify:** clone the repo into a new folder and run `python main.py` there

### Phase 10: Final output
- **What to do:** run the full pipeline, the tests and the API, then capture screenshots
- **Why:** interviewers want to see results they can trust
- **Expected output:** the checklist in `docs/GITHUB_GUIDE.md` is complete
- **Common mistakes:** quoting numbers you did not get from running the code; not understanding the code you are showing
- **How to verify:** you can explain each file in one or two sentences without looking at it

---

## 5. Virtual simulation: step by step

**Step 1: How the simulated data resembles real IoT data.** Every machine reports three sensors every hour. Each sensor has a normal baseline that differs slightly from machine to machine, a day and night cycle, and random noise. Real sensors also have occasional glitches, dropouts and short spikes, and the simulation includes those.

**Step 2: How failure is represented.** Each machine runs for 12 to 25 days, then fails. The hour of failure is recorded as `failure_event = 1`. For gradual failures (70% of cases), the sensors drift upward during the 72 hours before the failure, with the change accelerating near the end. Sudden failures (30%) show no drift.

**Step 3: How the model predicts breakdown.** For every hour, the features summarise the previous 6 hours. The label is 1 if a failure happens in the next 24 hours. The Random Forest learns which combinations of averages, variability and trends usually come before a failure.

**Step 4: Generate the data.** Run `python main.py --resimulate`, or `python -m src.simulate_data` to create the CSV alone.

**Step 5: Check the data.** Open the CSV and look at the first rows. Then open `screenshots/sensor_history.png` and check that the red failure lines follow visible drift.

**Step 6: Train and test.** Run `python main.py`. The model trains on the first 80% of the timeline and is tested on the last 20%.

### Outputs to generate
- `data/raw/iot_sensor_data.csv` (the simulated sensor data)
- `outputs/metrics.json` (model scores)
- `outputs/test_predictions.csv` (every test-hour prediction, with its alert)
- Four charts in `screenshots/`

### Graphs to show
- Sensor history (shows the wear before each gradual failure)
- Failure probability (shows the warnings rising before failures)
- Confusion matrix (shows right and wrong predictions)
- Feature importance (shows which signals matter)

### Proof students should capture
- The terminal output of `python main.py`
- The metrics table
- The four charts
- A successful API response
- Tests passing
- The GitHub repository page

---

## 6. Changes from the original plan

The original plan was a strong starting point, but I changed several things so the project runs correctly on a laptop:

- **No Raspberry Pi wiring.** You told me you have no hardware, so the sensors are simulated. The wiring and sensor-reading code from the plan is not in this repo. If you get hardware later, add it as a separate `src/sensors/` module.
- **Random Forest instead of LSTM.** The plan named an LSTM but its code used a Random Forest. The Random Forest is easier to explain and needs no GPU. An LSTM is listed as a next step.
- **Time-based split instead of a random split.** A random split would let the model learn from the future, which inflates the scores.
- **Bugs fixed from the plan's code:**
  - The ESP8266 snippet used `time.sleep` without importing `time`.
  - The `iot_sensor_data.csv` file it loads did not exist anywhere in the plan.
  - The training code used a `failure` column that the plan never created.
  - The Flask app ran with `debug=True`, which can allow remote code execution.
  - The sensor code labelled raw ADC readings (0 to 1023) as milliamps.
- **Market statistics removed.** I could not verify the market size, growth rates and company examples in the "Facts" section from here, so I left them out. If you want them in your README, check each figure against its original source first.
