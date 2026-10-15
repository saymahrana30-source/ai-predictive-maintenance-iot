# GitHub Guide

Everything you need to publish this project as proof of work.

---

## 1. Repository details

- **Repository name:** `ai-predictive-maintenance-iot`
- **Description:** AI-powered predictive maintenance for IoT machines: simulated sensor data, Random Forest failure prediction, alert levels, Flask API and tests.
- **Topics (tags):** `machine-learning`, `predictive-maintenance`, `iot`, `time-series`, `scikit-learn`, `random-forest`, `flask`, `python`, `data-science`, `portfolio`

---

## 2. Create and push the repository (step by step)

1. **Create a GitHub account** at github.com (if you don't have one) and verify your email.
2. **Install Git**
   - Windows: download from git-scm.com
   - Mac: run `xcode-select --install` in Terminal
   - Linux: run `sudo apt install git`
3. **Set your name and email once** (use the email on your GitHub account):
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "you@example.com"
   ```
4. **Go to the project folder:**
   ```bash
   cd AI-Predictive-Maintenance-IoT
   ```
5. **Start Git and name the main branch:**
   ```bash
   git init
   git branch -M main
   ```
6. **Check what will be uploaded.** The `.gitignore` file should keep `.venv/`, `models/*.joblib` and cache files out:
   ```bash
   git status
   ```
7. **Make your first commit:**
   ```bash
   git add .
   git commit -m "chore: project setup, folder structure and requirements"
   ```
8. **Create an empty repository on GitHub:** click **New**, enter the repository name, keep it **Public**, and do **not** tick "Add a README" (you already have one). Click **Create repository**.
9. **Connect your folder to GitHub** (use the address GitHub shows you):
   ```bash
   git remote add origin https://github.com/<your-username>/ai-predictive-maintenance-iot.git
   ```
10. **Upload the code:**
    ```bash
    git push -u origin main
    ```
    When asked for a password, use a **Personal Access Token** (GitHub Settings → Developer settings → Personal access tokens), not your account password. Alternatively, use GitHub Desktop.
11. **Finish the repository page:** click the gear icon next to **About**, paste the description and topics, and tick **Website** if you have a demo link.

---

## 3. Commit plan

Commit in logical steps. Each commit should match a real stage of the work, with a clear message.

| # | Commit message | Files |
|---|---|---|
| 1 | `chore: project setup, folder structure and requirements` | `README.md`, `requirements.txt`, `.gitignore`, folders |
| 2 | `feat: simulate hourly IoT sensor data with wear-out failures` | `src/config.py`, `src/simulate_data.py`, `data/raw/` |
| 3 | `feat: clean sensor data (impossible values and gaps)` | `src/preprocess.py` |
| 4 | `feat: rolling features and failure-within-24h labels` | `src/features.py` |
| 5 | `feat: train logistic regression and random forest with time-based split` | `src/train.py` |
| 6 | `feat: evaluation metrics and alert levels` | `src/evaluate.py`, `src/alerts.py` |
| 7 | `feat: prediction pipeline and visualization charts` | `src/predict.py`, `src/visualize.py`, `main.py`, `screenshots/` |
| 8 | `feat: flask api for real-time failure predictions` | `src/api.py`, `docs/sample_request.json` |
| 9 | `test: unit tests for labels, features, cleaning and splits` | `tests/test_pipeline.py` |
| 10 | `docs: full README with results, charts and usage` | `README.md`, `docs/` |

**To commit in stages**, add only the files for one step at a time:
```bash
git add src/config.py src/simulate_data.py data/raw
git commit -m "feat: simulate hourly IoT sensor data with wear-out failures"
git push
```

**Important: be honest with your history.** Do not backdate commits or make up a timeline. Read each file before you commit it, because interviewers often ask you to explain any part of the code, such as "why 72 hours?", "why is the split by time?" or "what happens if a sensor sends a bad value?" If you used an AI tool to help you write code, make sure you understand it well enough to explain it.

---

## 4. Uploading visuals

1. Keep your screenshots and charts in the `screenshots/` folder. Use lowercase names with underscores, no spaces, and names that describe the content, such as `sensor_history.png`.
2. Commit them: `git add screenshots/ && git commit -m "docs: add result charts"`
3. Show them in the README with a caption:
   ```markdown
   ![Failure probability](screenshots/failure_probability.png)
   ```
4. Use the GitHub website's **Add file → Upload files** button if you prefer not to use the command line.

---

## 5. Making the repository professional

- [ ] README has an overview, problem statement, architecture, results, screenshots and usage steps
- [ ] `python main.py` runs on a clean machine
- [ ] `python -m pytest -q` passes
- [ ] The API responds to `docs/sample_request.json`
- [ ] No passwords, tokens or personal data in any file
- [ ] `.gitignore` excludes `.venv/`, model files and cache folders
- [ ] Commit messages are clear and follow the commit plan
- [ ] Repository is public, has a description and topics
- [ ] Repository is pinned to your GitHub profile (profile page → **Customize your pins**)
- [ ] Optional: add an MIT license file so others know they can reuse the code

---

## 6. Seven-day proof plan

| Day | Focus | What you do | What to capture | Commits |
|---|---|---|---|---|
| 1 | Setup | Create the virtual environment, install requirements, check imports | Terminal showing the successful install | 1 |
| 2 | Dataset | Run the simulation, open the CSV, review the sensor chart | Dataset preview and `sensor_history.png` | 2 |
| 3 | Preprocessing and features | Run the cleaning step, read the feature list | Output showing values repaired | 3, 4 |
| 4 | Model | Run training, compare the two models | Metrics table | 5 |
| 5 | Evaluation | Read the confusion matrix and feature importance, then write what they mean | `confusion_matrix.png` and `feature_importance.png` | 6 |
| 6 | Visualization and API | Check the charts, start the API, send a request | Successful `curl` response and `failure_probability.png` | 7, 8 |
| 7 | Upload and polish | Write tests, finish the README, pin the repository, write a LinkedIn post | Repository page and passing tests | 9, 10 |

---

## 7. Proof checklist

The `screenshots/` folder holds the images for each proof item. The terminal and code images were rendered from the project's real output and source files. Replace them with screenshots from your own machine when you run the project yourself, because your own captures are stronger proof than anything generated.

| What to capture | File |
|---|---|
| Dataset preview (first rows of the CSV) | `screenshots/dataset_preview.png` |
| Terminal output of `python main.py` | `screenshots/pipeline_output.png` |
| Model metrics table | `screenshots/metrics_table.png` |
| Successful API response | `screenshots/api_response.png` |
| Failure probability chart | `screenshots/failure_probability.png` |
| Sensor history chart | `screenshots/sensor_history.png` |
| Confusion matrix | `screenshots/confusion_matrix.png` |
| Feature importance chart | `screenshots/feature_importance.png` |
| Tests passing (`python -m pytest -q`) | `screenshots/tests_passing.png` |
| Key code: failure label | `screenshots/code_features.png` |
| Key code: prediction endpoint | `screenshots/code_api.png` |
| Repository page on GitHub | `screenshots/repo_preview.png` (take this in your browser; it cannot be generated from the project) |

**Tips for good screenshots:** crop to the relevant output, keep the window title visible, use a readable font size, and avoid showing your username, file paths with personal folder names, or any tokens.

---

## 8. Sharing it

A short LinkedIn post works well. Use this structure and fill in your own details:

> I built an AI system that predicts machine failures 24 hours in advance from IoT sensor data. On unseen data, the Random Forest flags 71% of the hours before failures, with 81% precision, and it serves predictions through a Flask API. Everything is on GitHub: [link]. Feedback welcome!

Use only numbers you have produced yourself by running the code.
