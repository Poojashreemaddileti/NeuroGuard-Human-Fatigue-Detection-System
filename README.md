# NeuroGuard — Human Fatigue Detection

This is the deployable version of the original NeuroGuard interface. The existing visual design and workflow are preserved; the backend architecture has been adjusted so database credentials are administrator/server configuration, never end-user settings.

## Features
- Original NeuroGuard futuristic/light interface with sidebar, hero, cards and static brain artwork.
- Home **Get Started** opens Live EEG.
- Every portal section has a **Back to Home** action.
- CSV and EDF EEG upload.
- Clear instruction to upload the EEG file in **Live EEG** before processing.
- MNE-Python EDF processing.
- Delta, Theta, Alpha and Beta band-power feature extraction.
- Random Forest fatigue classification.
- Color-coded Fatigue, Risk and Confidence cards.
- Fatigue Gauge: green = normal, orange = average, red = danger.
- Drowsiness Analysis: separate colors for alertness, drowsiness and risk.
- Live EEG Plotly chart with colored axes/ticks/grid.
- Shared MySQL session history for the deployed application.
- User Settings contains only **Theme (Light/Dark)** and **Ambient sound**.
- Included subtle futuristic EEG-inspired ambient WAV audio.
- No MySQL credentials are exposed in the user interface.

## Production database architecture
The website users do **not** install MySQL and do not enter database credentials.

The administrator provisions one production MySQL database and configures the connection once on the hosting server using either environment variables or Streamlit secrets.

### Option A — hosting environment variables
```text
NEUROGUARD_DB_HOST=<shared-mysql-host>
NEUROGUARD_DB_PORT=3306
NEUROGUARD_DB_USER=<mysql-user>
NEUROGUARD_DB_PASSWORD=<mysql-password>
NEUROGUARD_DB_NAME=neuroguard
```

### Option B — Streamlit secrets
Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and replace the placeholders:

```toml
[mysql]
host = "YOUR_MYSQL_HOST"
port = 3306
user = "YOUR_MYSQL_USER"
password = "YOUR_MYSQL_PASSWORD"
database = "neuroguard"
```

**Never commit `.streamlit/secrets.toml` to GitHub.** It is excluded by `.gitignore`.

The app automatically creates the `fatigue_sessions` table if the configured database user has table-creation permission. For a managed database where table creation is restricted, create the table once using `database/database_schema.sql`.

## End-user workflow
1. Home → **Get Started**.
2. **Live EEG** → upload CSV or EDF.
3. **Prediction** → run Random Forest.
4. **Fatigue Gauge** / **Drowsiness Analysis** → view results.
5. **History** → view sessions saved in the central MySQL database.

The user never sees MySQL host, port, username, password, database name, or database initialization controls.

## Run on a server
```cmd
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

The included `run_neuroguard.bat` uses the same server binding.

## GitHub deployment
Push the project to GitHub, but do not push `.streamlit/secrets.toml`. On the hosting platform connected to GitHub, add the production MySQL values as private secrets/environment variables. The public application then uses that central database for History.

## EEG CSV format
A CSV should contain one or more numeric columns representing EEG channels. The default CSV sampling frequency is 160 Hz.

## Trained model
If `models/fatigue_model.joblib` exists, it is loaded automatically. Otherwise, a deterministic project-level Random Forest is generated so the interface can run immediately. For meaningful scientific performance, train and evaluate with labeled EEG data using `models/train_model.py`.

Expected training file: `data/training_features.csv`

Columns:
`delta,theta,alpha,beta,label`

Labels:
`0 = normal`, `1 = average`, `2 = danger`

## Important
This is a project prototype and is not a medical or clinical diagnostic system. The default demonstration model must not be presented as clinically validated accuracy.
