# **PulseSTORM**

This repository hosts a Python-based Streamlit application designed for analysis and visualization of SOAC filament data and large-scale STORM localization datasets.

🔗 Repository:
[https://github.com/Alejandro1400/BlinkFusion](https://github.com/Alejandro1400/BlinkFusion)

---

## **Features**

* Filament analysis pipeline for SOAC data
* STORM molecule merging and tracking
* Interactive Streamlit dashboards
* MongoDB-backed storage for high-density STORM data
* Automatic environment setup via Python bootstrap script

---

## **Why MongoDB is Used (STORM only)**

STORM localization datasets contain **millions of detections**, making repeated file-based reads extremely slow and memory heavy.

MongoDB is used only for STORM because it provides:

* Indexed storage for fast spatial and temporal queries
* Efficient retrieval during interactive dashboard analysis
* Scalable handling of molecule merging and tracking results

SOAC filament analysis does **not** require MongoDB.

---

## **Requirements**

### 1. Install Python

Python 3.11+ recommended
[https://www.python.org/downloads/](https://www.python.org/downloads/)

### 2. Install MongoDB Community (for STORM)

Download:
[https://www.mongodb.com/try/download/community](https://www.mongodb.com/try/download/community)

Windows setup guide:
[https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-windows/](https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-windows/)

MongoDB can run locally at:

```
localhost:27017
```

---

## **Environment Variables (.env)**

To avoid exposing credentials, MongoDB configuration is loaded from a `.env` file.

Create `.env` in the project root:

```env
STORM_DB_USER=myadmin
STORM_DB_PASS=Password123!
STORM_DB_HOST=localhost
STORM_DB_PORT=27017
STORM_DB_NAME=storm_db
```

The application loads this automatically using `python-dotenv`.

`.env` is excluded from Git via `.gitignore`.

---

## **Installation (Manual)**

```bash
git clone https://github.com/Alejandro1400/BlinkFusion
cd BlinkFusion
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py
```

---

## **Installation & Run (Automatic — Recommended for Windows)**

We created a bootstrap runner that handles everything automatically:

✔ creates virtual environment
✔ installs requirements
✔ launches Streamlit

Just run:

```bat
py -3.11 run.py app.py
```

No manual setup needed.

---

## **Project Structure**

* `Analysis/` → SOAC & STORM processing
* `Data_access/` → DB + file utilities
* `UI/` → Streamlit dashboards
* `Dashboard/` → Visualization modules
* `Docs/` → Documentation (Please review for understanding on how to use the app)

---

## **Typical STORM Workflow**

1. Install MongoDB locally
2. Create `.env` credentials
3. Launch app (`py run.py app.py`)

