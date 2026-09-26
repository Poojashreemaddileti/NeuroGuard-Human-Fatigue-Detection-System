import os
from datetime import datetime


def _get_setting(key, default):
    """Read deployment secrets first, then environment variables, then safe defaults.

    Normal website users never configure these values. The administrator configures
    them once in the hosting platform's secret/environment settings.
    """
    env_key = f"NEUROGUARD_DB_{key.upper()}"
    value = os.getenv(env_key)
    if value not in (None, ""):
        return value

    # Streamlit Cloud and similar hosts can expose secrets through st.secrets.
    try:
        import streamlit as st
        if "mysql" in st.secrets and key.lower() in st.secrets["mysql"]:
            value = st.secrets["mysql"][key.lower()]
            if value not in (None, ""):
                return value
    except Exception:
        pass
    return default


DEFAULT_CONFIG = {
    "host": _get_setting("host", "localhost"),
    "port": int(_get_setting("port", "3306")),
    "user": _get_setting("user", "root"),
    "password": _get_setting("password", ""),
    "database": _get_setting("database", "neuroguard"),
}

CONFIG = dict(DEFAULT_CONFIG)


def _mysql():
    try:
        import mysql.connector
        return mysql.connector
    except ImportError as exc:
        raise RuntimeError("mysql-connector-python is not installed on the server.") from exc


def connect():
    if not CONFIG.get("password"):
        raise RuntimeError("The NeuroGuard server database password has not been configured.")
    return _mysql().connect(**CONFIG, connection_timeout=8)


def init_database():
    """Initialize the session-history table in the administrator's database.

    The database itself is expected to be provisioned by the administrator/hosting
    provider. End users never see or edit these connection details.
    """
    db = connect()
    cur = db.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS fatigue_sessions(
            id INT AUTO_INCREMENT PRIMARY KEY,
            session_time DATETIME NOT NULL,
            source_file VARCHAR(255),
            fatigue_level VARCHAR(40) NOT NULL,
            confidence DECIMAL(7,3) NOT NULL,
            fatigue_score INT NOT NULL,
            risk_score INT NOT NULL,
            drowsiness_score DECIMAL(7,3) NOT NULL,
            alertness_score DECIMAL(7,3) NOT NULL,
            delta_power DECIMAL(14,10),
            theta_power DECIMAL(14,10),
            alpha_power DECIMAL(14,10),
            beta_power DECIMAL(14,10),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    db.commit()
    cur.close()
    db.close()
    return True


def save_prediction(result, source_file=""):
    db = connect()
    cur = db.cursor()
    f = result["features"]
    cur.execute("""
        INSERT INTO fatigue_sessions
        (session_time,source_file,fatigue_level,confidence,fatigue_score,risk_score,
         drowsiness_score,alertness_score,delta_power,theta_power,alpha_power,beta_power)
        VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        datetime.now(), source_file, result["fatigue"], result["confidence"],
        result["fatigue_score"], result["risk_score"], result["drowsiness"],
        result["alertness"], float(f[0]), float(f[1]), float(f[2]), float(f[3])
    ))
    db.commit()
    cur.close()
    db.close()
    return True


def fetch_history(limit=50):
    db = connect()
    cur = db.cursor(dictionary=True)
    cur.execute("SELECT * FROM fatigue_sessions ORDER BY id DESC LIMIT %s", (int(limit),))
    rows = cur.fetchall()
    cur.close()
    db.close()
    return rows
