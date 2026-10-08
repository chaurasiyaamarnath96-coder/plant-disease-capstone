import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = r"C:\\Projects\\plant-disease-capstone\\database\\plant_databse.db"


def create_table():
    """
    Create predictions table if it doesn't exist.
    """

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        timestamp TEXT NOT NULL,

        disease TEXT NOT NULL,

        confidence REAL NOT NULL,

        recommendation TEXT
    )
    """)

    conn.commit()
    conn.close()

    print("✅ Table created successfully")


def save_prediction(
    disease,
    confidence,
    recommendation
):
    """
    Save prediction to database.
    """

    
    
    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )
        
    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()
        
    cursor.execute("""
    INSERT INTO predictions
    (
        timestamp,
        disease,
        confidence,
        recommendation
    )

    VALUES
    (
        ?, ?, ?, ?
    )
    """, (
        timestamp,
        disease,
        confidence,
        recommendation

        ))

    conn.commit()
    conn.close()

    print("✅ Prediction saved")


def get_predictions():
    """
    Return all predictions as DataFrame.
    """

    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql_query(
        """
        SELECT *
        FROM predictions
        ORDER BY timestamp DESC
        """,
        conn
    )

    conn.close()

    return df

