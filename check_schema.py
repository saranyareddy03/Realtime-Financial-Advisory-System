from src.database.connection import DatabaseManager
from sqlalchemy import text
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

db = DatabaseManager()
with db.get_connection() as conn:
    # Check technical_indicators columns
    print("\n--- Technical Indicators Schema ---")
    try:
        # PostgreSQL specific query to get columns
        query = text("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'technical_indicators'
        """)
        cols = pd.read_sql(query, conn)
        print(cols)
    except Exception as e:
        print(f"Error getting schema: {e}")
