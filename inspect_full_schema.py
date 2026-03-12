from src.database.connection import DatabaseManager
from sqlalchemy import text
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

db = DatabaseManager()
with db.get_connection() as conn:
    print("\n--- Complete Database Schema ---")
    try:
        # Get all tables
        tables_query = text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        tables = pd.read_sql(tables_query, conn)['table_name'].tolist()
        
        for table in tables:
            print(f"\nTable: {table}")
            cols_query = text(f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = '{table}'
            """)
            cols = pd.read_sql(cols_query, conn)
            print(cols)
    except Exception as e:
        print(f"Error getting schema: {e}")
