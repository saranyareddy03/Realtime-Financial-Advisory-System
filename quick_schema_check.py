from src.database.connection import DatabaseManager
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def check_schemas():
    db_manager = DatabaseManager()
    
    try:
        with db_manager.get_connection() as conn:
            print("=== TECHNICAL INDICATORS TABLE ===")
            try:
                cols = pd.read_sql("SELECT * FROM technical_indicators LIMIT 0", conn).columns.tolist()
                print(cols)
            except Exception as e:
                print(f"Error reading technical_indicators: {e}")
            
            print("\n=== RISK METRICS TABLE ===") 
            try:
                cols = pd.read_sql("SELECT * FROM risk_metrics LIMIT 0", conn).columns.tolist()
                print(cols)
            except Exception as e:
                print(f"Error reading risk_metrics: {e}")
            
            print("\n=== SENTIMENT SCORES TABLE ===")
            try:
                cols = pd.read_sql("SELECT * FROM sentiment_scores LIMIT 0", conn).columns.tolist()
                print(cols)
            except Exception as e:
                print(f"Error reading sentiment_scores: {e}")

    except Exception as e:
        print(f"Database connection error: {e}")

if __name__ == "__main__":
    check_schemas()