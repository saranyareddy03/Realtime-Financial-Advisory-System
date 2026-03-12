from src.database.connection import DatabaseManager
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

db = DatabaseManager()
with db.get_connection() as conn:
    print("\n--- Stocks Table ---")
    print(pd.read_sql('SELECT count(*) FROM stocks', conn))
    
    print("\n--- Stock Prices Table ---")
    print(pd.read_sql('SELECT min(date), max(date), count(*) FROM stock_prices', conn))
