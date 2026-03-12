from src.database.connection import DatabaseManager
import pandas as pd
from datetime import datetime
from sqlalchemy import text
import logging

logging.basicConfig(level=logging.INFO)

db = DatabaseManager()
with db.get_connection() as conn:
    print("--- Checking AAPL stock ID ---")
    stock_res = pd.read_sql(text("SELECT id, symbol FROM stocks WHERE symbol = 'AAPL'"), conn)
    print(stock_res)
    
    if not stock_res.empty:
        stock_id = stock_res.iloc[0]['id']
        print(f"\n--- Checking prices for stock_id: {stock_id} ---")
        price_query = text("SELECT count(*) FROM stock_prices WHERE stock_id = :stock_id")
        price_count = pd.read_sql(price_query, conn, params={'stock_id': stock_id})
        print(price_count)
        
        print("\n--- Checking which stocks have prices ---")
        price_counts = pd.read_sql(text("""
            SELECT s.symbol, count(*) as count
            FROM stock_prices sp
            JOIN stocks s ON sp.stock_id = s.id
            GROUP BY s.symbol
            ORDER BY count DESC
        """), conn)
        print(price_counts)
    else:
        print("AAPL not found in stocks table")
