import asyncio
import json
from typing import Dict, Any
import re
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.config.settings import config as settings

class IntentEntityExtractor:
    """
    Extracts intent and entities from a user query using an LLM.
    """
    
    def __init__(self):
        self.settings = settings
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=self.settings.GEMINI_API_KEY,
            temperature=0.1,
            max_tokens=2000
        )

    async def extract_intent_and_entities(self, user_query: str) -> Dict[str, Any]:
        """
        Extracts the intent and entities from a user query.
        """
        extraction_prompt = f"""
        You are an expert in financial query analysis. Your task is to extract the intent and relevant entities from a user's query.

        USER QUERY: "{user_query}"

        Instructions:
        1.  **Identify the user's intent**. This should be one of the following:
            *   `stock_analysis`: For queries about stock prices, volume, and technical indicators.
            *   `risk_assessment`: For queries about risk metrics like volatility and beta.
            *   `sentiment_analysis`: For queries about news sentiment.
            *   `comparative_analysis`: For queries that compare two or more stocks.

        2.  **Extract the entities**. These are the key pieces of information in the query.
            *   `stocks`: A list of stock symbols (e.g., ["AAPL", "TSLA"]).
            *   `metrics`: A list of financial metrics (e.g., ["price", "volume", "volatility"]).
            *   `time_period`: The time period of interest (e.g., "1w", "1m", "1y").

        Provide your response in JSON format. For example:
        {{
            "intent": "stock_analysis",
            "entities": {{
                "stocks": ["AAPL"],
                "metrics": ["price", "volume"]
            }}
        }}
        """

        response = await self.llm.ainvoke([
            SystemMessage(content=extraction_prompt),
            HumanMessage(content=user_query)
        ])
        
        try:
            # The response content may be a string that needs to be parsed into a dict
            content = response.content
            if isinstance(content, str):
                # Find the JSON part of the string
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    content = json_match.group(0)
                return json.loads(content)
            elif isinstance(content, dict):
                return content
            else:
                return {}
        except (json.JSONDecodeError, AttributeError):
            return {}

async def main():
    """
    Main function for testing the intent and entity extractor.
    """
    extractor = IntentEntityExtractor()
    test_queries = [
        "What's the current price and volume of Apple stock?",
        "What are the latest risk metrics for Tesla?",
        "Show me recent news sentiment for Microsoft.",
        "Compare the trading volume of Google and Amazon over the last month.",
        "What are the RSI and MACD indicators for Netflix?"
    ]

    for query in test_queries:
        result = await extractor.extract_intent_and_entities(query)
        print(f"Query: '{query}'")
        print(f"Extraction Result: {result}\n")

if __name__ == "__main__":
    import re
    asyncio.run(main())
