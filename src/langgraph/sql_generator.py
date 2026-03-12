"""
Advanced SQL Query Generator
Real-Time Financial Advisory System - Phase 5 Component 2

This module implements sophisticated SQL generation using advanced LLM reasoning
strategies including chain-of-thought, schema-aware prompting, and multi-step construction.
"""

import asyncio
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy import text as sql_text

# Import our modules
from src.config.settings import config as settings


class QueryComplexity(Enum):
    """Query complexity levels for different reasoning strategies"""
    SIMPLE = "simple"      # Single table, basic filtering
    MODERATE = "moderate"  # 2-3 table joins, aggregations
    COMPLEX = "complex"    # Multi-table joins, subqueries, analytics
    ADVANCED = "advanced"  # Complex analytics, window functions, CTEs


@dataclass
class SQLQueryResult:
    """Result structure for SQL generation"""
    sql: str
    parameters: Dict[str, Any]
    reasoning: str
    complexity: QueryComplexity
    estimated_execution_time: float
    tables_involved: List[str]
    validation_errors: List[str]
    optimization_suggestions: List[str]


class DatabaseSchemaKnowledge:
    """
    Comprehensive database schema knowledge for intelligent SQL generation
    """
    
    @staticmethod
    def get_schema_context() -> str:
        """Detailed schema information for LLM reasoning"""
        return """
        DATABASE SCHEMA - FINANCIAL ADVISORY SYSTEM:
        
        1. stocks (Company Master Data):
           - id (uuid), symbol (varchar), company_name (varchar), sector (varchar), country (varchar), market_cap (bigint), is_active (boolean)
           
        2. stock_prices (Historical OHLCV Data):
           - id (uuid), stock_id (uuid), symbol (varchar), date (date), open_price (numeric), high_price (numeric), low_price (numeric), close_price (numeric), volume (bigint)
           
        3. technical_indicators (Technical Analysis):
           - id (uuid), stock_id (uuid), symbol (varchar), date (date), sma_20 (numeric), sma_50 (numeric), sma_200 (numeric), rsi_14 (numeric), macd (numeric), macd_signal (numeric), macd_histogram (numeric), bollinger_upper (numeric), bollinger_lower (numeric), volume_sma_20 (bigint)
           
        4. risk_metrics (Risk Calculations):
           - id (uuid), stock_id (uuid), symbol (varchar), calculation_date (date), volatility_30d (numeric), beta (numeric), sharpe_ratio (numeric), max_drawdown (numeric), value_at_risk_95 (numeric)
           
        5. sentiment_scores (News Sentiment):
           - id (uuid), news_id (uuid), stock_id (uuid), symbol (varchar), sentiment_score (numeric), confidence_score (numeric), sentiment_label (varchar)
           
        6. financial_news (News Data):
           - id (uuid), headline (text), content (text), publisher (varchar), published_at (timestamp), url (text)
           
        7. portfolios (User Portfolios):
           - id (uuid), user_id (uuid), name (varchar), total_value (numeric), cash_balance (numeric)
           
        8. portfolio_holdings (Portfolio Positions):
           - id (uuid), portfolio_id (uuid), stock_id (uuid), symbol (varchar), shares (numeric), avg_cost_basis (numeric), current_price (numeric), market_value (numeric)

        VIEWS:
        - v_latest_stock_prices: Latest prices for all stocks (symbol, company_name, close_price, volume, etc.)
        - v_daily_sentiment: Daily aggregated sentiment (symbol, avg_sentiment, news_count)
        - v_portfolio_summary: Summary per portfolio (portfolio_name, total_market_value, holdings_count)
        
        KEY RELATIONSHIPS:
        - stocks.id = stock_prices.stock_id
        - stocks.id = technical_indicators.stock_id
        - stocks.id = risk_metrics.stock_id
        - stocks.id = sentiment_scores.stock_id
        - financial_news.id = sentiment_scores.news_id
        - portfolios.id = portfolio_holdings.portfolio_id
        - stocks.id = portfolio_holdings.stock_id
        """
    
    @staticmethod
    def get_query_patterns() -> Dict[str, str]:
        """Common query patterns for different intents"""
        return {
            "stock_analysis": """
            SELECT s.symbol, s.company_name, sp.close_price, sp.volume,
                   ti.rsi_14, rm.volatility_30d, rm.beta
            FROM stocks s
            JOIN stock_prices sp ON s.id = sp.stock_id
            LEFT JOIN technical_indicators ti ON s.id = ti.stock_id AND ti.date = sp.date
            LEFT JOIN risk_metrics rm ON s.id = rm.stock_id
            WHERE s.symbol = :symbol AND sp.date = (SELECT MAX(date) FROM stock_prices WHERE stock_id = s.id)
            """,
            
            "sentiment_analysis": """
            SELECT s.symbol, fn.headline, ss.sentiment_score, fn.published_at
            FROM stocks s
            JOIN sentiment_scores ss ON s.id = ss.stock_id
            JOIN financial_news fn ON ss.news_id = fn.id
            WHERE s.symbol = :symbol AND fn.published_at >= CURRENT_DATE - INTERVAL '7 days'
            ORDER BY fn.published_at DESC
            """,
            
            "portfolio_review": """
            SELECT p.name as portfolio_name, ph.symbol, ph.shares, ph.avg_cost_basis,
                   ph.current_price, ph.market_value
            FROM portfolios p
            JOIN portfolio_holdings ph ON p.id = ph.portfolio_id
            WHERE p.user_id = :user_id
            """,
            
            "risk_assessment": """
            SELECT s.symbol, rm.volatility_30d, rm.beta, rm.sharpe_ratio, rm.value_at_risk_95
            FROM stocks s
            JOIN risk_metrics rm ON s.id = rm.stock_id
            WHERE s.symbol = :symbol
            ORDER BY rm.calculation_date DESC
            LIMIT 1
            """,
        }


class AdvancedSQLGenerator:
    """
    Advanced SQL Query Generator using sophisticated LLM reasoning strategies
    """
    
    def __init__(self):
        self.settings = settings
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=self.settings.GEMINI_API_KEY,
            temperature=0.1,
            max_tokens=2000
        )
        self.schema = DatabaseSchemaKnowledge()
    
    async def generate_sql(
        self, 
        intent: str, 
        entities: Dict[str, Any], 
        user_query: str
    ) -> SQLQueryResult:
        """
        Main SQL generation method using multi-strategy reasoning
        """
        try:
            # Step 1: Analyze query complexity
            complexity = await self._analyze_complexity(intent, entities, user_query)
            
            # Step 2: Choose reasoning strategy based on complexity
            if complexity == QueryComplexity.SIMPLE:
                result = await self._generate_simple_query(intent, entities)
            elif complexity == QueryComplexity.MODERATE:
                result = await self._generate_moderate_query(intent, entities)
            elif complexity == QueryComplexity.COMPLEX:
                result = await self._generate_complex_query(intent, entities, user_query)
            else:  # ADVANCED
                result = await self._generate_advanced_query(intent, entities, user_query)
            
            # Step 3: Validate and optimize
            validated_result = await self._validate_and_optimize(result)
            
            return validated_result
            
        except Exception as e:
            return SQLQueryResult(
                sql="SELECT 'Error in SQL generation' as error_message",
                parameters={},
                reasoning=f"SQL generation failed: {str(e)}",
                complexity=QueryComplexity.SIMPLE,
                estimated_execution_time=0.0,
                tables_involved=[],
                validation_errors=[str(e)],
                optimization_suggestions=[]
            )
    
    async def _analyze_complexity(
        self, 
        intent: str, 
        entities: Dict[str, Any], 
        user_query: str
    ) -> QueryComplexity:
        """
        Analyze query complexity using Chain-of-Thought reasoning
        """
        complexity_analysis_prompt = f"""
        You are a database query complexity analyzer. Use Chain-of-Thought reasoning to determine query complexity.
        
        USER QUERY: "{user_query}"
        DETECTED INTENT: {intent}
        EXTRACTED ENTITIES: {json.dumps(entities, indent=2)}
        
        COMPLEXITY ANALYSIS FRAMEWORK:
        
        SIMPLE (single table, basic filtering):
        - Basic stock price lookup
        - Single metric retrieval
        - Direct symbol-based queries
        
        MODERATE (2-3 tables, joins, aggregations):
        - Stock price + technical indicators
        - Portfolio holdings overview
        - Recent sentiment analysis
        
        COMPLEX (multiple tables, subqueries, analytics):
        - Stock comparison across metrics
        - Portfolio performance analysis
        - Multi-timeframe analysis
        - Risk-adjusted calculations
        
        ADVANCED (complex analytics, window functions, CTEs):
        - Portfolio optimization calculations
        - Multi-stock correlation analysis
        - Advanced risk modeling
        - Time-series pattern analysis
        
        REASONING STEPS:
        1. What data sources are needed?
        2. How many tables must be joined?
        3. What calculations are required?
        4. Are there time-series comparisons?
        5. Is aggregation across multiple dimensions needed?
        
        Provide your reasoning step-by-step, then conclude with ONLY the complexity level: SIMPLE, MODERATE, COMPLEX, or ADVANCED.
        """
        
        response = await self.llm.ainvoke([
            SystemMessage(content=complexity_analysis_prompt),
            HumanMessage(content=user_query)
        ])
        
        content = self._extract_content(response.content)
        
        # Extract complexity level
        if "SIMPLE" in content.upper():
            return QueryComplexity.SIMPLE
        elif "MODERATE" in content.upper():
            return QueryComplexity.MODERATE
        elif "COMPLEX" in content.upper():
            return QueryComplexity.COMPLEX
        else:
            return QueryComplexity.ADVANCED
    
    async def _generate_simple_query(
        self, 
        intent: str, 
        entities: Dict[str, Any]
    ) -> SQLQueryResult:
        """
        Generate simple queries using direct pattern matching
        """
        symbols = entities.get("stocks", [])
        if not symbols:
            symbols = ["AAPL"]  # Default fallback
        
        patterns = self.schema.get_query_patterns()
        
        if intent in patterns:
            sql = patterns[intent].replace(":symbol", f"'{symbols[0]}'")
            
            return SQLQueryResult(
                sql=sql,
                parameters={"symbol": symbols[0]},
                reasoning=f"Used predefined pattern for {intent} with symbol {symbols[0]}",
                complexity=QueryComplexity.SIMPLE,
                estimated_execution_time=0.1,
                tables_involved=self._extract_tables_from_sql(sql),
                validation_errors=[],
                optimization_suggestions=[]
            )
        
        # Fallback to basic stock price query using view for better performance
        sql = f"""
        SELECT symbol, company_name, close_price, volume, date
        FROM v_latest_stock_prices
        WHERE symbol = '{symbols[0]}'
        """
        
        return SQLQueryResult(
            sql=sql,
            parameters={"symbol": symbols[0]},
            reasoning=f"Generated basic stock price query for {symbols[0]} using latest prices view",
            complexity=QueryComplexity.SIMPLE,
            estimated_execution_time=0.05,
            tables_involved=["v_latest_stock_prices"],
            validation_errors=[],
            optimization_suggestions=[]
        )
    
    async def _generate_moderate_query(
        self, 
        intent: str, 
        entities: Dict[str, Any]
    ) -> SQLQueryResult:
        """
        Generate moderate complexity queries using few-shot prompting
        """
        few_shot_examples = """
        EXAMPLE 1:
        User Query: "What is the RSI and MACD for Apple stock?"
        Intent: technical_analysis
        Entities: {"stocks": ["AAPL"], "metrics": ["rsi", "macd"]}
        
        Generated SQL:
        SELECT s.symbol, s.company_name, ti.date, ti.rsi_14, ti.macd, ti.macd_signal
        FROM stocks s
        JOIN technical_indicators ti ON s.id = ti.stock_id
        WHERE s.symbol = 'AAPL'
        ORDER BY ti.date DESC
        LIMIT 10;
        
        EXAMPLE 2:
        User Query: "Show me the recent sentiment for Tesla"
        Intent: sentiment_analysis
        Entities: {"stocks": ["TSLA"], "time_period": "1w"}
        
        Generated SQL:
        SELECT s.symbol, fn.headline, ss.sentiment_score, fn.published_at
        FROM stocks s
        JOIN sentiment_scores ss ON s.id = ss.stock_id
        JOIN financial_news fn ON ss.news_id = fn.id
        WHERE s.symbol = 'TSLA' 
          AND fn.published_at >= CURRENT_DATE - INTERVAL '7 days'
        ORDER BY fn.published_at DESC
        LIMIT 20;
        """
        
        moderate_generation_prompt = f"""
        You are an expert SQL generator for financial queries. Use the examples below to generate similar queries.
        
        {few_shot_examples}
        
        DATABASE SCHEMA:
        {self.schema.get_schema_context()}
        
        NOW GENERATE SQL FOR:
        Intent: {intent}
        Entities: {json.dumps(entities, indent=2)}
        
        REQUIREMENTS:
        1. Use proper JOINs between related tables
        2. Include meaningful columns for the intent
        3. Add appropriate WHERE clauses for filtering
        4. Include ORDER BY for time-series data
        5. Add LIMIT for performance
        6. Use proper table aliases
        
        Generate ONLY the SQL query, no additional text.
        """
        
        response = await self.llm.ainvoke([
            SystemMessage(content=moderate_generation_prompt),
            HumanMessage(content=f"Generate SQL for: {intent} with entities: {json.dumps(entities)}")
        ])
        
        sql = self._extract_sql_from_response(str(response.content))
        
        return SQLQueryResult(
            sql=sql,
            parameters=entities,
            reasoning="Generated using few-shot prompting with examples",
            complexity=QueryComplexity.MODERATE,
            estimated_execution_time=0.3,
            tables_involved=self._extract_tables_from_sql(sql),
            validation_errors=[],
            optimization_suggestions=[]
        )
    
    async def _generate_complex_query(
        self, 
        intent: str, 
        entities: Dict[str, Any], 
        user_query: str
    ) -> SQLQueryResult:
        """
        Generate complex queries using step-by-step construction reasoning
        """
        step_by_step_prompt = f"""
        You are an expert financial database analyst. Generate a complex SQL query using step-by-step reasoning.
        
        USER QUERY: "{user_query}"
        INTENT: {intent}
        ENTITIES: {json.dumps(entities, indent=2)}
        
        DATABASE SCHEMA:
        {self.schema.get_schema_context()}
        
        STEP-BY-STEP QUERY CONSTRUCTION:
        
        STEP 1 - IDENTIFY DATA REQUIREMENTS:
        What specific data points are needed to answer the query?
        Which tables contain this data?
        
        STEP 2 - PLAN JOIN STRATEGY:
        What is the logical join sequence?
        Which columns are the join keys?
        Are LEFT/INNER joins appropriate?
        
        STEP 3 - DETERMINE FILTERING:
        What WHERE conditions are needed?
        Are date ranges required?
        Should we filter for active stocks only?
        
        STEP 4 - PLAN CALCULATIONS:
        What derived calculations are needed?
        Should we use window functions?
        Are aggregations required?
        
        STEP 5 - OPTIMIZE PERFORMANCE:
        What indexes can be leveraged?
        Should we use CTEs for readability?
        Is result limiting needed?
        
        Work through each step, then provide the final optimized SQL query.
        """
        
        response = await self.llm.ainvoke([
            SystemMessage(content=step_by_step_prompt),
            HumanMessage(content=user_query)
        ])
        
        content = self._extract_content(response.content)
        sql = self._extract_sql_from_response(content)
        
        return SQLQueryResult(
            sql=sql,
            parameters=entities,
            reasoning="Generated using step-by-step construction reasoning",
            complexity=QueryComplexity.COMPLEX,
            estimated_execution_time=0.8,
            tables_involved=self._extract_tables_from_sql(sql),
            validation_errors=[],
            optimization_suggestions=[]
        )
    
    async def _generate_advanced_query(
        self, 
        intent: str, 
        entities: Dict[str, Any], 
        user_query: str
    ) -> SQLQueryResult:
        """
        Generate advanced queries using multi-agent reasoning
        """
        multi_agent_prompt = f"""
        You are a team of database experts working together. Each expert contributes to building an advanced SQL query.
        
        USER QUERY: "{user_query}"
        INTENT: {intent}
        ENTITIES: {json.dumps(entities, indent=2)}
        
        SCHEMA: {self.schema.get_schema_context()}
        
        EXPERT TEAM COLLABORATION:
        
        📊 DATA ARCHITECT: "I need to design the optimal table join strategy..."
        🔍 PERFORMANCE EXPERT: "I need to ensure query optimization..."
        📈 FINANCIAL ANALYST: "I need to include relevant financial calculations..."
        🛡️ SECURITY EXPERT: "I need to validate data access patterns..."
        
        DATA ARCHITECT REASONING:
        - Which tables form the core of this query?
        - What is the most efficient join sequence?
        - Should we use CTEs for complex logic?
        
        PERFORMANCE EXPERT REASONING:
        - Which indexes will be utilized?
        - Are there opportunities for query optimization?
        - Should we use window functions vs subqueries?
        
        FINANCIAL ANALYST REASONING:
        - What financial metrics are most relevant?
        - Are time-series calculations needed?
        - Should we include risk-adjusted measures?
        
        SECURITY EXPERT REASONING:
        - Are we accessing only necessary data?
        - Should we add data validation checks?
        - Are there potential injection vulnerabilities?
        
        COLLABORATIVE SYNTHESIS:
        Now synthesize all expert insights into a single, highly optimized SQL query.
        Include comments explaining complex parts.
        Use best practices for readability and performance.
        
        Provide the final collaborative SQL query:
        """
        
        response = await self.llm.ainvoke([
            SystemMessage(content=multi_agent_prompt),
            HumanMessage(content=user_query)
        ])
        
        content = self._extract_content(response.content)
        sql = self._extract_sql_from_response(content)
        
        return SQLQueryResult(
            sql=sql,
            parameters=entities,
            reasoning="Generated using multi-agent collaborative reasoning",
            complexity=QueryComplexity.ADVANCED,
            estimated_execution_time=1.5,
            tables_involved=self._extract_tables_from_sql(sql),
            validation_errors=[],
            optimization_suggestions=[]
        )
    
    async def _validate_and_optimize(self, result: SQLQueryResult) -> SQLQueryResult:
        """
        Validate SQL syntax and suggest optimizations
        """
        validation_prompt = f"""
        You are a SQL validation and optimization expert. Review this query for correctness and efficiency.
        
        SQL QUERY:
        {result.sql}
        
        VALIDATION CHECKLIST:
        1. Syntax correctness
        2. Table and column name accuracy
        3. Join logic validation
        4. WHERE clause efficiency
        5. Missing indexes recommendations
        6. Performance optimization opportunities
        
        Provide:
        1. List of validation errors (if any)
        2. List of optimization suggestions
        3. Estimated execution time category (fast/medium/slow)
        
        Format as JSON:
        {{
            "validation_errors": ["error1", "error2"],
            "optimization_suggestions": ["suggestion1", "suggestion2"],
            "execution_category": "fast|medium|slow"
        }}
        """
        
        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=validation_prompt),
                HumanMessage(content="Please validate this SQL query")
            ])
            
            content = self._extract_content(response.content)
            validation_data = json.loads(content)
            
            result.validation_errors = validation_data.get("validation_errors", [])
            result.optimization_suggestions = validation_data.get("optimization_suggestions", [])
            
            # Update estimated execution time based on category
            category = validation_data.get("execution_category", "medium")
            if category == "fast":
                result.estimated_execution_time = min(result.estimated_execution_time, 0.2)
            elif category == "slow":
                result.estimated_execution_time = max(result.estimated_execution_time, 2.0)
                
        except Exception as e:
            result.validation_errors.append(f"Validation failed: {str(e)}")
        
        return result
    
    def _extract_content(self, response_content: Any) -> str:
        """Extract text content from LLM response using the working pattern from workflow.py"""
        content = response_content
        if isinstance(content, list):
            # If it's a list, join the text parts or take the first one
            content = " ".join([part if isinstance(part, str) else str(part) for part in content])
        
        return str(content)
    
    def _extract_sql_from_response(self, content: str) -> str:
        """Extract SQL query from LLM response and clean it"""
        sql = ""
        if isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    match = re.search(r"```sql\n(.*?)\n```", item, re.DOTALL)
                    if match:
                        sql = match.group(1).strip()
                        break
        elif isinstance(content, str):
            match = re.search(r"```sql\n(.*?)\n```", content, re.DOTALL)
            if match:
                sql = match.group(1).strip()
        
        if not sql and "SELECT" in content.upper():
            sql = content.strip()
            
        # Clean the SQL query
        if sql:
            # Remove single-line comments
            sql = re.sub(r"--.*", "", sql)
            # Remove multi-line comments
            sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
            # Normalize whitespace
            sql = re.sub(r'\s+', ' ', sql).strip()
            return sql

        return "SELECT 'No valid SQL found' as error_message"
    
    def _extract_tables_from_sql(self, sql: str) -> List[str]:
        """Extract table names from SQL query"""
        sql_upper = sql.upper()
        tables = set()
        
        # Extract FROM and JOIN clauses
        table_patterns = [
            r'FROM\s+(\w+)',
            r'JOIN\s+(\w+)',
            r'INNER\s+JOIN\s+(\w+)',
            r'LEFT\s+JOIN\s+(\w+)',
            r'RIGHT\s+JOIN\s+(\w+)'
        ]
        
        for pattern in table_patterns:
            matches = re.findall(pattern, sql_upper)
            tables.update(matches)
        
        return list(tables)


# Integration function for the main workflow
async def generate_sql_query(
    intent: str, 
    entities: Dict[str, Any], 
    user_query: str
) -> SQLQueryResult:
    """
    Main entry point for SQL generation
    """
    generator = AdvancedSQLGenerator()
    return await generator.generate_sql(intent, entities, user_query)


if __name__ == "__main__":
    # Test the SQL generator
    async def test_sql_generation():
        test_cases = [
            {
                "intent": "stock_analysis",
                "entities": {"stocks": ["AAPL"], "metrics": ["price", "volume"]},
                "user_query": "What's the current price and volume of Apple stock?"
            },
            {
                "intent": "sentiment_analysis",
                "entities": {"stocks": ["TSLA"], "time_period": "1w"},
                "user_query": "Show me recent news sentiment for Tesla"
            },
            {
                "intent": "technical_analysis",
                "entities": {"stocks": ["MSFT"], "metrics": ["rsi", "macd"]},
                "user_query": "What are the RSI and MACD indicators for Microsoft?"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{'='*60}")
            print(f"TEST CASE {i}: {test_case['user_query']}")
            print(f"{'='*60}")
            
            result = await generate_sql_query(
                test_case["intent"],
                test_case["entities"],
                test_case["user_query"]
            )
            
            print(f"COMPLEXITY: {result.complexity.value}")
            print(f"REASONING: {result.reasoning}")
            print(f"TABLES: {result.tables_involved}")
            print(f"ESTIMATED TIME: {result.estimated_execution_time}s")
            print(f"VALIDATION ERRORS: {result.validation_errors}")
            print(f"OPTIMIZATIONS: {result.optimization_suggestions}")
            print(f"\nGENERATED SQL:")
            print("-" * 40)
            print(result.sql)
            print("-" * 40)
    
    # Run tests
    asyncio.run(test_sql_generation())
