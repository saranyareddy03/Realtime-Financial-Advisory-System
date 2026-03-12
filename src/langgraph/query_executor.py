"""
Database Query Executor
Real-Time Financial Advisory System - Phase 5 Component 3

This module implements secure and efficient SQL query execution with performance monitoring,
result caching, and comprehensive error handling for financial analytics queries.
"""

import asyncio
import time
import hashlib
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
from contextlib import asynccontextmanager
import re

from sqlalchemy import text as sql_text, exc as sqlalchemy_exc
from sqlalchemy.engine import Row
import pandas as pd

# Import our modules
from src.config.settings import config as settings
from src.database.connection import db_manager
from src.langgraph.sql_generator import SQLQueryResult, QueryComplexity
from src.langgraph.sql_generator import generate_sql_query

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Query execution status"""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    CACHED = "cached"


class SecurityLevel(Enum):
    """Security validation levels"""
    LOW = "low"         # Basic validation
    MEDIUM = "medium"   # Standard financial queries
    HIGH = "high"       # Administrative queries
    CRITICAL = "critical"  # System modifications


@dataclass
class QueryExecutionResult:
    """Comprehensive result structure for query execution"""
    status: ExecutionStatus
    data: Optional[List[Dict[str, Any]]] = None
    row_count: int = 0
    execution_time: float = 0.0
    query_hash: Optional[str] = None
    cached: bool = False
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    performance_metrics: Optional[Dict[str, Any]] = None
    column_info: Optional[List[Dict[str, str]]] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


@dataclass
class QueryCacheEntry:
    """Cache entry for query results"""
    result: "QueryExecutionResult"
    timestamp: datetime
    expiry: datetime
    access_count: int = 0
    last_access: Optional[datetime] = None


class SecurityValidator:
    """
    SQL Security Validator for preventing injection attacks and unauthorized operations
    """
    
    # Dangerous SQL patterns that should be blocked
    DANGEROUS_PATTERNS = [
        r';\s*(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|TRUNCATE)',
        r'(UNION|UNION\s+ALL).*SELECT',
        r'(--\s|/\*|\*/)',
        r'(xp_|sp_|cmd|shell)',
        r'(EXEC|EXECUTE)\s+',
        r'(DECLARE|CAST|CONVERT).*@',
        r'(WAITFOR|DELAY)',
        r'(LOAD_FILE|INTO\s+OUTFILE)',
    ]
    
    # Allowed read-only operations
    ALLOWED_OPERATIONS = [
        'SELECT', 'WITH', 'EXPLAIN', 'SHOW', 'DESCRIBE', 'ANALYZE'
    ]
    
    # Allowed tables for financial queries
    ALLOWED_TABLES = [
        'stocks', 'stock_prices', 'technical_indicators', 'risk_metrics',
        'sentiment_scores', 'financial_news', 'news_stock_mentions',
        'portfolios', 'portfolio_holdings', 'users', 'user_queries'
    ]
    
    @classmethod
    def validate_query(cls, sql: str, security_level: SecurityLevel = SecurityLevel.MEDIUM) -> Tuple[bool, List[str]]:
        """
        Validate SQL query for security and compliance
        
        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings = []
        sql_upper = sql.upper().strip()
        
        # 1. Check for dangerous patterns
        import re
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, sql_upper, re.IGNORECASE):
                return False, [f"Dangerous SQL pattern detected: {pattern}"]
        
        # 2. Ensure read-only operations only
        first_word = sql_upper.split()[0] if sql_upper.split() else ""
        if first_word not in cls.ALLOWED_OPERATIONS:
            return False, [f"Operation not allowed: {first_word}"]
        
        # 3. Check table access (extract table names)
        table_pattern = r'(?:FROM|JOIN)\s+(\w+)'
        tables_found = re.findall(table_pattern, sql_upper, re.IGNORECASE)
        
        for table in tables_found:
            if table.lower() not in cls.ALLOWED_TABLES:
                warnings.append(f"Table access warning: {table}")
        
        # 4. Security level specific checks
        if security_level == SecurityLevel.HIGH:
            # Additional checks for high security
            if len(sql) > 5000:
                warnings.append("Query length exceeds recommended limit")
            
            if sql.count('(') > 20:  # Complex nested queries
                warnings.append("High complexity query detected")
        
        return True, warnings
    
    @classmethod
    def sanitize_parameters(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize query parameters to prevent injection"""
        sanitized = {}
        
        for key, value in params.items():
            if isinstance(value, str):
                # Remove potentially dangerous characters
                sanitized[key] = re.sub(r'[;"\'\\]', '', str(value)[:100])
            elif isinstance(value, (int, float)):
                sanitized[key] = value
            elif isinstance(value, list):
                # Sanitize list elements
                sanitized[key] = [re.sub(r'[;"\'\\]', '', str(v)[:50]) for v in value[:10]]
            else:
                sanitized[key] = str(value)[:100]
        
        return sanitized


class QueryCache:
    """
    Intelligent query result caching system
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.cache: Dict[str, QueryCacheEntry] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
    
    def _generate_hash(self, sql: str, params: Dict[str, Any]) -> str:
        """Generate unique hash for SQL query and parameters"""
        content = f"{sql}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, sql: str, params: Dict[str, Any]) -> Optional[QueryExecutionResult]:
        """Get cached result if available and not expired"""
        query_hash = self._generate_hash(sql, params)
        
        if query_hash in self.cache:
            cache_entry = self.cache[query_hash]
            
            # Check if expired
            if datetime.now() > cache_entry.expiry:
                del self.cache[query_hash]
                return None
            
            # Update access statistics
            cache_entry.access_count += 1
            cache_entry.last_access = datetime.now()
            
            # Return cached result
            cached_result = cache_entry.result
            cached_result.cached = True
            cached_result.query_hash = query_hash
            
            return cached_result
        
        return None
    
    def set(self, sql: str, params: Dict[str, Any], result: QueryExecutionResult, ttl: Optional[int] = None) -> None:
        """Cache query result"""
        if result.status != ExecutionStatus.SUCCESS:
            return  # Don't cache errors
        
        query_hash = self._generate_hash(sql, params)
        ttl = ttl or self.default_ttl
        
        # Evict old entries if cache is full
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        # Cache the result
        cache_entry = QueryCacheEntry(
            result=result,
            timestamp=datetime.now(),
            expiry=datetime.now() + timedelta(seconds=ttl),
            access_count=1,
            last_access=datetime.now()
        )
        
        self.cache[query_hash] = cache_entry
        result.query_hash = query_hash
    
    def _evict_oldest(self) -> None:
        """Remove least recently used cache entries"""
        if not self.cache:
            return
        
        # Remove 20% of oldest entries
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1].last_access or x[1].timestamp
        )
        
        evict_count = max(1, len(self.cache) // 5)
        for i in range(evict_count):
            if i < len(sorted_entries):
                del self.cache[sorted_entries[i][0]]
    
    def clear_expired(self) -> int:
        """Remove expired cache entries"""
        now = datetime.now()
        expired_keys = [
            key for key, entry in self.cache.items()
            if now > entry.expiry
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        return len(expired_keys)


class PerformanceMonitor:
    """
    Monitor and analyze query performance
    """
    
    def __init__(self):
        self.query_stats = {}
    
    def record_execution(
        self, 
        sql: str, 
        execution_time: float, 
        row_count: int, 
        complexity: QueryComplexity
    ) -> Dict[str, Any]:
        """Record query execution metrics"""
        
        # Generate simplified query signature for grouping
        query_signature = self._get_query_signature(sql)
        
        if query_signature not in self.query_stats:
            self.query_stats[query_signature] = {
                'total_executions': 0,
                'total_time': 0.0,
                'avg_time': 0.0,
                'min_time': float('inf'),
                'max_time': 0.0,
                'total_rows': 0,
                'avg_rows': 0,
                'complexity': complexity.value,
                'recent_executions': []
            }
        
        stats = self.query_stats[query_signature]
        stats['total_executions'] += 1
        stats['total_time'] += execution_time
        stats['avg_time'] = stats['total_time'] / stats['total_executions']
        stats['min_time'] = min(stats['min_time'], execution_time)
        stats['max_time'] = max(stats['max_time'], execution_time)
        stats['total_rows'] += row_count
        stats['avg_rows'] = stats['total_rows'] / stats['total_executions']
        
        # Keep recent execution history (last 10)
        stats['recent_executions'].append({
            'timestamp': datetime.now().isoformat(),
            'execution_time': execution_time,
            'row_count': row_count
        })
        
        if len(stats['recent_executions']) > 10:
            stats['recent_executions'] = stats['recent_executions'][-10:]
        
        return {
            'query_signature': query_signature,
            'current_execution_time': execution_time,
            'average_execution_time': stats['avg_time'],
            'performance_rating': self._get_performance_rating(execution_time, complexity)
        }
    
    def _get_query_signature(self, sql: str) -> str:
        """Generate a simplified signature for query grouping"""
        import re
        
        # Remove specific values, keep structure
        normalized = re.sub(r"'[^']*'", "'?'", sql)
        normalized = re.sub(r'\b\d+\b', '?', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized[:200]  # Limit length
    
    def _get_performance_rating(self, execution_time: float, complexity: QueryComplexity) -> str:
        """Rate query performance based on complexity and time"""
        
        thresholds = {
            QueryComplexity.SIMPLE: {'excellent': 0.1, 'good': 0.3, 'fair': 0.8},
            QueryComplexity.MODERATE: {'excellent': 0.3, 'good': 0.8, 'fair': 2.0},
            QueryComplexity.COMPLEX: {'excellent': 0.8, 'good': 2.0, 'fair': 5.0},
            QueryComplexity.ADVANCED: {'excellent': 2.0, 'good': 5.0, 'fair': 10.0}
        }
        
        limits = thresholds.get(complexity, thresholds[QueryComplexity.SIMPLE])
        
        if execution_time <= limits['excellent']:
            return 'excellent'
        elif execution_time <= limits['good']:
            return 'good'
        elif execution_time <= limits['fair']:
            return 'fair'
        else:
            return 'slow'


class DatabaseQueryExecutor:
    """
    Main Database Query Executor with security, caching, and performance monitoring
    """
    
    def __init__(self):
        self.security_validator = SecurityValidator()
        self.cache = QueryCache(max_size=500, default_ttl=300)  # 5 min default TTL
        self.performance_monitor = PerformanceMonitor()
        self.max_execution_time = 30.0  # 30 seconds timeout
        self.max_result_size = 10000    # Max rows to return
    
    async def execute_query(
        self, 
        sql_result: SQLQueryResult, 
        security_level: SecurityLevel = SecurityLevel.MEDIUM,
        use_cache: bool = True,
        max_rows: Optional[int] = None
    ) -> QueryExecutionResult:
        """
        Main query execution method with full security and performance features
        """
        start_time = time.time()
        
        try:
            # Step 1: Security validation
            is_valid, security_warnings = self.security_validator.validate_query(
                sql_result.sql, security_level
            )
            
            if not is_valid:
                return QueryExecutionResult(
                    status=ExecutionStatus.ERROR,
                    error_message=f"Security validation failed: {security_warnings[0]}",
                    error_code="SECURITY_VIOLATION",
                    warnings=security_warnings,
                    execution_time=time.time() - start_time
                )
            
            # Step 2: Sanitize parameters
            safe_params = self.security_validator.sanitize_parameters(sql_result.parameters)
            
            # Step 3: Check cache
            if use_cache:
                cached_result = self.cache.get(sql_result.sql, safe_params)
                if cached_result:
                    logger.info(f"Cache hit for query hash: {cached_result.query_hash}")
                    return cached_result
            
            # Step 4: Execute query
            execution_result = await self._execute_with_timeout(
                sql_result.sql, 
                safe_params, 
                max_rows or self.max_result_size
            )
            
            # Step 5: Record performance metrics
            performance_metrics = self.performance_monitor.record_execution(
                sql_result.sql,
                execution_result.execution_time,
                execution_result.row_count,
                sql_result.complexity
            )
            
            execution_result.performance_metrics = performance_metrics
            execution_result.warnings.extend(security_warnings)
            
            # Step 6: Cache successful results
            if use_cache and execution_result.status == ExecutionStatus.SUCCESS:
                cache_ttl = self._calculate_cache_ttl(sql_result.complexity, execution_result.execution_time)
                self.cache.set(sql_result.sql, safe_params, execution_result, cache_ttl)
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Query execution error: {str(e)}")
            return QueryExecutionResult(
                status=ExecutionStatus.ERROR,
                error_message=str(e),
                error_code="EXECUTION_ERROR",
                execution_time=time.time() - start_time
            )
    
    async def _execute_with_timeout(
        self, 
        sql: str, 
        parameters: Dict[str, Any], 
        max_rows: int
    ) -> QueryExecutionResult:
        """Execute query with timeout protection"""
        
        start_time = time.time()
        
        try:
            # Use asyncio timeout for execution
            async with asyncio.timeout(self.max_execution_time):
                result = await self._execute_query_internal(sql, parameters, max_rows)
                
            result.execution_time = time.time() - start_time
            return result
            
        except asyncio.TimeoutError:
            return QueryExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                error_message=f"Query execution timed out after {self.max_execution_time} seconds",
                error_code="TIMEOUT_ERROR",
                execution_time=time.time() - start_time
            )
    
    async def _execute_query_internal(
        self, 
        sql: str, 
        parameters: Dict[str, Any], 
        max_rows: int
    ) -> QueryExecutionResult:
        """Internal query execution with connection management"""
        
        try:
            # Use database session manager
            with db_manager.get_session() as session:
                
                # Prepare SQL text with parameters
                sql_text_obj = sql_text(sql)
                
                # Execute query
                result_proxy = session.execute(sql_text_obj, parameters)
                
                # Fetch results
                try:
                    data = [dict(row) for row in result_proxy.mappings().fetchmany(max_rows)]
                except sqlalchemy_exc.ResourceClosedError:
                    # This happens for non-returning statements (e.g., INSERT, UPDATE, DELETE)
                    return QueryExecutionResult(
                        status=ExecutionStatus.SUCCESS,
                        data=[],
                        row_count=0, # Can't get rowcount here
                        column_info=[]
                    )

                # Convert to list of dictionaries
                if data:
                    # Get column names
                    columns = list(data[0].keys())
                    
                    # Generate column information
                    column_info = [
                        {"name": col, "type": str(type(data[0][col]).__name__)}
                        for col in columns
                    ]
                    
                    return QueryExecutionResult(
                        status=ExecutionStatus.SUCCESS,
                        data=data,
                        row_count=len(data),
                        column_info=column_info,
                        warnings=[] if len(data) < max_rows else 
                                 [f"Result limited to {max_rows} rows"]
                    )
                else:
                    return QueryExecutionResult(
                        status=ExecutionStatus.SUCCESS,
                        data=[],
                        row_count=0,
                        column_info=[]
                    )
                
        except AttributeError as e:
            logger.error(f"Database session error: {e}")
            return QueryExecutionResult(
                status=ExecutionStatus.ERROR,
                error_message=str(e),
                error_code="DB_ATTRIBUTE_ERROR"
            )
        except sqlalchemy_exc.SQLAlchemyError as e:
            error_code = type(e).__name__
            return QueryExecutionResult(
                status=ExecutionStatus.ERROR,
                error_message=str(e),
                error_code=error_code
            )
        return QueryExecutionResult(
            status=ExecutionStatus.ERROR,
            error_message="Unknown error in _execute_query_internal",
            error_code="UNKNOWN_ERROR"
        )
    
    def _calculate_cache_ttl(self, complexity: QueryComplexity, execution_time: float) -> int:
        """Calculate appropriate cache TTL based on query characteristics"""
        
        # Base TTL by complexity
        base_ttl = {
            QueryComplexity.SIMPLE: 300,      # 5 minutes
            QueryComplexity.MODERATE: 600,    # 10 minutes  
            QueryComplexity.COMPLEX: 1200,    # 20 minutes
            QueryComplexity.ADVANCED: 1800    # 30 minutes
        }
        
        ttl = base_ttl.get(complexity, 300)
        
        # Longer cache for slower queries (they're expensive)
        if execution_time > 2.0:
            ttl *= 2
        elif execution_time > 5.0:
            ttl *= 3
        
        return min(ttl, 3600)  # Max 1 hour
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_entries = len(self.cache.cache)
        
        if total_entries == 0:
            return {"total_entries": 0, "hit_rate": 0.0}
        
        total_access = sum(entry.access_count for entry in self.cache.cache.values())
        recent_access = sum(
            1 for entry in self.cache.cache.values()
            if entry.last_access and entry.last_access > datetime.now() - timedelta(minutes=10)
        )
        
        return {
            "total_entries": total_entries,
            "total_access_count": total_access,
            "recent_activity": recent_access,
            "cache_size_limit": self.cache.max_size,
            "cache_utilization": (total_entries / self.cache.max_size) * 100
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get query performance statistics"""
        return {
            "total_query_types": len(self.performance_monitor.query_stats),
            "query_stats": self.performance_monitor.query_stats
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform system health check"""
        try:
            # Test basic database connectivity
            start_time = time.time()
            test_result = await self._execute_query_internal(
                "SELECT 1 as health_check",
                {},
                1
            )
            db_response_time = time.time() - start_time
            
            # Clear expired cache entries
            expired_count = self.cache.clear_expired()
            
            return {
                "status": "healthy" if test_result.status == ExecutionStatus.SUCCESS else "unhealthy",
                "database_response_time": db_response_time,
                "cache_stats": self.get_cache_stats(),
                "expired_cache_entries_cleared": expired_count,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


# Main execution function for integration with workflow
if __name__ == "__main__":
    # Test the query executor
    async def test_query_executor():
        from src.langgraph.sql_generator import generate_sql_query
        
        # Test cases
        test_cases = [
            {
                "intent": "stock_analysis",
                "entities": {"stocks": ["AAPL"], "metrics": ["price", "volume"]},
                "user_query": "What's the current price and volume of Apple stock?"
            }
        ]
        
        executor = DatabaseQueryExecutor()
        
        for test_case in test_cases:
            print(f"\n{'='*60}")
            print(f"TESTING: {test_case['user_query']}")
            print(f"{'='*60}")
            
            # Generate SQL
            sql_result = await generate_sql_query(
                test_case["intent"],
                test_case["entities"],
                test_case["user_query"]
            )
            
            print(f"GENERATED SQL:")
            print(sql_result.sql)
            print(f"\nCOMPLEXITY: {sql_result.complexity.value}")
            
            # Execute query
            execution_result = await executor.execute_query(sql_result)
            
            print(f"\nEXECUTION RESULT:")
            print(f"Status: {execution_result.status.value}")
            print(f"Rows: {execution_result.row_count}")
            print(f"Time: {execution_result.execution_time:.3f}s")
            print(f"Cached: {execution_result.cached}")
            
            if execution_result.warnings:
                print(f"Warnings: {execution_result.warnings}")
                
            if execution_result.error_message:
                print(f"Error: {execution_result.error_message}")
            
            if execution_result.data and len(execution_result.data) > 0:
                print(f"Sample Data: {execution_result.data[0]}")
        
        # Test health check
        print(f"\n{'='*60}")
        print("HEALTH CHECK")
        print(f"{'='*60}")
        health = await executor.health_check()
        print(json.dumps(health, indent=2))
    
    # Run tests
    asyncio.run(test_query_executor())
