"""
Streamlit Frontend Application
Real-Time Financial Advisory System - Phase 6

This module provides a sophisticated web interface for the AI-powered financial advisory system,
integrating all backend components into an intuitive user experience.
"""

import streamlit as st
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Import your backend components
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langgraph.intent_entity_extractor import IntentEntityExtractor
from langgraph.sql_generator import generate_sql_query
from langgraph.query_executor import DatabaseQueryExecutor, ExecutionStatus
from langgraph.response_formatter import format_financial_response, ResponseStyle


class FinancialAdvisoryUI:
    """
    Streamlit-based user interface for the Financial Advisory System
    """
    
    def __init__(self):
        self.setup_page_config()
        self.setup_session_state()
        self.extractor = IntentEntityExtractor()
        self.executor = DatabaseQueryExecutor()
    
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="AI Financial Advisory System",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better styling
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1E88E5;
            text-align: center;
            margin-bottom: 2rem;
        }
        .response-container {
            background-color: #f0f2f6;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .insight-box {
            background-color: #e8f5e8;
            padding: 1rem;
            border-left: 4px solid #4CAF50;
            margin: 0.5rem 0;
        }
        .warning-box {
            background-color: #fff3cd;
            padding: 1rem;
            border-left: 4px solid #ffc107;
            margin: 0.5rem 0;
        }
        .recommendation-box {
            background-color: #e3f2fd;
            padding: 1rem;
            border-left: 4px solid #2196F3;
            margin: 0.5rem 0;
        }
        .metric-card {
            background-color: white;
            padding: 1rem;
            border-radius: 0.3rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
            margin: 0.5rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def setup_session_state(self):
        """Initialize session state variables"""
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        if 'query_count' not in st.session_state:
            st.session_state.query_count = 0
        if 'last_response' not in st.session_state:
            st.session_state.last_response = None
    
    def render_header(self):
        """Render the application header"""
        st.markdown('<div class="main-header">🤖 AI Financial Advisory System</div>', unsafe_allow_html=True)
        st.markdown("---")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Queries", st.session_state.query_count)
        with col2:
            st.metric("Session Started", datetime.now().strftime("%H:%M"))
        with col3:
            st.metric("System Status", "🟢 Online")
        with col4:
            if st.button("🗑️ Clear History"):
                st.session_state.conversation_history = []
                st.session_state.query_count = 0
                st.rerun()
    
    def render_sidebar(self):
        """Render the sidebar with examples and settings"""
        with st.sidebar:
            st.header("📋 Quick Examples")
            
            example_queries = [
                "What's the current price and volume of Apple stock?",
                "Show me the risk metrics for Tesla (TSLA)",
                "What's the recent sentiment for Microsoft?",
                "Compare Google and Amazon trading volumes",
                "What are the RSI and MACD indicators for Netflix?",
                "How volatile is Amazon stock?",
                "What's the beta for Apple compared to the market?"
            ]
            
            for i, example in enumerate(example_queries):
                if st.button(f"📊 {example}", key=f"example_{i}"):
                    st.session_state.selected_query = example
                    st.rerun()
            
            st.header("⚙️ Settings")
            
            response_style = st.selectbox(
                "Response Style",
                ["conversational", "analytical", "advisory", "educational", "executive"],
                index=0
            )
            st.session_state.response_style = response_style
            
            show_technical_details = st.checkbox("Show Technical Details", False)
            st.session_state.show_technical_details = show_technical_details
            
            show_query_performance = st.checkbox("Show Query Performance", False)
            st.session_state.show_query_performance = show_query_performance
            
            st.header("📈 System Stats")
            if st.button("🔍 Health Check"):
                with st.spinner("Checking system health..."):
                    health_status = asyncio.run(self.executor.health_check())
                    st.json(health_status)
    
    def render_query_input(self):
        """Render the main query input interface"""
        st.header("💬 Ask Your Financial Question")
        
        # Check if there's a selected query from sidebar
        default_query = ""
        if hasattr(st.session_state, 'selected_query'):
            default_query = st.session_state.selected_query
            del st.session_state.selected_query
        
        user_query = st.text_input(
            "Enter your financial question:",
            value=default_query,
            placeholder="e.g., What's the current price of Apple stock?",
            key="user_input"
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.button("🚀 Analyze", type="primary", use_container_width=True):
                if user_query.strip():
                    self.process_user_query(user_query.strip())
                else:
                    st.warning("Please enter a financial question.")
        
        with col2:
            if st.button("🔄 Try Random", use_container_width=True):
                import random
                random_examples = [
                    "What's the current price of AAPL?",
                    "Show me TSLA risk metrics",
                    "MSFT sentiment analysis",
                    "Compare GOOGL and AMZN",
                    "Netflix technical indicators"
                ]
                random_query = random.choice(random_examples)
                st.session_state.selected_query = random_query
                st.rerun()
        
        with col3:
            if st.button("📊 Portfolio", use_container_width=True):
                st.session_state.selected_query = "Show me my portfolio performance"
                st.rerun()
    
    def process_user_query(self, user_query: str):
        """Process the user query through the financial advisory pipeline"""
        
        start_time = time.time()
        
        # Progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            with st.spinner("Processing your financial query..."):
                
                # Step 1: Extract Intent and Entities
                status_text.text("🧠 Understanding your question...")
                progress_bar.progress(20)
                
                extraction_result = asyncio.run(
                    self.extractor.extract_intent_and_entities(user_query)
                )
                
                if not extraction_result:
                    st.error("❌ Could not understand the query. Please try rephrasing.")
                    return
                
                intent = extraction_result.get("intent", "stock_analysis")
                entities = extraction_result.get("entities", {})
                
                # Step 2: Generate SQL Query
                status_text.text("⚙️ Generating database query...")
                progress_bar.progress(40)
                
                sql_result = asyncio.run(
                    generate_sql_query(intent, entities, user_query)
                )
                
                if not sql_result.sql or "Error" in sql_result.sql:
                    st.error(f"❌ Error generating query: {sql_result.reasoning}")
                    return
                
                # Step 3: Execute Query
                status_text.text("🗄️ Executing database query...")
                progress_bar.progress(60)
                
                execution_result = asyncio.run(
                    self.executor.execute_query(sql_result)
                )
                
                # Step 4: Format Response
                status_text.text("🎯 Formatting intelligent response...")
                progress_bar.progress(80)
                
                response_style_map = {
                    "conversational": ResponseStyle.CONVERSATIONAL,
                    "analytical": ResponseStyle.ANALYTICAL,
                    "advisory": ResponseStyle.ADVISORY,
                    "educational": ResponseStyle.EDUCATIONAL,
                    "executive": ResponseStyle.EXECUTIVE
                }
                
                style = response_style_map.get(
                    st.session_state.get('response_style', 'conversational'),
                    ResponseStyle.CONVERSATIONAL
                )
                
                formatted_response = asyncio.run(
                    format_financial_response(
                        user_query, intent, entities, sql_result, execution_result, style
                    )
                )
                
                # Step 5: Display Results
                status_text.text("✅ Analysis complete!")
                progress_bar.progress(100)
                
                processing_time = time.time() - start_time
                
                # Store in session state
                st.session_state.last_response = {
                    'query': user_query,
                    'intent': intent,
                    'entities': entities,
                    'sql_result': sql_result,
                    'execution_result': execution_result,
                    'formatted_response': formatted_response,
                    'processing_time': processing_time,
                    'timestamp': datetime.now()
                }
                
                st.session_state.conversation_history.append(st.session_state.last_response)
                st.session_state.query_count += 1
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Display the response
                self.display_response()
                
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ System Error: {str(e)}")
            st.exception(e)
    
    def display_response(self):
        """Display the formatted response with all components"""
        
        if not st.session_state.last_response:
            return
        
        response_data = st.session_state.last_response
        formatted_response = response_data['formatted_response']
        execution_result = response_data['execution_result']
        
        st.markdown("## 🎯 Analysis Results")
        
        # Main Response
        st.markdown('<div class="response-container">', unsafe_allow_html=True)
        st.markdown("### 💬 AI Financial Advisor Response")
        st.write(formatted_response.natural_language)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Create columns for organized display
        col1, col2 = st.columns(2)
        
        with col1:
            # Key Insights
            if formatted_response.key_insights:
                st.markdown("### 💡 Key Insights")
                for insight in formatted_response.key_insights:
                    st.markdown(f'<div class="insight-box">📊 {insight}</div>', unsafe_allow_html=True)
            
            # Recommendations
            if formatted_response.recommendations:
                st.markdown("### 🎯 Recommendations")
                for rec in formatted_response.recommendations:
                    st.markdown(f'<div class="recommendation-box">💡 {rec}</div>', unsafe_allow_html=True)
        
        with col2:
            # Risk Warnings
            if formatted_response.risk_warnings:
                st.markdown("### ⚠️ Risk Considerations")
                for warning in formatted_response.risk_warnings:
                    st.markdown(f'<div class="warning-box">⚠️ {warning}</div>', unsafe_allow_html=True)
            
            # Confidence and Metadata
            st.markdown("### 📊 Analysis Metadata")
            confidence_color = {
                "high": "🟢",
                "medium": "🟡", 
                "low": "🟠",
                "insufficient": "🔴"
            }
            
            conf_icon = confidence_color.get(formatted_response.confidence.value, "⚪")
            st.write(f"**Confidence Level:** {conf_icon} {formatted_response.confidence.value.title()}")
            st.write(f"**Processing Time:** {response_data['processing_time']:.2f} seconds")
            st.write(f"**Data Points:** {execution_result.row_count}")
        
        # Data Visualization
        if execution_result.status == ExecutionStatus.SUCCESS and execution_result.data:
            self.render_data_visualization(execution_result.data, response_data['intent'])
        
        # Follow-up Suggestions
        if formatted_response.follow_up_suggestions:
            st.markdown("### 🔄 Suggested Follow-up Questions")
            cols = st.columns(2)
            for i, suggestion in enumerate(formatted_response.follow_up_suggestions[:4]):
                col_idx = i % 2
                with cols[col_idx]:
                    if st.button(f"❓ {suggestion}", key=f"followup_{i}"):
                        st.session_state.selected_query = suggestion
                        st.rerun()
        
        # Technical Details (if enabled)
        if st.session_state.get('show_technical_details', False):
            self.render_technical_details(response_data)
        
        # Query Performance (if enabled)
        if st.session_state.get('show_query_performance', False):
            self.render_query_performance(response_data)
        
        # Disclaimer
        if formatted_response.disclaimer:
            st.markdown("---")
            st.caption(f"📝 **Disclaimer:** {formatted_response.disclaimer}")
    
    def render_data_visualization(self, data: List[Dict], intent: str):
        """Render data visualizations based on the query intent and data"""
        
        if not data:
            return
        
        st.markdown("### 📈 Data Visualization")
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(data)
        
        # Stock Analysis Visualization
        if intent == "stock_analysis" and 'close_price' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                if 'volume' in df.columns:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=df.get('symbol', ['Stock']),
                        y=df['volume'],
                        name='Volume',
                        marker_color='lightblue'
                    ))
                    fig.update_layout(title="Trading Volume", xaxis_title="Stock", yaxis_title="Volume")
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'close_price' in df.columns:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=df.get('symbol', ['Stock']),
                        y=df['close_price'],
                        name='Price',
                        marker_color='lightgreen'
                    ))
                    fig.update_layout(title="Stock Price", xaxis_title="Stock", yaxis_title="Price ($)")
                    st.plotly_chart(fig, use_container_width=True)
        
        # Technical Indicators Visualization
        if intent == "technical_analysis" or "rsi_14" in df.columns:
            if 'rsi_14' in df.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df.get('symbol', range(len(df))),
                    y=df['rsi_14'],
                    mode='lines+markers',
                    name='RSI',
                    line=dict(color='purple')
                ))
                fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
                fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
                fig.update_layout(title="RSI Indicator", xaxis_title="Stock", yaxis_title="RSI Value")
                st.plotly_chart(fig, use_container_width=True)
        
        # Risk Metrics Visualization
        if intent == "risk_assessment" or "volatility_30d" in df.columns:
            risk_cols = ['volatility_30d', 'beta', 'sharpe_ratio']
            available_risk_cols = [col for col in risk_cols if col in df.columns]
            
            if available_risk_cols:
                risk_df = df[['symbol'] + available_risk_cols] if 'symbol' in df.columns else df[available_risk_cols]
                
                fig = go.Figure()
                for col in available_risk_cols:
                    fig.add_trace(go.Bar(
                        x=df.get('symbol', [f'Stock {i}' for i in range(len(df))]),
                        y=df[col],
                        name=col.replace('_', ' ').title()
                    ))
                
                fig.update_layout(title="Risk Metrics Comparison", barmode='group')
                st.plotly_chart(fig, use_container_width=True)
        
        # Raw Data Table
        with st.expander("📋 View Raw Data"):
            st.dataframe(df, use_container_width=True)
    
    def render_technical_details(self, response_data):
        """Render technical details about the query execution"""
        
        with st.expander("🔧 Technical Details"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Query Analysis:**")
                st.json({
                    "Intent": response_data['intent'],
                    "Entities": response_data['entities'],
                    "SQL Complexity": response_data['sql_result'].complexity.value,
                    "Tables Involved": response_data['sql_result'].tables_involved
                })
            
            with col2:
                st.markdown("**Execution Metrics:**")
                exec_result = response_data['execution_result']
                st.json({
                    "Status": exec_result.status.value,
                    "Execution Time": f"{exec_result.execution_time:.3f}s",
                    "Row Count": exec_result.row_count,
                    "Cached": exec_result.cached,
                    "Warnings": exec_result.warnings
                })
            
            st.markdown("**Generated SQL Query:**")
            st.code(response_data['sql_result'].sql, language="sql")
    
    def render_query_performance(self, response_data):
        """Render query performance analysis"""
        
        with st.expander("⚡ Performance Analysis"):
            exec_result = response_data['execution_result']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Query Time", f"{exec_result.execution_time:.3f}s")
            with col2:
                st.metric("Rows Returned", exec_result.row_count)
            with col3:
                st.metric("Cache Status", "Hit" if exec_result.cached else "Miss")
            with col4:
                processing_time = response_data['processing_time']
                st.metric("Total Time", f"{processing_time:.2f}s")
            
            if exec_result.performance_metrics:
                st.json(exec_result.performance_metrics)
    
    def render_conversation_history(self):
        """Render conversation history"""
        
        if st.session_state.conversation_history:
            st.markdown("## 💭 Conversation History")
            
            for i, conv in enumerate(reversed(st.session_state.conversation_history[-5:])):
                with st.expander(f"Query {len(st.session_state.conversation_history) - i}: {conv['query'][:50]}..."):
                    st.write(f"**Time:** {conv['timestamp'].strftime('%H:%M:%S')}")
                    st.write(f"**Intent:** {conv['intent']}")
                    st.write(f"**Response:** {conv['formatted_response'].natural_language[:200]}...")
                    
                    if st.button(f"🔄 Ask Again", key=f"repeat_{i}"):
                        st.session_state.selected_query = conv['query']
                        st.rerun()
    
    def run(self):
        """Main application runner"""
        
        # Header
        self.render_header()
        
        # Sidebar
        self.render_sidebar()
        
        # Main content
        self.render_query_input()
        
        # Display last response if available
        if st.session_state.last_response:
            self.display_response()
        
        # Conversation history
        self.render_conversation_history()
        
        # Footer
        st.markdown("---")
        st.markdown(
            "🤖 **AI Financial Advisory System** | "
            "Built with advanced LLM reasoning and real-time data analysis | "
            f"Session: {st.session_state.query_count} queries processed"
        )


if __name__ == "__main__":
    app = FinancialAdvisoryUI()
    app.run()