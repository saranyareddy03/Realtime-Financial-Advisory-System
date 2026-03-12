import asyncio
from src.langgraph.sql_generator import generate_sql_query
from src.langgraph.query_executor import DatabaseQueryExecutor, ExecutionStatus
from src.langgraph.intent_entity_extractor import IntentEntityExtractor

from src.langgraph.response_formatter import format_financial_response

async def process_user_query(user_query: str):
    """
    Processes a user query by generating SQL, executing it, and returning the result.
    """
    print(f"\nProcessing query: '{user_query}'")

    # Extract intent and entities using the LLM
    extractor = IntentEntityExtractor()
    extraction_result = await extractor.extract_intent_and_entities(user_query)
    
    if not extraction_result:
        print("Could not understand the query. Please try again.")
        return

    intent = extraction_result.get("intent", "stock_analysis")
    entities = extraction_result.get("entities", {})

    print(f"Intent: {intent}, Entities: {entities}")

    # Generate SQL query
    sql_result = await generate_sql_query(intent, entities, user_query)
    
    if not sql_result.sql or "Error" in sql_result.sql:
        print(f"Error generating SQL query: {sql_result.reasoning}")
        return

    print(f"Generated SQL: {sql_result.sql}")

    # Execute the query
    executor = DatabaseQueryExecutor()
    execution_result = await executor.execute_query(sql_result)

    # Display results
    if execution_result.status == ExecutionStatus.SUCCESS:
        if execution_result.data:
            # Format the response
            formatted_response = await format_financial_response(
                user_query,
                intent,
                entities,
                sql_result,
                execution_result
            )
            print("Query Result:")
            print(formatted_response.natural_language)
        else:
            print("Query executed successfully, but no data was returned.")
    else:
        print(f"Error executing query: {execution_result.error_message}")

async def main():
    """
    Main function to run the user query processor.
    """
    hardcoded_questions = [
        "What is the current price and volume of Apple stock?",
        "What are the latest risk metrics for Tesla?",
        "Show me the recent news sentiment for Microsoft.",
        "Compare the trading volume of Google and Amazon.",
        "What are the technical indicators for Netflix?"
    ]

    print("Welcome to the Financial Advisory System!")
    print("You can ask questions about stocks, or choose from the following examples:")
    for i, question in enumerate(hardcoded_questions, 1):
        print(f"{i}. {question}")

    while True:
        choice = input("\nEnter the number of a question or type your own (or 'quit' to exit): ")
        if choice.lower() == 'quit':
            break

        try:
            question_index = int(choice) - 1
            if 0 <= question_index < len(hardcoded_questions):
                user_query = hardcoded_questions[question_index]
            else:
                print("Invalid number. Please try again.")
                continue
        except ValueError:
            user_query = choice

        await process_user_query(user_query)

if __name__ == "__main__":
    asyncio.run(main())
