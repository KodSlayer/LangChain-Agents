"""
Simple AI Assistant using Groq API and Tavily Search with LangChain
Requirements: pip install langchain langchain-groq tavily-python
"""

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from tavily import TavilyClient
import os
from dotenv import load_dotenv
_ = load_dotenv()

# Initialize API clients
def create_assistant(groq_api_key, tavily_api_key, temperature=0.7):
    """Create the AI assistant with Groq and Tavily
    
    Args:
        groq_api_key: Your Groq API key
        tavily_api_key: Your Tavily API key
        temperature: Controls randomness (0.0-1.0). Lower = more focused, Higher = more creative
    """
    
    # Create Groq LLM
    llm = ChatGroq(
        api_key=groq_api_key,
        model="llama-3.3-70b-versatile",
        temperature=temperature
    )
    
    # Create Tavily search client
    tavily = TavilyClient(api_key=tavily_api_key)
    
    return llm, tavily


def search_web(tavily, query):
    """Search the web using Tavily"""
    print("🔍 Searching the web...")
    
    # Perform search
    results = tavily.search(query=query, max_results=3)
    
    # Format results
    search_context = ""
    for result in results['results']:
        search_context += f"Title: {result['title']}\n"
        search_context += f"Content: {result['content']}\n"
        search_context += f"Source: {result['url']}\n\n"
    
    return search_context


def chat(llm, tavily, user_message):
    """Send message to AI and get response"""
    
    # Always search for every query to get the most current information
    print("🔍 Searching the web...")
    
    # Get search results
    search_results = search_web(tavily, user_message)
    
    # Create messages with search context
    messages = [
        SystemMessage(content="You are a helpful AI assistant. Use the search results below to answer the question accurately. Provide current and relevant information."),
        HumanMessage(content=f"Search Results:\n{search_results}\n\nQuestion: {user_message}\n\nAnswer based on the search results above.")
    ]
    
    # Get response from Groq
    response = llm.invoke(messages)
    
    return response.content


def main():
    """Main function"""
    
    print("=" * 60)
    print("Simple AI Assistant")
    print("=" * 60)
    
    # Get API keys
    groq_key = os.getenv("GROQ_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    
    # Get temperature (optional)
    temp_input = input("Enter temperature (0.0-1.0, default 0.7): ").strip()
    temperature = float(temp_input) if temp_input else 0.7
    
    # Create assistant
    print(f"\n🤖 Starting assistant with temperature {temperature}...\n")
    llm, tavily = create_assistant(groq_key, tavily_key, temperature)
    
    print("Assistant ready! All queries will search the web.")
    print("Type 'quit' to exit.\n")
    
    # Chat loop
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        # Check if user wants to quit
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        # Get response
        try:
            response = chat(llm, tavily, user_input)
            print(f"\nAssistant: {response}\n")
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()