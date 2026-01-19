import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage

# Load env variables
load_dotenv()

# Create LangChain LLM connection
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    deployment_name=os.getenv("OPENAI_DEPLOYMENT_NAME"),
    temperature=0.7,
)

# Ask a question
response = llm.invoke([
    HumanMessage(content="What is crewAI?")
])

# Print response
print(response.content)
