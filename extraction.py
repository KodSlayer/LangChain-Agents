"""
Simple PDF Extraction with Azure OpenAI
Ask any question about your PDF and get precise answers
"""

# Install required packages:
# pip install langchain langchain-openai pypdf

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
load_dotenv()
# ============================================================================
# CONFIGURATION
# ============================================================================

class AzureConfig:
    """Your Azure OpenAI credentials"""
    AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")  # e.g., https://your-resource.openai.azure.com/
    API_KEY = os.getenv("OPENAI_API_KEY")
    DEPLOYMENT_NAME = os.getenv("OPENAI_DEPLOYMENT_NAME")  # e.g., gpt-4, gpt-35-turbo
    API_VERSION = os.getenv("OPENAI_API_VERSION")  # e.g., 2024-06-01-preview

# ============================================================================
# SIMPLE PDF Q&A FUNCTION
# ============================================================================

def extract_from_pdf(pdf_path: str, user_prompt: str):
    """
    Extract information from PDF based on your question
    
    Args:
        pdf_path: Path to your PDF file
        user_prompt: Your question/instruction (e.g., "give me the first line")
    
    Returns:
        Answer from the LLM based on PDF content
    """
    
    # Step 1: Load the PDF
    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    # Combine all pages into one text
    full_text = "\n\n".join([page.page_content for page in pages])
    
    print(f"Loaded {len(pages)} pages from PDF\n")
    
    # Step 2: Initialize Azure OpenAI
    llm = AzureChatOpenAI(
        azure_endpoint=AzureConfig.AZURE_ENDPOINT,
        api_key=AzureConfig.API_KEY,
        deployment_name=AzureConfig.DEPLOYMENT_NAME,
        api_version=AzureConfig.API_VERSION,
        temperature=0  # 0 = deterministic, more precise
    )
    
    # Step 3: Create prompt template
    prompt = PromptTemplate(
        input_variables=["pdf_content", "question"],
        template="""
You are analyzing a PDF document. Based on the content below, answer the user's question precisely.

PDF Content:
{pdf_content}

User Question: {question}

Answer:"""
    )
    
    # Step 4: Create and run the chain
    chain = prompt | llm                        #This is the LLMChain object

    print(f"Processing your request: '{user_prompt}'")

    #This is the LLM call
    result = chain.invoke({
        "pdf_content": full_text,
        "question": user_prompt
    }) 
    
    return result.content

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    
    # Your PDF file path
    pdf_path = "sample-local.pdf"
    
    # Example 1: Get first line
    print("="*60)
    print("Example 1: Extract first line")
    print("="*60)
    answer = extract_from_pdf(pdf_path, "Give me the first line of the PDF")
    print(f"\nAnswer: {answer}\n")
    
    # Example 2: Get specific information
#     print("="*60)
#     print("Example 2: Extract specific info")
#     print("="*60)
#     answer = extract_from_pdf(pdf_path, "What is the title of this document?")
#     print(f"\nAnswer: {answer}\n")
    
#     # Example 3: Get last paragraph
#     print("="*60)
#     print("Example 3: Extract last paragraph")
#     print("="*60)
#     answer = extract_from_pdf(pdf_path, "Give me the last paragraph of the PDF")
#     print(f"\nAnswer: {answer}\n")
    
#     # Example 4: Extract dates
#     print("="*60)
#     print("Example 4: Extract dates")
#     print("="*60)
#     answer = extract_from_pdf(pdf_path, "List all dates mentioned in this document")
#     print(f"\nAnswer: {answer}\n")
    
#     # Example 5: Count pages
#     print("="*60)
#     print("Example 5: Get page count")
#     print("="*60)
#     answer = extract_from_pdf(pdf_path, "How many pages does this document have?")
#     print(f"\nAnswer: {answer}\n")


# # ============================================================================
# # INTERACTIVE MODE
# # ============================================================================

# def interactive_pdf_chat(pdf_path: str):
#     """
#     Chat with your PDF interactively
#     """
#     print("\n" + "="*60)
#     print("Interactive PDF Chat - Type 'quit' to exit")
#     print("="*60 + "\n")
    
#     # Load PDF once
#     loader = PyPDFLoader(pdf_path)
#     pages = loader.load()
#     full_text = "\n\n".join([page.page_content for page in pages])
    
#     print(f"✓ Loaded PDF with {len(pages)} pages")
#     print(f"✓ Ready to answer your questions!\n")
    
#     # Initialize LLM
#     llm = AzureChatOpenAI(
#         azure_endpoint=AzureConfig.AZURE_ENDPOINT,
#         api_key=AzureConfig.API_KEY,
#         deployment_name=AzureConfig.DEPLOYMENT_NAME,
#         api_version=AzureConfig.API_VERSION,
#         temperature=0
#     )
    
#     prompt_template = PromptTemplate(
#         input_variables=["pdf_content", "question"],
#         template="""
# Based on the PDF content below, answer the question precisely and concisely.

# PDF Content:
# {pdf_content}

# Question: {question}

# Answer:"""
#     )
    
#     chain = LLMChain(llm=llm, prompt=prompt_template)
    
#     # Interactive loop
#     while True:
#         user_question = input("\nYour question: ").strip()
        
#         if user_question.lower() in ['quit', 'exit', 'q']:
#             print("Goodbye!")
#             break
            
#         if not user_question:
#             continue
        
#         try:
#             answer = chain.run(pdf_content=full_text, question=user_question)
#             print(f"\nAnswer: {answer}")
#         except Exception as e:
#             print(f"Error: {str(e)}")

# # Run interactive mode
# # Uncomment below to use:
# # interactive_pdf_chat("your_document.pdf")