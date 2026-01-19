'''
This is the project to extract the data from PDF into JSON format using LLMs.
We have used LangChain framework to build this simple extraction engine.
The extraction is done using gpt-4o model from OpenAI.
'''
import io, base64, fitz, json
from PIL import Image
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI

from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()


PDF_JSON_EXTRACTOR_PROMPT = ( '''
You are a high-precision document extraction engine.

Your task is to extract structured information ONLY from the provided document images (PDF pages rendered as images).

Rules:
- Use ONLY what is explicitly visible in the document images.
- Do NOT guess, infer, assume, or calculate missing values.
- Do NOT use external knowledge.
- If a field is not clearly present, set its value to null.
- Preserve numbers, dates, units, and formatting exactly as shown.
- Do NOT add explanations, comments, markdown, or extra text.
- Output MUST be strictly valid JSON and match the provided schema exactly.
- Return ONLY the JSON object and nothing else.''')


BATCH_SIZE = 40

def pdf_to_images(pdf_path, dpi=200):                   #pdf to image conversion using PyMuPDF
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append(img)
    return images


def batch_images(images, size=BATCH_SIZE):
    for i in range(0, len(images), size):
        yield images[i:i + size]


def build_vision_content(images, instruction):
    content = [{"type": "text", "text": instruction}]
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_b64}",
                "detail": "high"
            }
        })
    return content


llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    deployment_name=os.getenv("OPENAI_DEPLOYMENT_NAME"),
    temperature=0,
    max_tokens=1000
)

@tool                                                                     #Extraction tool
def extract_pdf_as_json(pdf_path: str, fields_json: str) -> str:
    """
    Extracts structured information from any PDF using GPT-4o Vision.
    fields_json must be a JSON schema defining required fields.
    """
    schema = json.loads(fields_json)
    images = pdf_to_images(pdf_path)
    partial_results = []

    extraction_instruction = (
        f"Extract the following fields from this document and return JSON only:\n"
        f"{json.dumps(schema, indent=2)}"
    )

    for batch in batch_images(images):
        messages = [
            SystemMessage(content=PDF_JSON_EXTRACTOR_PROMPT),
            HumanMessage(content=build_vision_content(batch, extraction_instruction))
        ]
        response = llm.invoke(messages)
        partial_results.append(response.content)

    # Final merge step
    merge_prompt = (
        "Merge the following partial JSON objects into one final JSON. "
        "Prefer non-null values and keep the structure unchanged.\n\n"
        + "\n".join(partial_results)
    )

    final = llm.invoke(merge_prompt).content
    return final


# Langchain Agent setup

tools = [extract_pdf_as_json]

# Create a simple LangChain agent with tool binding
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a document intelligence agent. Use tools to extract structured information from PDFs."),
    ("human", "{input}")
])

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)

# Create a simple agent chain
agent = agent_prompt | llm_with_tools

# Create agent executor using LangChain
agent_executor = agent

## This is the main function to run the extraction

if __name__ == "__main__":

    pdf_path = "R59SAR_ITC0006_2093761 - test8.pdf"

    extraction_schema = {
        "document_type": "Stock Accounting Summary",

        "company_details": {
            "operator_company": None,
            "operator_address": None,
            "customer_company": None,
            "customer_address": None
        },

        "product_details": {
            "product_code": None,
            "product_name": None,
            "unit": "Barrels"
        },

        "report_details": {
            "page_number": None,
            "begin_date": None,
            "end_date": None,
            "tsa_number": None
        },

        "transaction_summary": {
            "total_received": None,
            "total_shipped": None,
            "total_vapor": None,
            "total_nitrogen": None
        },

        "inventory_summary": {
            "beginning_inventory": None,
            "total_net_movements": None,
            "closing_book_inventory": None,
            "tank_inventory": None,
            "line_inventory": None,
            "total_physical_inventory": None,
            "variation": None
        },

        "measurement_details": {
            "low_gauge": None,
            "safe_fill_height": None,
            "ending_tank_gauge": None,
            "swing_gauge": None,
            "measurement_timestamp": None
        }
    }


    query = (
        f"Extract structured information from the PDF at {pdf_path}. "
        f"Use this JSON schema:\n{json.dumps(extraction_schema)}"
    )

    # Simple invoke - LangChain will handle tool calling
result = agent_executor.invoke({"input": query})

# Process tool calls if they exist
if hasattr(result, 'tool_calls') and result.tool_calls:
    
    for tool_call in result.tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        
        # Execute the tool using invoke
        if tool_name == 'extract_pdf_as_json':
            final_json = extract_pdf_as_json.invoke(tool_args)
        else:
            final_json = None
else:
    # No tool calls, use the content directly
    final_json = result.content if hasattr(result, "content") else str(result)

print("\n" + "="*60)
print("EXTRACTED JSON OUTPUT:")
print("="*60)
print(final_json)
print("="*60)


