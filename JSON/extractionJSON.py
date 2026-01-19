import io, base64, fitz, json, os

from PIL import Image
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI

load_dotenv()

PDF_JSON_EXTRACTOR_PROMPT = '''
You are a high-precision document extraction engine with ZERO tolerance for errors.

CRITICAL INSTRUCTIONS:
1. You will receive multiple images labeled "PAGE X of Y"
2. Extract data from EVERY SINGLE page - do NOT skip any
3. Return EXACTLY N JSON objects where N = number of pages provided
4. Each object corresponds to one page in order
5. Use ONLY visible information from each page
6. Do NOT combine data from multiple pages
7. Do NOT guess or infer values - use null for missing data
8. Preserve exact formatting (numbers, dates, units)
9. Return ONLY the JSON array - no markdown or explanations
10. If image is blank/empty, return object with all fields as null
'''

BATCH_SIZE = 5  # 40 images per batch (under the 50 image limit)


def pdf_to_images(pdf_path, dpi=150):
    """Convert PDF pages to images with optimized DPI for vision accuracy"""
    doc = fitz.open(pdf_path)
    images = []

    for page_num, page in enumerate(doc, 1):
        pix = page.get_pixmap(dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        img.page_number = page_num
        images.append(img)

    return images


def batch_images(images, size=BATCH_SIZE):
    for i in range(0, len(images), size):
        yield images[i:i + size], i


def build_vision_content(images, instruction):
    content = [{"type": "text", "text": instruction}]

    for idx, img in enumerate(images):
        page_label = f"PAGE {idx + 1} of {len(images)}: "
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
        content.append({"type": "text", "text": page_label})

    return content


def validate_extracted_data(data_list: list) -> dict:
    report = {
        "total_pages": len(data_list),
        "missing_pages": [],
        "duplicate_pages": [],
        "pages_with_errors": [],
        "pages_with_null_data": []
    }

    page_numbers_found = set()

    for item in data_list:
        page_num = item.get("page_number")

        if page_num in page_numbers_found:
            report["duplicate_pages"].append(page_num)
        page_numbers_found.add(page_num)

        if "error" in item.get("data", {}):
            report["pages_with_errors"].append(page_num)

        data_values = [
            v for k, v in item.get("data", {}).items()
            if k != "error"
        ]
        null_count = sum(1 for v in data_values if v is None)

        if data_values and null_count > len(data_values) * 0.7:
            report["pages_with_null_data"].append(page_num)

    return report


llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    deployment_name=os.getenv("OPENAI_DEPLOYMENT_NAME"),
    temperature=0,
    max_tokens=4000
)


def save_json_to_file(data: str, filename: str = "extracted_data.json") -> str:
    try:
        filepath = os.path.abspath(filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(data)

        file_size = os.path.getsize(filepath)
        print(f"\n[SUCCESS] File saved: {filepath}")
        print(f"[INFO] File size: {file_size / 1024:.2f} KB")
        return filepath

    except Exception as e:
        print(f"[ERROR] Failed to save file: {str(e)}")
        return None


@tool
def extract_pdf_as_json(pdf_path: str, fields_json: str) -> str:
    """Extract JSON data from a PDF file based on specified fields."""
    schema = json.loads(fields_json)
    images = pdf_to_images(pdf_path)
    all_pages_data = []

    print(f"\n[INFO] PDF loaded with {len(images)} pages")
    print(f"[INFO] Processing in batches of {BATCH_SIZE} pages...")

    extraction_instruction = (
        "CRITICAL: Extract the following fields from EVERY SINGLE page in this batch. "
        "Return a JSON ARRAY with EXACTLY one object per page - same number as images provided. "
        "Do NOT skip any pages. Do NOT combine pages. One page = one object. "
        "Return ONLY valid JSON array, no markdown:\n"
        f"{json.dumps(schema, indent=2)}"
    )

    batch_count = 0
    pages_sent = {}

    for batch, start_idx in batch_images(images):
        batch_count += 1
        end_page = min(start_idx + len(batch), len(images))
        pages_in_batch = list(range(start_idx + 1, end_page + 1))
        pages_sent[batch_count] = pages_in_batch

        print(
            f"[INFO] Batch {batch_count}: Pages {start_idx + 1}-{end_page} "
            f"({len(batch)} images)...",
            end=" ",
            flush=True
        )

        messages = [
            SystemMessage(content=PDF_JSON_EXTRACTOR_PROMPT),
            HumanMessage(content=build_vision_content(batch, extraction_instruction))
        ]

        response = llm.invoke(messages)

        try:
            raw_response = response.content.strip()

            if raw_response.startswith("```json"):
                raw_response = raw_response[7:]
            if raw_response.startswith("```"):
                raw_response = raw_response[3:]
            if raw_response.endswith("```"):
                raw_response = raw_response[:-3]

            raw_response = raw_response.strip()
            batch_data = json.loads(raw_response)

            if isinstance(batch_data, list):
                for idx, page_data in enumerate(batch_data):
                    all_pages_data.append({
                        "page_number": start_idx + idx + 1,
                        "data": page_data
                    })

                status = "[OK]" if len(batch_data) == len(batch) else "[MISMATCH]"
                print(f"{status} {len(batch_data)}/{len(batch)} pages")

            else:
                all_pages_data.append({
                    "page_number": start_idx + 1,
                    "data": batch_data
                })

                status = "[OK]" if len(batch) == 1 else "[MISMATCH]"
                print(f"{status} 1/{len(batch)} pages")

        except json.JSONDecodeError as e:
            print(f"[ERROR] {str(e)[:40]}")
            all_pages_data.append({
                "page_number": start_idx + 1,
                "data": {
                    "error": f"JSON parse failed: {str(e)[:30]}",
                    "raw_length": len(response.content)
                }
            })

    extracted_page_nums = {item["page_number"] for item in all_pages_data}
    all_expected_pages = set(range(1, len(images) + 1))
    missing_pages = sorted(all_expected_pages - extracted_page_nums)

    print(f"\n[INFO] Total pages extracted: {len(all_pages_data)}/{len(images)}")
    if missing_pages:
        print(f"[WARNING] Missing pages: {missing_pages}")
        print("[INFO] These pages may be blank, error pages, or had parsing issues")

    return json.dumps(all_pages_data, indent=2)


tools = [extract_pdf_as_json]

agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a document intelligence agent. Use tools to extract structured information from PDFs."),
    ("human", "{input}")
])

llm_with_tools = llm.bind_tools(tools)
agent = agent_prompt | llm_with_tools
agent_executor = agent


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

    result = agent_executor.invoke({"input": query})

    if hasattr(result, "tool_calls") and result.tool_calls:
        for tool_call in result.tool_calls:
            if tool_call["name"] == "extract_pdf_as_json":
                final_json = extract_pdf_as_json.invoke(tool_call["args"])
            else:
                final_json = None
    else:
        final_json = result.content if hasattr(result, "content") else str(result)

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)

    filepath = save_json_to_file(final_json, "extracted_data.json")

    try:
        data_list = json.loads(final_json)
        print(f"[INFO] Total pages extracted: {len(data_list)}")

        validation = validate_extracted_data(data_list)
        print("\n[VALIDATION REPORT]")
        print(f"  Pages with errors: {len(validation['pages_with_errors'])} {validation['pages_with_errors']}")
        print(f"  Pages with mostly null data: {len(validation['pages_with_null_data'])} {validation['pages_with_null_data']}")
        print(f"  Duplicate pages: {len(validation['duplicate_pages'])} {validation['duplicate_pages']}")

        if validation["pages_with_errors"] or validation["pages_with_null_data"]:
            print("\n[WARNING] Review the pages listed above - they may have extraction issues")

        print("\nFirst 2 pages preview:")
        print(json.dumps(data_list[:2], indent=2))

    except Exception as e:
        print(f"Could not parse JSON: {str(e)}")

    print("=" * 60)
