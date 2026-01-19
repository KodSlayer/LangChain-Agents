'''
This extraction method is using PyMuPDF (fitz) to convert PDF pages to images and the images is passed into the llm model
for extraction. We have used GPT-4o Vision in this example but you can use any multimodal model which supports image
inputs.
'''

import base64
import io
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import fitz  # PyMuPDF
from PIL import Image

load_dotenv()


class AzureConfig:
    """Your Azure OpenAI credentials"""
    AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    API_KEY = os.getenv("OPENAI_API_KEY")
    DEPLOYMENT_NAME = os.getenv("OPENAI_DEPLOYMENT_NAME")
    API_VERSION = os.getenv("OPENAI_API_VERSION")


def pdf_to_images(pdf_path: str, dpi: int = 300):
    """
    Convert all PDF pages to PIL Images using PyMuPDF (NO POPPLER)
    """
    doc = fitz.open(pdf_path)
    images = []

    zoom = dpi / 72  # PDF default DPI = 72
    matrix = fitz.Matrix(zoom, zoom)

    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        pix = page.get_pixmap(matrix=matrix)

        image = Image.frombytes(
            "RGB",
            [pix.width, pix.height],
            pix.samples
        )
        images.append(image)

    return images


def ask_pdf(pdf_path: str, question: str, dpi: int = 300):
    """
    Ask a question about your PDF using GPT-4o Vision
    Returns ONLY the answer text
    """

    # Convert PDF → Images
    images = pdf_to_images(pdf_path, dpi=dpi)

    # Initialize Azure OpenAI
    client = AzureOpenAI(
        api_key=AzureConfig.API_KEY,
        api_version=AzureConfig.API_VERSION,
        azure_endpoint=AzureConfig.AZURE_ENDPOINT
    )

    # Prepare multimodal content
    content = [{"type": "text", "text": question}]

    for image in images:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_base64}",
                "detail": "high"
            }
        })


    PDF_READER_SYSTEM_PROMPT = (
        "You are a strict PDF document reader. "
        "You must answer ONLY using the information visible in the provided document images. "
        "Do NOT add explanations, assumptions, or extra text. "
        "If the answer is not visible in the document, reply with 'Not found in document.' "
        "Follow the user's instructions exactly and keep the response concise."
    )


    # Call GPT-4o Vision
    response = client.chat.completions.create(
        model=AzureConfig.DEPLOYMENT_NAME,
        messages=[{"role": "system", "content": PDF_READER_SYSTEM_PROMPT}, {"role": "user", "content": content}],
        max_tokens=2000,
        temperature=0
    )

    return response.choices[0].message.content


# =============================================================================
# USAGE
# =============================================================================

if __name__ == "__main__":

    pdf_file = "Naac_appLetter.pdf"

    user_prompt = input("Enter your question about the PDF: ")
    answer = ask_pdf(pdf_file, user_prompt)
    print(f"A: {answer}")
