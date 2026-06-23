from langchain.tools import tool
import fitz  # PyMuPDF
import requests
import tempfile
import os
from dotenv import load_dotenv
load_dotenv()
print(os.getenv("LANGSMITH_TRACING"))
print(os.getenv("LANGSMITH_PROJECT"))
print(os.getenv("LANGSMITH_API_KEY")[:10])

@tool
def parse_uploaded_pdf(file_path: str):
    """
    Extract text from a local PDF file.
    """

    doc = fitz.open(file_path)

    text = ""

    for page in doc:
        text += page.get_text()

    doc.close()

    return text


@tool
def parse_paper_pdf(pdf_url: str):
    """
    Download a PDF from a URL and extract its text.
    """

    response = requests.get(pdf_url)

    if response.status_code != 200:
        return f"Failed to download PDF: {response.status_code}"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(response.content)
        temp_path = temp_file.name

    doc = fitz.open(temp_path)

    text = ""

    for page in doc:
        text += page.get_text()

    doc.close()

    return text
if __name__ == "__main__":

    pdf_url = "https://arxiv.org/pdf/1706.03762.pdf"

    result = parse_paper_pdf.invoke({
        "pdf_url": pdf_url
    })

    print("\n----- FIRST 2000 CHARACTERS -----\n")
    print(result[:2000])