import fitz  # PyMuPDF


def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extract all text from an uploaded PDF file.
    Works with Streamlit's UploadedFile object.
    """
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    full_text = ""
    for page in doc:
        full_text += page.get_text()

    doc.close()
    return full_text.strip()
