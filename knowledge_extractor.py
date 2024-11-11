import PyPDF2

def extract_text_from_pdf(pdf_path):
    text_chunks = []
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + " "
        
        words = text.split()
        chunk_size = 200  
        overlap = 100     
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk.strip()) > 0:
                text_chunks.append(chunk)
    return text_chunks