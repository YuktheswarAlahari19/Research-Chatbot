import PyPDF2
from paddleocr import PaddleOCR
from pathlib import Path

class DocumentProcessor:

    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)
    
    def process_pdf(self, file_path: str) -> list:
        pages = []
        try:
            print(f"Opening PDF: {file_path}")
            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                num_pages = len(reader.pages)
                print(f"Number of pages detected: {num_pages}")
                
                for page_num in range(num_pages):
                    page = reader.pages[page_num]
                    text = page.extract_text() or ""
                    pages.append({"page": page_num + 1, "text": text})
                del reader 
        except Exception as e:
            print(f"Error processing PDF: {e}")
        return pages
        
    def process_image(self, file_path: str) -> list:
        try:
            result = self.ocr.ocr(file_path, cls=True)
            text = " ".join([line[1][0] for line in result[0]])
            return [{"page": 1, "text": text}]
        except Exception as e:
            print(f"Error with image {file_path}: {e}")
            return [{"page": 1, "text": ""}]
    
    def process(self, file_path: str) -> list:
        file_ext = Path(file_path).suffix.lower()
        if file_ext == ".pdf":
            return self.process_pdf(file_path)
        elif file_ext in [".jpg", ".jpeg", ".png"]:
            return self.process_image(file_path)
        else:
            print(f"Unsupported file type: {file_ext}")
            return []