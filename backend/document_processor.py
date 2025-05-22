# Importing necessary libraries for processing PDFs and images
# PyPDF2: Used to read and extract text from PDF files
# PaddleOCR: A library for performing Optical Character Recognition (OCR) to extract text from images
# Path: A module from pathlib to handle file paths easily

import PyPDF2
from paddleocr import PaddleOCR
from pathlib import Path

# Creating a class called DocumentProcessor to handle processing of PDFs and images
class DocumentProcessor:

    # This is the constructor method, which runs when a new DocumentProcessor object is created
    # It initializes an OCR (Optical Character Recognition) tool using PaddleOCR
    # use_angle_cls=True: Enables angle classification for better text orientation detection
    # lang='en': Sets the language to English for OCR
    # use_gpu=True: Uses GPU if available for faster processing
    
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)
    
    # This method processes a PDF file to extract text from each page
    # It takes a file path (string) as input and returns a list of dictionaries containing page numbers and text
    
    def process_pdf(self, file_path: str) -> list:
        pages = []  # Empty list to store page data
        try:
            print(f"Opening PDF: {file_path}")  # Printing the file path for debugging
            with open(file_path, "rb") as file:  # Opening the PDF file in binary read mode
                reader = PyPDF2.PdfReader(file)  # Creating a PDF reader object using PyPDF2
                num_pages = len(reader.pages)  # Getting the total number of pages in the PDF
                print(f"Number of pages detected: {num_pages}")  # Printing the number of pages for debugging
                
                # Looping through each page in the PDF (page numbers start from 0)
                
                for page_num in range(num_pages):
                    page = reader.pages[page_num]  # Getting the page object for the current page number
                    text = page.extract_text() or ""  # Extracting text from the page; if no text, use empty string
                    pages.append({"page": page_num + 1, "text": text})  # Adding page number (1-based) and text to the list
                del reader  # Deleting the reader object to free up memory
                
        except Exception as e:
            print(f"Error processing PDF: {e}")  # Printing any errors that occur during processing
        return pages  # Returning the list of pages with extracted text

    
    # This method processes an image file to extract text using OCR
    # It takes a file path (string) as input and returns a list with a single dictionary containing the text
    
    def process_image(self, file_path: str) -> list:
        try:
            result = self.ocr.ocr(file_path, cls=True)  # Using PaddleOCR to extract text from the image
            text = " ".join([line[1][0] for line in result[0]])  # Combining all extracted text lines into a single string
            return [{"page": 1, "text": text}]  # Returning a list with one entry for the image (page 1)
        except Exception as e:
            print(f"Error with image {file_path}: {e}")  # Printing any errors that occur during OCR
            return [{"page": 1, "text": ""}]  # Returning an empty text entry if OCR fails

    
    # This method decides whether to process the file as a PDF or an image based on its extension
    # It takes a file path (string) as input and returns the processed result as a list
    
    
    def process(self, file_path: str) -> list:
        file_ext = Path(file_path).suffix.lower()  # Getting the file extension (e.g., ".pdf", ".jpg") in lowercase
        if file_ext == ".pdf":  # If the file is a PDF
            return self.process_pdf(file_path)  # Process it as a PDF
        elif file_ext in [".jpg", ".jpeg", ".png"]:  # If the file is an image (JPG, JPEG, or PNG)
            return self.process_image(file_path)  # Process it as an image
        else:
            print(f"Unsupported file type: {file_ext}")  # Printing a message for unsupported file types
            return []  # Returning an empty list for unsupported files
