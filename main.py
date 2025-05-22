# Importing libraries needed for building a web application with FastAPI
# FastAPI: Main library for creating the web application and handling HTTP requests
# UploadFile, File: Classes for handling file uploads in FastAPI
# BackgroundTasks: Class for running tasks in the background (e.g., file processing)
# HTTPException: Class for raising HTTP errors (e.g., 404 Not Found)
# Request: Class for handling HTTP requests in FastAPI
# Jinja2Templates: Library for rendering HTML templates with dynamic data
# HTMLResponse, RedirectResponse: Classes for sending HTML responses and redirects
# StaticFiles: Class for serving static files like images or icons
# os, Path, shutil, uuid, time, sys: Standard Python libraries for file handling, paths, and more


from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import os
from pathlib import Path
import shutil
import uuid
import time
import sys


# Defining a function to add the project root directory to the Python path
# This ensures that modules from the project can be imported correctly


def add_project_root_to_path():
    # Getting the absolute path of the project root (parent directory of this file)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # Checking if the project root is already in the Python path
    if project_root not in sys.path:
        # Adding the project root to the Python path if it's not already there
        sys.path.insert(0, project_root)


# Calling the function to add the project root to the Python path
# This runs when the script starts to ensure proper module imports


add_project_root_to_path()


# Importing backend modules for document processing, vector storage, query processing, and theme identification
# These modules handle the core functionality of the application (e.g., extracting text, searching, answering queries)


from chatbot_theme_identifier.backend.document_processor import DocumentProcessor
from chatbot_theme_identifier.backend.vector_store import VectorStore
from chatbot_theme_identifier.backend.query_processor import QueryProcessor
from chatbot_theme_identifier.backend.theme_identifier import ThemeIdentifier


# Creating a FastAPI application instance to handle HTTP requests
app = FastAPI()
# Setting up Jinja2 templates for rendering HTML pages, using the "templates" directory
templates = Jinja2Templates(directory="templates")
# Mounting the "static" directory to serve static files (e.g., favicon.ico) at the "/static" URL path
app.mount("/static", StaticFiles(directory="static"), name="static")


# Initializing backend classes to handle document processing, vector storage, query processing, and theme identification
# These objects will be used to process files and queries throughout the application

doc_processor = DocumentProcessor()
vector_store = VectorStore()
query_processor = QueryProcessor(vector_store)
theme_identifier = ThemeIdentifier(query_processor)


# Setting up storage for uploaded files and trackers for progress, results, and documents
# UPLOAD_DIR: Directory where uploaded files will be saved
# progress_tracker: Dictionary to track the progress of tasks (e.g., upload, query processing)
# results_tracker: Dictionary to store the results of query processing
# documents_tracker: Dictionary to store information about processed documents


UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)  # Creating the uploads directory if it doesn't exist
progress_tracker = {}
results_tracker = {}
documents_tracker = {}


# Defining a route to handle requests for the favicon (icon shown in browser tab)
# This redirects requests for "/favicon.ico" to "/static/favicon.ico"


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return RedirectResponse(url="/static/favicon.ico")


# Defining a route to handle GET requests to the root URL ("/")
# This renders the main HTML page (index.html) using the Jinja2 template


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Defining a route to handle POST requests for uploading files
# This endpoint accepts a list of files and processes them in the background


@app.post("/upload")
async def upload_files(background_tasks: BackgroundTasks, files: list[UploadFile] = File(...)):
    print("DEBUG: Entering /upload endpoint")  # Printing a debug message to show the endpoint is called
    print("DEBUG: Files received:", files)  # Printing the list of files received
    task_id = str(uuid.uuid4())  # Generating a unique task ID for this upload
    
    
    # Initializing the progress tracker for this task with a starting status and progress
    progress_tracker[task_id] = {"status": "Uploading files...", "progress": 0}
    results_tracker[task_id] = None  # Initializing the results tracker for this task as None
    documents_tracker[task_id] = None  # Initializing the documents tracker for this task as None

    saved_files = []  # List to store the paths of saved files
    
    
    # Looping through each uploaded file to save it to the uploads directory
    for file in files:
        print("DEBUG: Processing file:", file)  # Printing the file being processed
        print("DEBUG: File filename type:", type(file.filename))  # Printing the type of the filename
        print("DEBUG: File filename value:", file.filename)  # Printing the filename
        # Ensuring the filename is a string and checking if it's a PDF
        filename = file.filename if isinstance(file.filename, str) else ""
        if not filename or not filename.lower().endswith(".pdf"):
            # Raising an error if the file is not a PDF
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        
        # Creating a unique file path for the uploaded file in the uploads directory
        file_path = UPLOAD_DIR / f"{uuid.uuid4()}_{filename}"
        # Saving the file to the specified path
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(file_path)  # Adding the saved file path to the list
        print("DEBUG: Saved file path:", file_path)  # Printing the saved file path

    # Adding a background task to process the documents (runs separately from the main request)
    background_tasks.add_task(process_documents, task_id, saved_files)
    
    # Returning a success response with the task ID
    return {"status": "success", "task_id": task_id}


# Defining an async function to process documents in the background
# It takes a task ID and a list of file paths as input

async def process_documents(task_id: str, file_paths: list[Path]):
    print("DEBUG: Entering process_documents with task_id:", task_id)  # Printing the task ID
    print("DEBUG: File paths:", file_paths)  # Printing the file paths
    
    
    try:
        doc_id_to_path = {}  # Dictionary to map document IDs to file paths
        # Updating the progress tracker to show that OCR processing is starting
        progress_tracker[task_id] = {"status": "Processing documents with OCR...", "progress": 10}
        documents = []  # List to store processed documents
        # Looping through each file path to process the document
        
        
        for file_path in file_paths:
            print("DEBUG: Processing file path:", file_path)  # Printing the file path being processed
            # Processing the document using the document processor (e.g., extracting text)
            pages = doc_processor.process(str(file_path))
            print("DEBUG: Pages extracted:", pages)  # Printing the extracted pages
            # Combining the text from all pages into a single string
            text = " ".join(page["text"] for page in pages if page["text"])
            # Adding the document details to the list
            documents.append({"path": file_path, "text": text, "pages": pages})
            time.sleep(1)  # Simulating slow processing with a 1-second delay

        # Updating the progress tracker to show that vector store building is starting
        progress_tracker[task_id] = {"status": "Building vector store...", "progress": 40}
        
        
        # Looping through each document to add it to the vector store
        for doc in documents:
            doc_id = str(uuid.uuid4())  # Generating a unique document ID
            doc_id_to_path[doc_id] = str(doc["path"])  # Mapping the document ID to its file path
            print("DEBUG: Adding document to vector store, doc_id:", doc_id)  # Printing the document ID
            # Adding the document to the vector store
            vector_store.adding_documents(doc_id, doc["pages"])
        time.sleep(1)  # Simulating slow processing with a 1-second delay

        # Storing the processed documents and their mapping in the documents tracker
        documents_tracker[task_id] = {"documents": documents, "doc_id_to_path": doc_id_to_path}
        
        # Updating the progress tracker to show that processing is complete
        progress_tracker[task_id] = {"status": "Document processing complete", "progress": 100}
        
        
    except Exception as e:
        # Printing any errors that occur during document processing
        print("DEBUG: Error in process_documents:", str(e))
        # Updating the progress tracker with the error status
        progress_tracker[task_id] = {"status": f"Error: {str(e)}", "progress": -1}


# Defining a route to handle POST requests for processing queries
# This endpoint accepts a dictionary with a task ID and query, and processes the query in the background


@app.post("/process_query")
async def process_query(background_tasks: BackgroundTasks, data: dict):
    print("DEBUG: Entering /process_query with data:", data)  # Printing the received data
    task_id = data.get("task_id")  # Getting the task ID from the request data
    query = data.get("query")  # Getting the query from the request data
    # Checking if the task ID and query are provided
    
    
    if not task_id or not query:
        # Raising an error if either is missing
        raise HTTPException(status_code=400, detail="Task ID and query are required")
    # Checking if the task ID exists in the documents tracker
    if task_id not in documents_tracker:
        # Raising an error if the task ID is not found
        raise HTTPException(status_code=404, detail="Task not found")
    
    
    query_task_id = str(uuid.uuid4())  # Generating a unique task ID for this query
    
    
    # Initializing the progress tracker for this query task
    progress_tracker[query_task_id] = {"status": "Starting query processing...", "progress": 0}
    results_tracker[query_task_id] = None  # Initializing the results tracker for this query task
    
    
    # Adding a background task to process the query
    background_tasks.add_task(process_query_task, query_task_id, task_id, query)
    # Returning a success response with the query task ID
    return {"status": "success", "task_id": query_task_id}


# Defining an async function to process a query in the background
# It takes a query task ID, the original task ID, and the query as input


async def process_query_task(query_task_id: str, original_task_id: str, query: str):
    print("DEBUG: Processing query_task_id:", query_task_id, "Query:", query)  # Printing the query task ID and query
    
    
    try:
        # Updating the progress tracker to show that theme identification is starting
        progress_tracker[query_task_id] = {"status": "Identifying themes...", "progress": 50}
        # Identifying themes for the query using the theme identifier
        themes = theme_identifier.identify_themes(query)
        print("DEBUG: Themes identified:", themes)  # Printing the identified themes
        # Updating the progress tracker to show that processing is complete
        progress_tracker[query_task_id] = {"status": "Processing complete", "progress": 100}
        # Storing the results (themes) in the results tracker
        results_tracker[query_task_id] = themes
        
        
    except Exception as e:
        # Printing any errors that occur during query processing
        print("DEBUG: Error in process_query_task:", str(e))
        # Updating the progress tracker with the error status
        progress_tracker[query_task_id] = {"status": f"Error: {str(e)}", "progress": -1}


# Defining a route to handle GET requests for checking the progress of a task
# It takes a task ID as input and returns the current progress


@app.get("/progress/{task_id}")
async def get_progress(task_id: str):
    # Checking if the task ID exists in the progress tracker
    if task_id not in progress_tracker:
        # Raising an error if the task ID is not found
        raise HTTPException(status_code=404, detail="Task not found")
    # Returning the current progress for the task
    return progress_tracker[task_id]


# Defining a route to handle GET requests for retrieving the results of a task
# It takes a task ID as input and returns the results


@app.get("/results/{task_id}")
async def get_results(task_id: str):
    # Checking if the task ID exists in the results tracker and has results
    if task_id not in results_tracker or results_tracker[task_id] is None:
        # Raising an error if the results are not ready
        raise HTTPException(status_code=425, detail="Results not ready yet")
    print("DEBUG: Returning results for task_id:", task_id, "Data:", results_tracker[task_id])  # Printing the results
    # Returning the results for the task
    return results_tracker[task_id]


# Defining a function to clean up the uploads directory when the application shuts down
# This is triggered by the shutdown event in FastAPI


@app.on_event("shutdown")
async def cleanup():
    # Checking if the uploads directory exists
    if UPLOAD_DIR.exists():
        # Deleting the uploads directory and its contents
        shutil.rmtree(UPLOAD_DIR)


# Checking if this script is being run directly (not imported as a module)
# If so, starting the FastAPI application using Uvicorn


if __name__ == "__main__":
    import uvicorn  # Importing Uvicorn to run the FastAPI application
    # Running the application on host "0.0.0.0" (accessible from any IP) and port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
