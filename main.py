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

# Ensure project root is on path
def add_project_root_to_path():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

add_project_root_to_path()

from chatbot_theme_identifier.backend.document_processor import DocumentProcessor
from chatbot_theme_identifier.backend.vector_store import VectorStore
from chatbot_theme_identifier.backend.query_processor import QueryProcessor
from chatbot_theme_identifier.backend.theme_identifier import ThemeIdentifier

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize backend classes
doc_processor = DocumentProcessor()
vector_store = VectorStore()
query_processor = QueryProcessor(vector_store)
theme_identifier = ThemeIdentifier(query_processor)

# Storage directories and trackers
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
progress_tracker = {}
results_tracker = {}
documents_tracker = {}

# Redirect /favicon.ico to /static/favicon.ico
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return RedirectResponse(url="/static/favicon.ico")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_files(background_tasks: BackgroundTasks, files: list[UploadFile] = File(...)):
    print("DEBUG: Entering /upload endpoint")
    print("DEBUG: Files received:", files)
    task_id = str(uuid.uuid4())
    progress_tracker[task_id] = {"status": "Uploading files...", "progress": 0}
    results_tracker[task_id] = None
    documents_tracker[task_id] = None

    saved_files = []
    for file in files:
        print("DEBUG: Processing file:", file)
        print("DEBUG: File filename type:", type(file.filename))
        print("DEBUG: File filename value:", file.filename)
        filename = file.filename if isinstance(file.filename, str) else ""
        if not filename or not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        file_path = UPLOAD_DIR / f"{uuid.uuid4()}_{filename}"
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(file_path)
        print("DEBUG: Saved file path:", file_path)

    background_tasks.add_task(process_documents, task_id, saved_files)
    return {"status": "success", "task_id": task_id}

async def process_documents(task_id: str, file_paths: list[Path]):
    print("DEBUG: Entering process_documents with task_id:", task_id)
    print("DEBUG: File paths:", file_paths)
    try:
        doc_id_to_path = {}
        progress_tracker[task_id] = {"status": "Processing documents with OCR...", "progress": 10}
        documents = []
        for file_path in file_paths:
            print("DEBUG: Processing file path:", file_path)
            pages = doc_processor.process(str(file_path))
            print("DEBUG: Pages extracted:", pages)
            text = " ".join(page["text"] for page in pages if page["text"])
            documents.append({"path": file_path, "text": text, "pages": pages})
            time.sleep(1)  # Simulate slow processing

        progress_tracker[task_id] = {"status": "Building vector store...", "progress": 40}
        for doc in documents:
            doc_id = str(uuid.uuid4())
            doc_id_to_path[doc_id] = str(doc["path"])
            print("DEBUG: Adding document to vector store, doc_id:", doc_id)
            vector_store.adding_documents(doc_id, doc["pages"])
        time.sleep(1)  # Simulate slow processing

        documents_tracker[task_id] = {"documents": documents, "doc_id_to_path": doc_id_to_path}
        progress_tracker[task_id] = {"status": "Document processing complete", "progress": 100}
    except Exception as e:
        print("DEBUG: Error in process_documents:", str(e))
        progress_tracker[task_id] = {"status": f"Error: {str(e)}", "progress": -1}

@app.post("/process_query")
async def process_query(background_tasks: BackgroundTasks, data: dict):
    print("DEBUG: Entering /process_query with data:", data)
    task_id = data.get("task_id")
    query = data.get("query")
    if not task_id or not query:
        raise HTTPException(status_code=400, detail="Task ID and query are required")
    if task_id not in documents_tracker:
        raise HTTPException(status_code=404, detail="Task not found")
    query_task_id = str(uuid.uuid4())
    progress_tracker[query_task_id] = {"status": "Starting query processing...", "progress": 0}
    results_tracker[query_task_id] = None
    background_tasks.add_task(process_query_task, query_task_id, task_id, query)
    return {"status": "success", "task_id": query_task_id}

async def process_query_task(query_task_id: str, original_task_id: str, query: str):
    print("DEBUG: Processing query_task_id:", query_task_id, "Query:", query)
    try:
        progress_tracker[query_task_id] = {"status": "Identifying themes...", "progress": 50}
        themes = theme_identifier.identify_themes(query)
        print("DEBUG: Themes identified:", themes)
        progress_tracker[query_task_id] = {"status": "Processing complete", "progress": 100}
        results_tracker[query_task_id] = themes
    except Exception as e:
        print("DEBUG: Error in process_query_task:", str(e))
        progress_tracker[query_task_id] = {"status": f"Error: {str(e)}", "progress": -1}

@app.get("/progress/{task_id}")
async def get_progress(task_id: str):
    if task_id not in progress_tracker:
        raise HTTPException(status_code=404, detail="Task not found")
    return progress_tracker[task_id]

@app.get("/results/{task_id}")
async def get_results(task_id: str):
    if task_id not in results_tracker or results_tracker[task_id] is None:
        raise HTTPException(status_code=425, detail="Results not ready yet")
    print("DEBUG: Returning results for task_id:", task_id, "Data:", results_tracker[task_id])
    return results_tracker[task_id]

@app.on_event("shutdown")
async def cleanup():
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)