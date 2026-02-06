import os, io, base64, httpx, asyncio, psutil, time, json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from PIL import Image
from pdf2image import convert_from_bytes
import pytesseract # High-speed traditional OCR

app = FastAPI()
TARGET_MODEL = "moondream"
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://ollama-vlm:11434")

# Optimization: Pre-load model into RAM Forever
@app.on_event("startup")
async def startup():
    async with httpx.AsyncClient() as client:
        await client.post(f"{OLLAMA_URL}/api/chat", json={"model": TARGET_MODEL, "keep_alive": -1})

@app.get("/")
async def get_index(): return FileResponse("index.html")


@app.post("/scan")
async def scan(file: UploadFile = File(...), prompt: str = Form(...)):
    start_total = time.time()
    try:
        content = await file.read()
        
        # STEP 1: Parallel Ripping (DPI 100 for speed)
        pages = convert_from_bytes(content, dpi=100, thread_count=16)
        results = []

        async with httpx.AsyncClient() as client:
            for i, p in enumerate(pages):
                page_start = time.time()
                
                # STEP 2: Traditional OCR (Fastest way to get text)
                raw_text = pytesseract.image_to_string(p)
                
                # STEP 3: AI "Refining" (AI only handles the string, not the image)
                # This is 10x faster because the AI doesn't have to "see" pixels
                response = await client.post(f"{OLLAMA_URL}/api/chat", json={
                    "model": TARGET_MODEL,
                    "messages": [{"role": "user", "content": f"Convert this OCR text to JSON with fields {prompt}: {raw_text}"}],
                    "stream": False, "format": "json",
                    "options": {"num_thread": 16, "num_ctx": 2048, "temperature": 0}
                }, timeout=15.0)
                
                elapsed = round(time.time() - page_start, 2)
                results.append({"page": i+1, "data": response.json().get("message", {}).get("content"), "perf": elapsed})
        
        return {"results": results, "total_time": round(time.time() - start_total, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))