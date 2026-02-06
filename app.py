import os, io, base64, httpx, asyncio, psutil, time, json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket
from fastapi.responses import FileResponse
from PIL import Image
from pdf2image import convert_from_bytes

app = FastAPI()
TARGET_MODEL = "moondream" # Ensure you have run: ollama pull moondream
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://ollama-vlm:11434")
active_connections = set()

@app.on_event("startup")
async def startup():
    # Force model to stay in your 128GB RAM forever
    async with httpx.AsyncClient() as client:
        await client.post(f"{OLLAMA_URL}/api/chat", json={"model": TARGET_MODEL, "keep_alive": -1})

@app.post("/scan")
async def scan(file: UploadFile = File(...), prompt: str = Form(...)):
    start_total = time.time()
    try:
        content = await file.read()
        # Optimized for Moondream: 100 DPI is the sweet spot for speed
        pages = convert_from_bytes(content, dpi=100)
        results = []

        async with httpx.AsyncClient(timeout=60.0) as client:
            for i, p in enumerate(pages):
                page_start = time.time()
                
                # Encode Image
                buf = io.BytesIO()
                p.save(buf, format='JPEG', quality=70)
                img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

                # Request with strict CPU limits
                response = await client.post(f"{OLLAMA_URL}/api/chat", json={
                    "model": TARGET_MODEL,
                    "messages": [{"role": "user", "content": f"Extract JSON: {prompt}", "images": [img_b64]}],
                    "stream": False, "format": "json",
                    "options": {
                        "num_thread": 8, # Use 8 threads to prevent core fighting
                        "num_ctx": 2048,
                        "temperature": 0
                    }
                })
                
                elapsed = round(time.time() - page_start, 2)
                results.append({"page": i+1, "data": response.json().get("message", {}).get("content"), "perf": elapsed})
        
        return {"results": results, "total_time": round(time.time() - start_total, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))