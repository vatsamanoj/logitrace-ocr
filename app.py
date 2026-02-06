import os, io, base64, httpx, asyncio, psutil, time, json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from PIL import Image
from pdf2image import convert_from_bytes

app = FastAPI()
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://ollama-vlm:11434")
active_connections = set()

async def get_hardware_stats():
    """Captures real-time CPU and RAM usage"""
    return {
        "cpu": psutil.cpu_percent(),
        "ram": psutil.virtual_memory().percent,
        "ram_gb": round(psutil.virtual_memory().used / (1024**3), 2)
    }

async def broadcast_stats():
    """Background task to send hardware heartbeat to the UI"""
    while True:
        if active_connections:
            stats = await get_hardware_stats()
            for ws in active_connections:
                try:
                    await ws.send_json({"type": "stats", "data": stats})
                except: pass
        await asyncio.sleep(1)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(broadcast_stats())

    # --- ROUTES ---
@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serves the dashboard and fixes the 404 issue"""
    index_path = os.path.join(os.getcwd(), "index.html")
    return FileResponse(index_path)

async def log_to_ws(msg: str, status="info"):
    payload = {"type": "log", "time": time.strftime("%H:%M:%S"), "msg": msg, "status": status}
    for ws in active_connections:
        try: await ws.send_json(payload)
        except: pass

async def ai_worker(queue, client, prompt, threads, results):
    while True:
        item = await queue.get()
        if item is None: break
        page_num, img_b64 = item
        await log_to_ws(f"🔥 Processing Page {page_num}...")
        
        try:
            # We use a strict prompt to ensure field-wise pretty JSON
            clean_prompt = f"Extract the following fields from this document: {prompt}. Return ONLY a pretty-printed JSON object with keys and values. No extra text."
            
            response = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": "qwen2.5vl:7b",
                    "messages": [{"role": "user", "content": clean_prompt, "images": [img_b64]}],
                    "stream": False, "format": "json", "keep_alive": "30s",
                    "options": {"num_thread": threads, "temperature": 0}
                }, timeout=120.0
            )
            raw_data = response.json().get("message", {}).get("content", "{}")
            results.append({"page": page_num, "data": json.loads(raw_data)})
            await log_to_ws(f"✅ Page {page_num} Complete.", "success")
        except Exception as e:
            await log_to_ws(f"❌ Page {page_num} Error: {str(e)}", "error")
        queue.task_done()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.add(websocket)
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections.remove(websocket)

@app.post("/scan")
async def scan(file: UploadFile = File(...), prompt: str = Form(...)):
    try:
        content = await file.read()
        queue = asyncio.Queue()
        results = []
        # Parallelize: Process 2 pages at once to leverage that 128GB RAM
        async with httpx.AsyncClient() as client:
            workers = [asyncio.create_task(ai_worker(queue, client, prompt, 8, results)) for _ in range(2)]

            if file.content_type == "application/pdf":
                pages = convert_from_bytes(content, thread_count=8)
                for i, page in enumerate(pages):
                    img = page.convert("RGB")
                    img.thumbnail((1600, 1600))
                    buf = io.BytesIO()
                    img.save(buf, format='JPEG', quality=90)
                    await queue.put((i + 1, base64.b64encode(buf.getvalue()).decode('utf-8')))
            
            for _ in range(2): await queue.put(None)
            await asyncio.gather(*workers)
        
        return {"filename": file.filename, "results": sorted(results, key=lambda x: x['page'])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))