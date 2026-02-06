import os, io, base64, httpx, asyncio, psutil, time, json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse # Added FileResponse here
from PIL import Image
from pdf2image import convert_from_bytes

app = FastAPI()
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://ollama-vlm:11434")
active_connections = set()

# --- HARDWARE SENSING & BROADCAST ---
async def broadcast_stats():
    """Senses hardware load and broadcasts to browser console every second"""
    while True:
        if active_connections:
            # Senses actual CPU and RAM usage on your 16-core VPS
            vm = psutil.virtual_memory()
            stats = {
                "cpu": psutil.cpu_percent(interval=None),
                "ram": vm.percent,
                "ram_gb": round(vm.used / (1024**3), 2),
                "ram_total": round(vm.total / (1024**3), 1)
            }
            for ws in list(active_connections):
                try:
                    await ws.send_json({"type": "stats", "data": stats})
                except:
                    active_connections.discard(ws)
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

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections.discard(websocket)

async def log_to_ws(msg: str, status="info"):
    payload = {"type": "log", "time": time.strftime("%H:%M:%S"), "msg": msg, "status": status}
    for ws in list(active_connections):
        try: await ws.send_json(payload)
        except: pass

# --- THE INTELLIGENT WORKER ---
async def ai_worker(queue, client, prompt, threads, results):
    """Processes input chunks and reclaims resources immediately after"""
    while True:
        item = await queue.get()
        if item is None: break
        
        page_num, img_b64 = item
        await log_to_ws(f"🚀 AI analyzing Page {page_num} using {threads} threads...")
        
        try:
            # Force pretty field-wise JSON
            enriched_prompt = f"Field-wise extraction: {prompt}. Return ONLY clean JSON."
            
            response = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": "qwen2.5vl:7b",
                    "messages": [{"role": "user", "content": enriched_prompt, "images": [img_b64]}],
                    "stream": False, 
                    "format": "json", 
                    "keep_alive": 0, # RECLAIM RAM INSTANTLY
                    "options": {"num_thread": threads, "temperature": 0}
                }, 
                timeout=180.0
            )
            
            content = response.json().get("message", {}).get("content", "{}")
            # Parse string to JSON object so it's pretty in the browser
            results.append({"page": page_num, "data": json.loads(content)})
            await log_to_ws(f"✅ Page {page_num} processing complete.", "success")
            
        except Exception as e:
            await log_to_ws(f"❌ Error on Page {page_num}: {str(e)}", "error")
        
        queue.task_done()

@app.post("/scan")
async def scan(file: UploadFile = File(...), prompt: str = Form(...)):
    num_workers = 2 # Process 2 pages at a time using your 128GB RAM
    threads_per_worker = 8 # 8 threads per worker uses 16 cores total

    try:
        content = await file.read()
        queue = asyncio.Queue()
        results = []

        async with httpx.AsyncClient() as client:
            workers = [asyncio.create_task(ai_worker(queue, client, prompt, threads_per_worker, results)) for _ in range(num_workers)]

            if file.content_type == "application/pdf":
                await log_to_ws("📂 PDF detected. Breaking into chunks...")
                pages = convert_from_bytes(content, thread_count=8) 
                for i, page in enumerate(pages):
                    img = page.convert("RGB")
                    img.thumbnail((1600, 1600))
                    buf = io.BytesIO()
                    img.save(buf, format='JPEG', quality=90)
                    await queue.put((i + 1, base64.b64encode(buf.getvalue()).decode('utf-8')))
            else:
                img = Image.open(io.BytesIO(content)).convert("RGB")
                img.thumbnail((1600, 1600))
                buf = io.BytesIO()
                img.save(buf, format='JPEG', quality=90)
                await queue.put((1, base64.b64encode(buf.getvalue()).decode('utf-8')))

            for _ in range(num_workers): await queue.put(None)
            await asyncio.gather(*workers)

        return {"results": sorted(results, key=lambda x: x['page'])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))