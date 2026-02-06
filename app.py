import os, io, base64, httpx, asyncio, psutil, time, json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from PIL import Image
from pdf2image import convert_from_bytes

app = FastAPI()
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://ollama-vlm:11434")
active_connections = set()

# Map the exact tags from your system
PRECISION_MODELS = {
    "speed": "qwen2.5vl:7b-q4_K_M",
    "high": "qwen2.5vl:7b-q8_0"
}

async def broadcast_stats():
    while True:
        if active_connections:
            vm = psutil.virtual_memory()
            stats = {"cpu": psutil.cpu_percent(), "ram_gb": round(vm.used / (1024**3), 2)}
            for ws in list(active_connections):
                try: await ws.send_json({"type": "stats", "data": stats})
                except: active_connections.discard(ws)
        await asyncio.sleep(1)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(broadcast_stats())

async def log_to_ws(msg: str, status="info"):
    payload = {"type": "log", "time": time.strftime("%H:%M:%S"), "msg": msg, "status": status}
    for ws in list(active_connections):
        try: await ws.send_json(payload)
        except: active_connections.discard(ws)

@app.get("/", response_class=HTMLResponse)
async def get_index():
    return FileResponse("index.html")

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept(); active_connections.add(websocket)
    await log_to_ws("System Active. Select Precision Mode to begin.", "success")
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect: active_connections.discard(websocket)

async def ai_worker(queue, client, prompt, threads, model_alias, results):
    model_name = PRECISION_MODELS.get(model_alias, PRECISION_MODELS["speed"])
    while True:
        item = await queue.get()
        if item is None: break
        page_num, img_b64 = item
        start_t = time.time()
        
        await log_to_ws(f"🚀 Page {page_num}: Fast-Processing with {model_name}...")
        
        try:
            response = await client.post(f"{OLLAMA_URL}/api/chat", json={
                "model": model_name,
                "messages": [{"role": "user", "content": f"OCR extraction: {prompt}. JSON only.", "images": [img_b64]}],
                "stream": False, 
                "format": "json", 
                "keep_alive": "1m", # RECLAIM RAM FASTER
                "options": {
                    "num_thread": 8,        # FIX: Lower threads to prevent CPU thrashing
                    "num_ctx": 4096,        # FIX: Limit context window to speed up inference
                    "temperature": 0,
                    "num_predict": 1024
                }
            }, timeout=120.0) # Lower timeout to prevent infinite hangs
            
            res_json = response.json()
            content = res_json.get("message", {}).get("content", "").strip()
            elapsed = round(time.time() - start_t, 2)
            
            if content:
                results.append({"page": page_num, "data": json.loads(content), "perf": elapsed})
                await log_to_ws(f"✅ Page {page_num} Complete in {elapsed}s", "success")
            else:
                await log_to_ws(f"⚠️ Page {page_num}: Model gave empty response.", "error")
        except Exception as e:
            await log_to_ws(f"❌ Page {page_num} HANGED: {str(e)}", "error")
        queue.task_done()

@app.post("/scan")
async def scan(file: UploadFile = File(...), prompt: str = Form(...), mode: str = Form(...)):
    try:
        content = await file.read()
        queue = asyncio.Queue(); results = []
        async with httpx.AsyncClient() as client:
            # ONLY 1 WORKER: Running 2 pages at once on CPU causes the 84% "doing nothing" freeze.
            workers = [asyncio.create_task(ai_worker(queue, client, prompt, 8, mode, results))]
            
            # OPTIMIZED IMAGE SIZE: 1200px is enough for Qwen 7B to read shipping docs.
            pages = convert_from_bytes(content, dpi=150, thread_count=8, strict=False)
            await log_to_ws(f"📂 PDF Parsed. Starting Sequential Inference...")
            
            for i, p in enumerate(pages):
                img = p.convert("RGB")
                img.thumbnail((1200, 1200)) # FIX: Smaller resolution = 4x faster CPU processing
                buf = io.BytesIO(); img.save(buf, format='JPEG', quality=75)
                await queue.put((i + 1, base64.b64encode(buf.getvalue()).decode('utf-8')))
            
            await queue.put(None)
            await asyncio.gather(*workers)
            
        return {"results": sorted(results, key=lambda x: x['page'])}