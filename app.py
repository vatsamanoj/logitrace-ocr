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
        
        await log_to_ws(f"🚀 Page {page_num}: Analyzing with {model_name}...")
        
        try:
            response = await client.post(f"{OLLAMA_URL}/api/chat", json={
                "model": model_name,
                "messages": [{"role": "user", "content": f"OCR extraction: {prompt}. JSON only.", "images": [img_b64]}],
                "stream": False, "format": "json", "keep_alive": "5m",
                "options": {"num_thread": threads, "temperature": 0}
            }, timeout=180.0)
            
            res_json = response.json()
            content = res_json.get("message", {}).get("content", "").strip()
            elapsed = round(time.time() - start_t, 2)
            
            if content:
                results.append({"page": page_num, "data": json.loads(content), "perf": elapsed})
                await log_to_ws(f"✅ Page {page_num} Complete ({elapsed}s)", "success")
        except Exception as e:
            await log_to_ws(f"❌ Page {page_num} Error: {str(e)}", "error")
        queue.task_done()

@app.post("/scan")
async def scan(file: UploadFile = File(...), prompt: str = Form(...), mode: str = Form(...)):
    try:
        content = await file.read()
        queue = asyncio.Queue(); results = []
        async with httpx.AsyncClient() as client:
            # Using 14 threads to leave headroom for the OS on your 16-core machine
            workers = [asyncio.create_task(ai_worker(queue, client, prompt, 14, mode, results))]
            
            # Robust PDF ripping
            pages = convert_from_bytes(content, dpi=200, thread_count=8, strict=False)
            await log_to_ws(f"📂 Document parsed: {len(pages)} pages.")
            
            for i, p in enumerate(pages):
                img = p.convert("RGB")
                img.thumbnail((1600, 1600))
                buf = io.BytesIO(); img.save(buf, format='JPEG', quality=85)
                await queue.put((i + 1, base64.b64encode(buf.getvalue()).decode('utf-8')))
            
            await queue.put(None)
            await asyncio.gather(*workers)
            
        return {"results": sorted(results, key=lambda x: x['page'])}
    except Exception as e:
        await log_to_ws(f"🔥 Failure: {str(e)}", "error")
        raise HTTPException(status_code=500, detail=str(e))