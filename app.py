import os, io, base64, httpx, asyncio, psutil, time, json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from PIL import Image
from pdf2image import convert_from_bytes
from pathlib import Path

app = FastAPI()
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://ollama-vlm:11434")
METRICS_FILE = Path("ocr_performance.json")
active_connections = set()

# Model Precision Tiers
MODELS = {
    "speed": "qwen2.7b-instruct-q2_K",
    "balanced": "qwen2.7b-instruct-q4_K_M",
    "precision": "qwen2.7b-instruct-q8_0"
}


if not METRICS_FILE.exists():
    METRICS_FILE.write_text(json.dumps({"history": []}))

# --- Hardware Sensing ---
async def broadcast_stats():
    while True:
        if active_connections:
            vm = psutil.virtual_memory()
            stats = {
                "cpu": psutil.cpu_percent(),
                "ram": vm.percent,
                "ram_gb": round(vm.used / (1024**3), 2)
            }
            for ws in list(active_connections):
                try: await ws.send_json({"type": "stats", "data": stats})
                except: active_connections.discard(ws)
        await asyncio.sleep(1)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(broadcast_stats())

# --- Intelligence & Logging ---
def decide_model(mode=None):
    if mode in MODELS: return MODELS[mode]
    try:
        data = json.loads(METRICS_FILE.read_text())
        hist = data.get("history", [])[-5:]
        if not hist: return MODELS["balanced"]
        avg = sum(h['time'] for h in hist) / len(hist)
        return MODELS["speed"] if avg > 40 else MODELS["precision"] if avg < 10 else MODELS["balanced"]
    except: return MODELS["balanced"]

async def log_to_ws(msg: str, status="info"):
    payload = {"type": "log", "time": time.strftime("%H:%M:%S"), "msg": msg, "status": status}
    for ws in list(active_connections):
        try: await ws.send_json(payload)
        except: active_connections.discard(ws)

# --- The AI Worker ---
async def ai_worker(queue, client, prompt, threads, results, mode):
    while True:
        item = await queue.get()
        if item is None: break
        page_num, img_b64 = item
        model = decide_model(mode)
        start_t = time.time()
        
        await log_to_ws(f"🚀 Page {page_num}: Using {model} ({threads} threads)")
        try:
            res = await client.post(f"{OLLAMA_URL}/api/chat", json={
                "model": model,
                "messages": [{"role": "user", "content": f"Extract fields: {prompt}. JSON only.", "images": [img_b64]}],
                "stream": False, "format": "json", "keep_alive": 0,
                "options": {"num_thread": threads, "temperature": 0}
            }, timeout=150.0)
            
            elapsed = round(time.time() - start_t, 2)
            content = res.json().get("message", {}).get("content", "{}")
            results.append({"page": page_num, "data": json.loads(content), "perf": elapsed, "model": model})
            
            # Persist performance logs
            m = json.loads(METRICS_FILE.read_text())
            m["history"].append({"time": elapsed, "model": model})
            METRICS_FILE.write_text(json.dumps(m))
            await log_to_ws(f"✅ Page {page_num} done in {elapsed}s", "success")
        except Exception as e:
            await log_to_ws(f"❌ Page {page_num} Error: {str(e)}", "error")
        queue.task_done()

@app.get("/")
async def get_index():
    return FileResponse("index.html")

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept(); active_connections.add(websocket)
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect: active_connections.discard(websocket)

@app.post("/scan")
async def scan(file: UploadFile = File(...), prompt: str = Form(...), mode: str = Form(None)):
    try:
        content = await file.read()
        queue = asyncio.Queue()
        results = []
        async with httpx.AsyncClient() as client:
            workers = [asyncio.create_task(ai_worker(queue, client, prompt, 8, results, mode)) for _ in range(2)]
            if file.content_type == "application/pdf":
                pages = convert_from_bytes(content, thread_count=8)
                for i, p in enumerate(pages):
                    img = p.convert("RGB")
                    img.thumbnail((1500, 1500))
                    buf = io.BytesIO(); img.save(buf, format='JPEG', quality=85)
                    await queue.put((i + 1, base64.b64encode(buf.getvalue()).decode('utf-8')))
            else:
                img = Image.open(io.BytesIO(content)).convert("RGB")
                img.thumbnail((1500, 1500))
                buf = io.BytesIO(); img.save(buf, format='JPEG', quality=85)
                await queue.put((1, base64.b64encode(buf.getvalue()).decode('utf-8')))
            
            for _ in range(2): await queue.put(None)
            await asyncio.gather(*workers)
        return {"results": sorted(results, key=lambda x: x['page'])}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))