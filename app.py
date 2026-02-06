import os, io, base64, httpx, asyncio, psutil, time, json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from PIL import Image
from pdf2image import convert_from_bytes

app = FastAPI()
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://ollama-vlm:11434")
TARGET_MODEL = "qwen2.5vl:7b"
active_connections = set()

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

async def log_to_ws(msg: str, status="info", data=None):
    payload = {"type": "log", "time": time.strftime("%H:%M:%S"), "msg": msg, "status": status, "data": data}
    for ws in list(active_connections):
        try: await ws.send_json(payload)
        except: pass

@app.get("/", response_class=HTMLResponse)
async def get_index():
    return FileResponse("index.html")

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept(); active_connections.add(websocket)
    await log_to_ws(f"System Ready. Target Model: {TARGET_MODEL}", "success")
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect: active_connections.discard(websocket)

async def ai_worker(queue, client, prompt, threads, results):
    while True:
        item = await queue.get()
        if item is None: break
        page_num, img_b64 = item
        start_t = time.time()
        
        await log_to_ws(f"🚀 Page {page_num}: Processing with {TARGET_MODEL}...")
        
        try:
            # Optimized prompt for Qwen2.5-VL logic
            response = await client.post(f"{OLLAMA_URL}/api/chat", json={
                "model": TARGET_MODEL,
                "messages": [{"role": "user", "content": f"Extract these fields from the document: {prompt}. Return as JSON.", "images": [img_b64]}],
                "stream": False, "format": "json", "keep_alive": "5m",
                "options": {"num_thread": threads, "temperature": 0}
            }, timeout=300.0)
            
            res_json = response.json()
            content = res_json.get("message", {}).get("content", "").strip()
            elapsed = round(time.time() - start_t, 2)
            
            if content:
                results.append({"page": page_num, "data": json.loads(content), "perf": elapsed})
                await log_to_ws(f"✅ Page {page_num} Done ({elapsed}s)", "success")
            else:
                await log_to_ws(f"⚠️ Page {page_num} returned empty result.", "error")
        except Exception as e:
            await log_to_ws(f"❌ Page {page_num} Error: {str(e)}", "error")
        queue.task_done()

@app.post("/scan")
async def scan(file: UploadFile = File(...), prompt: str = Form(...)):
    try:
        content = await file.read()
        queue = asyncio.Queue(); results = []
        async with httpx.AsyncClient() as client:
            # Parallelize: 2 workers using 8 threads each = 16 cores used
            workers = [asyncio.create_task(ai_worker(queue, client, prompt, 8, results)) for _ in range(2)]
            
            # High-DPI ripping for Qwen-VL precision
            pages = convert_from_bytes(content, dpi=250, thread_count=8)
            await log_to_ws(f"📂 Document parsed: {len(pages)} pages.")
            
            for i, p in enumerate(pages):
                img = p.convert("RGB")
                img.thumbnail((1800, 1800)) # Maximum effective res for Qwen 7B
                buf = io.BytesIO(); img.save(buf, format='JPEG', quality=90)
                await queue.put((i + 1, base64.b64encode(buf.getvalue()).decode('utf-8')))
            
            for _ in range(2): await queue.put(None)
            await asyncio.gather(*workers)
            
        return {"results": sorted(results, key=lambda x: x['page'])}
    except Exception as e:
        await log_to_ws(f"🔥 System Failure: {str(e)}", "error")
        raise HTTPException(status_code=500, detail=str(e))