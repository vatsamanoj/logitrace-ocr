import os, io, base64, httpx, asyncio, psutil, time, json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from PIL import Image
from pdf2image import convert_from_bytes

app = FastAPI()
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://ollama-vlm:11434")
TARGET_MODEL = "moondream" # Pre-pulled model
active_connections = set()

# Broadcaster for the 128GB RAM / 16-Core HUD
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
    # Lock model into 128GB RAM permanently to avoid reload delays
    async with httpx.AsyncClient() as client:
        await client.post(f"{OLLAMA_URL}/api/chat", json={"model": TARGET_MODEL, "keep_alive": -1})
    asyncio.create_task(broadcast_stats())

async def log_to_ws(msg: str, status="info"):
    payload = {"type": "log", "time": time.strftime("%H:%M:%S"), "msg": msg, "status": status}
    for ws in list(active_connections):
        try: await ws.send_json(payload)
        except: pass

@app.get("/")
async def get_index(): return FileResponse("index.html")

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept(); active_connections.add(websocket)
    await log_to_ws(f"🚀 High-Speed Engine Active: {TARGET_MODEL}", "success")
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect: active_connections.discard(websocket)

# Worker optimized for CPU inference
async def ai_worker(queue, client, prompt, results):
    while True:
        item = await queue.get()
        if item is None: break
        page_num, img_b64 = item
        start_t = time.time()
        
        try:
            # Using 8 threads prevents context-switching lag on 16 cores
            response = await client.post(f"{OLLAMA_URL}/api/chat", json={
                "model": TARGET_MODEL,
                "messages": [{"role": "user", "content": f"Extract JSON: {prompt}", "images": [img_b64]}],
                "stream": False, "format": "json", "keep_alive": -1,
                "options": {"num_thread": 8, "num_ctx": 2048, "temperature": 0}
            }, timeout=30.0)
            
            elapsed = round(time.time() - start_t, 2)
            content = response.json().get("message", {}).get("content", "{}")
            results.append({"page": page_num, "data": json.loads(content), "perf": elapsed})
            await log_to_ws(f"✅ Page {page_num} Done: {elapsed}s", "success")
        except Exception as e:
            await log_to_ws(f"❌ Page {page_num} Error: {str(e)}", "error")
        queue.task_done()

@app.post("/scan")
async def scan(file: UploadFile = File(...), prompt: str = Form(...)):
    start_total = time.time()
    try:
        content = await file.read()
        queue = asyncio.Queue(); results = []
        
        async with httpx.AsyncClient() as client:
            worker = asyncio.create_task(ai_worker(queue, client, prompt, results))
            
            # 100 DPI is the 'Speed Sweet Spot' for Moondream
            pages = convert_from_bytes(content, dpi=100, thread_count=16)
            await log_to_ws(f"📂 PDF Ripped: {len(pages)} pages.")
            
            for i, p in enumerate(pages):
                img = p.convert("RGB")
                img.thumbnail((768, 768)) # Matches Moondream native res for instant processing
                buf = io.BytesIO(); img.save(buf, format='JPEG', quality=65)
                await queue.put((i + 1, base64.b64encode(buf.getvalue()).decode('utf-8')))
            
            await queue.put(None)
            await asyncio.gather(worker)
            
        total_time = round(time.time() - start_total, 2)
        await log_to_ws(f"🏆 TOTAL SCAN TIME: {total_time}s", "success")
        return {"results": sorted(results, key=lambda x: x['page'])}
    except Exception as e:
        await log_to_ws(f"🔥 Failure: {str(e)}", "error")
        raise HTTPException(status_code=500, detail=str(e))