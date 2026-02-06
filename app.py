import os, io, base64, httpx, asyncio, psutil, time, json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from PIL import Image
from pdf2image import convert_from_bytes
from concurrent.futures import ProcessPoolExecutor

app = FastAPI()
# We use 'moondream' or 'smollm' for sub-1s performance
TARGET_MODEL = "moondream" 
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://ollama-vlm:11434")
executor = ProcessPoolExecutor(max_workers=4) # Parallel PDF ripping
active_connections = set()

@app.on_event("startup")
async def startup_event():
    # Ensure model is pre-loaded in the 128GB RAM permanently
    async with httpx.AsyncClient() as client:
        await client.post(f"{OLLAMA_URL}/api/chat", json={"model": TARGET_MODEL, "keep_alive": -1})
    asyncio.create_task(broadcast_stats())

async def broadcast_stats():
    while True:
        if active_connections:
            vm = psutil.virtual_memory()
            stats = {"cpu": psutil.cpu_percent(), "ram_gb": round(vm.used / (1024**3), 2)}
            for ws in list(active_connections):
                try: await ws.send_json({"type": "stats", "data": stats})
                except: active_connections.discard(ws)
        await asyncio.sleep(1)

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
    await log_to_ws(f"🚀 ULTRA-SPEED ENGINE ACTIVE: {TARGET_MODEL}", "success")
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect: active_connections.discard(websocket)

async def ai_worker(queue, client, prompt, results):
    while True:
        item = await queue.get()
        if item is None: break
        page_num, img_b64 = item
        start_t = time.time()
        
        try:
            # OPTIMIZATION: Reduced context and zero-temp for instant response
            response = await client.post(f"{OLLAMA_URL}/api/chat", json={
                "model": TARGET_MODEL,
                "messages": [{"role": "user", "content": f"JSON: {prompt}", "images": [img_b64]}],
                "stream": False, "format": "json", "keep_alive": -1,
                "options": {"num_thread": 12, "num_ctx": 1024, "temperature": 0}
            }, timeout=30.0)
            
            elapsed = round(time.time() - start_t, 2)
            content = response.json().get("message", {}).get("content", "{}")
            results.append({"page": page_num, "data": json.loads(content), "perf": elapsed})
            await log_to_ws(f"✅ Page {page_num} Processed: {elapsed}s", "success")
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
            
            # OPTIMIZATION: Ultra-fast Low-Res Ripping
            # 100 DPI is enough for these tiny models to process instantly
            pages = convert_from_bytes(content, dpi=100, thread_count=16, strict=False)
            await log_to_ws(f"📂 Ripped {len(pages)} pages in {round(time.time()-start_total, 2)}s")
            
            for i, p in enumerate(pages):
                img = p.convert("RGB")
                img.thumbnail((768, 768)) # Moondream/SmolVLM native resolution
                buf = io.BytesIO(); img.save(buf, format='JPEG', quality=60)
                await queue.put((i + 1, base64.b64encode(buf.getvalue()).decode('utf-8')))
            
            await queue.put(None)
            await asyncio.gather(worker)
            
        total_time = round(time.time() - start_total, 2)
        await log_to_ws(f"🏆 TOTAL SCAN TIME: {total_time}s", "success")
        return {"results": sorted(results, key=lambda x: x['page'])}
    except Exception as e:
        await log_to_ws(f"🔥 Failure: {str(e)}", "error")
        raise HTTPException(status_code=500, detail=str(e))