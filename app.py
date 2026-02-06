import os, io, base64, httpx, asyncio, psutil, time, json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from PIL import Image
from pdf2image import convert_from_bytes

app = FastAPI()
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://ollama-vlm:11434")
active_connections = set()

# --- AUTO-DISCOVERY: SENSE VISION MODELS ---
async def get_vision_models():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            res = await client.get(f"{OLLAMA_URL}/api/tags")
            if res.status_code != 200: return []
            all_models = res.json().get('models', [])
            vision_models = []
            for m in all_models:
                name = m['name']
                details = m.get('details', {})
                families = details.get('families', []) or []
                # Sense for vision capabilities
                if 'vision' in families or 'mllm' in families or '-vl' in name.lower():
                    vision_models.append({
                        "name": name,
                        "size": f"{round(m['size']/(1024**3), 1)}GB"
                    })
            return vision_models
    except: return []

# --- HARDWARE HUD BROADCASTER ---
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

# --- HELPER: REAL-TIME LOGGING ---
async def log_to_ws(msg: str, status="info", data=None):
    payload = {"type": "log", "time": time.strftime("%H:%M:%S"), "msg": msg, "status": status, "data": data}
    for ws in list(active_connections):
        try: await ws.send_json(payload)
        except: pass

# --- ROUTES: FRONTEND ---
@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_path = os.path.join(os.getcwd(), "index.html")
    return FileResponse(index_path)

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept(); active_connections.add(websocket)
    models = await get_vision_models()
    await log_to_ws(f"System Online. Sensed {len(models)} Vision Models.", "success", {"models": models})
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect: active_connections.discard(websocket)

# --- MODEL PULLER (BACKGROUND TASK) ---
@app.post("/pull")
async def pull_model(model_name: str = Form(...)):
    async def pull_task():
        async with httpx.AsyncClient(timeout=None) as client:
            await log_to_ws(f"📥 Starting pull for {model_name}...", "info")
            try:
                async with client.stream("POST", f"{OLLAMA_URL}/api/pull", json={"name": model_name}) as response:
                    async for line in response.aiter_lines():
                        if line:
                            data = json.loads(line)
                            status = data.get("status", "")
                            if "downloading" in status.lower():
                                await log_to_ws(f"📥 {model_name}: {status}", "info")
                            elif status == "success":
                                await log_to_ws(f"✅ {model_name} Pulled!", "success")
                                models = await get_vision_models()
                                await log_to_ws("Refreshing Models...", "info", {"models": models})
            except Exception as e:
                await log_to_ws(f"❌ Pull Failed: {str(e)}", "error")
    
    asyncio.create_task(pull_task())
    return {"message": "Started"}

# --- AI WORKER ---
async def ai_worker(queue, client, prompt, threads, results, mode):
    while True:
        item = await queue.get()
        if item is None: break
        page_num, img_b64 = item
        start_t = time.time()
        try:
            response = await client.post(f"{OLLAMA_URL}/api/chat", json={
                "model": mode,
                "messages": [{"role": "user", "content": f"Analyze this document. Extract {prompt}. JSON only.", "images": [img_b64]}],
                "stream": False, "format": "json", "keep_alive": 0,
                "options": {"num_thread": threads}
            }, timeout=180.0)
            
            res_json = response.json()
            content = res_json.get("message", {}).get("content", "").strip()
            elapsed = round(time.time() - start_t, 2)
            
            if not content:
                await log_to_ws(f"❌ Page {page_num}: Empty result.", "error")
            else:
                results.append({"page": page_num, "data": json.loads(content), "perf": elapsed})
                await log_to_ws(f"✅ Page {page_num} processed in {elapsed}s", "success")
        except Exception as e:
            await log_to_ws(f"❌ Page {page_num} Error: {str(e)}", "error")
        queue.task_done()

# --- SCAN ENDPOINT ---
@app.post("/scan")
async def scan(file: UploadFile = File(...), prompt: str = Form(...), mode: str = Form(...)):
    try:
        content = await file.read()
        queue = asyncio.Queue(); results = []
        async with httpx.AsyncClient() as client:
            workers = [asyncio.create_task(ai_worker(queue, client, prompt, 8, results, mode)) for _ in range(2)]
            pages = convert_from_bytes(content, thread_count=8, dpi=200)
            await log_to_ws(f"📂 PDF split into {len(pages)} chunks.")
            for i, p in enumerate(pages):
                img = p.convert("RGB"); img.thumbnail((1800, 1800))
                buf = io.BytesIO(); img.save(buf, format='JPEG', quality=85)
                await queue.put((i+1, base64.b64encode(buf.getvalue()).decode('utf-8')))
            for _ in range(2): await queue.put(None)
            await asyncio.gather(*workers)
        return {"results": sorted(results, key=lambda x: x['page'])}
    except Exception as e:
        await log_to_ws(f"🔥 Failure: {str(e)}", "error")
        raise HTTPException(status_code=500, detail=str(e))