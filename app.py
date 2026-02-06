import os, io, base64, httpx, asyncio, psutil, time
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from PIL import Image
from pdf2image import convert_from_bytes

app = FastAPI()
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://ollama-vlm:11434")

# Global set to manage active WebSocket connections for logging
active_connections = set()

async def log_to_ws(msg: str):
    timestamp = time.strftime("%H:%M:%S")
    payload = {"time": timestamp, "msg": msg}
    for connection in active_connections:
        try:
            await connection.send_json(payload)
        except:
            pass

async def reclaim_ram():
    await log_to_ws("🧹 Reclaiming RAM: Flushing model weights...")
    try:
        async with httpx.AsyncClient() as client:
            await client.post(f"{OLLAMA_URL}/api/chat", json={"model": "qwen2.5vl:7b", "keep_alive": 0})
        await log_to_ws("✅ System Idle. RAM fully reclaimed.")
    except:
        pass

async def ai_worker(queue, client, prompt, threads, results):
    while True:
        item = await queue.get()
        if item is None: break
        
        page_num, img_b64 = item
        start_time = time.time()
        await log_to_ws(f"🧠 Page {page_num}: AI Analysis started ({threads} threads)...")
        
        try:
            response = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": "qwen2.5vl:7b",
                    "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
                    "stream": False, "format": "json", "keep_alive": "1m",
                    "options": {"num_thread": threads, "temperature": 0}
                }, timeout=120.0
            )
            elapsed = round(time.time() - start_time, 2)
            await log_to_ws(f"✨ Page {page_num}: Completed in {elapsed}s")
            results.append({"page": page_num, "data": response.json().get("message", {}).get("content", "")})
        except Exception as e:
            await log_to_ws(f"❌ Page {page_num}: Error - {str(e)}")
        
        queue.task_done()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.add(websocket)
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections.remove(websocket)

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("index.html", "r") as f: return f.read()

@app.post("/scan")
async def scan(file: UploadFile = File(...), prompt: str = Form(...)):
    cores = os.cpu_count() or 4
    threads = max(1, int(cores * 0.75))
    
    try:
        content = await file.read()
        queue = asyncio.Queue(maxsize=2)
        results = []

        async with httpx.AsyncClient() as client:
            worker = asyncio.create_task(ai_worker(queue, client, prompt, threads, results))

            if file.content_type == "application/pdf":
                await log_to_ws(f"📄 PDF Detected. Initializing pipeline for {file.filename}...")
                # Convert pages one by one to feed the pipeline instantly
                pages = convert_from_bytes(content, thread_count=4)
                for i, page in enumerate(pages):
                    img = page.convert("RGB")
                    img.thumbnail((1600, 1600)) # High res for legal/LC docs
                    buf = io.BytesIO()
                    img.save(buf, format='JPEG', quality=85)
                    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                    await queue.put((i + 1, b64))
            else:
                img = Image.open(io.BytesIO(content)).convert("RGB")
                img.thumbnail((1600, 1600))
                buf = io.BytesIO()
                img.save(buf, format='JPEG', quality=85)
                await queue.put((1, base64.b64encode(buf.getvalue()).decode('utf-8')))

            await queue.put(None)
            await worker
            await reclaim_ram()

        return {"filename": file.filename, "results": sorted(results, key=lambda x: x['page'])}

    except Exception as e:
        await reclaim_ram()
        raise HTTPException(status_code=500, detail=str(e))