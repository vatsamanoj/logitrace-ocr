import os, io, base64, httpx, asyncio
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image
from pdf2image import convert_from_bytes

app = FastAPI()
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://ollama-vlm:11434")

@app.get("/", response_class=HTMLResponse)
async def home():
    if os.path.exists("index.html"):
        with open("index.html", "r") as f:
            return f.read()
    return "<h1>LogiTrace OCR Parallel Ready</h1>"

async def process_page(client, image_data, page_num, prompt):
    response = await client.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": "smolvlm",
            "messages": [{
                "role": "user",
                "content": f"Extract from page {page_num} as JSON: {prompt}",
                "images": [image_data]
            }],
            "stream": False,
            "format": "json",
            "keep_alive": -1,
            "options": {
                "num_predict": 150,
                "num_thread": 4 # Distributed threads for parallel tasks
            }
        }
    )
    res_json = response.json()
    return {
        "page": page_num,
        "data": res_json.get("message", {}).get("content", "")
    }

@app.post("/scan")
async def scan(file: UploadFile = File(...), prompt: str = Form(...)):
    try:
        content = await file.read()
        
        if file.content_type == "application/pdf":
            pages = convert_from_bytes(content, thread_count=8)
            images = [p.convert("RGB") for p in pages]
        else:
            images = [Image.open(io.BytesIO(content)).convert("RGB")]

        tasks = []
        async with httpx.AsyncClient(timeout=120.0) as client:
            for i, img in enumerate(images):
                img.thumbnail((768, 768)) 
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG', quality=85)
                image_data = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
                tasks.append(process_page(client, image_data, i + 1, prompt))

            results = await asyncio.gather(*tasks)

        return {
            "filename": file.filename, 
            "total_pages": len(results), 
            "results": sorted(results, key=lambda x: x['page'])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))