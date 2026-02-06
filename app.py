import os, io, base64, httpx
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
    return "<h1>System Reset - Ready</h1>"

@app.post("/scan")
async def scan(file: UploadFile = File(...), prompt: str = Form(...)):
    try:
        content = await file.read()
        
        # Determine images
        if file.content_type == "application/pdf":
            # thread_count=2 is safer for your CPU stability
            pages = convert_from_bytes(content, thread_count=2)
            images = [p.convert("RGB") for p in pages]
        else:
            images = [Image.open(io.BytesIO(content)).convert("RGB")]

        results = []
        
        # Use a single client for all pages
        async with httpx.AsyncClient(timeout=300.0) as client:
            for i, img in enumerate(images):
                # Standardize size to 1024px - safe for 7B model
                img.thumbnail((1024, 1024)) 
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG', quality=85)
                image_data = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

                response = await client.post(
                    f"{OLLAMA_URL}/api/chat",
                    json={
                        "model": "qwen2.5vl:7b",
                        "messages": [{
                            "role": "user",
                            "content": f"Extract: {prompt}. JSON output only.",
                            "images": [image_data]
                        }],
                        "stream": False,
                        "format": "json",
                        "keep_alive": "5m", # Automatically UNLOAD model after 5 mins of inactivity
                        "options": {
                            "num_thread": 8 # Strictly use 8 threads per scan
                        }
                    }
                )
                
                if response.status_code != 200:
                    results.append({"page": i+1, "error": "Ollama Busy or Offline"})
                    continue

                res_json = response.json()
                results.append({
                    "page": i + 1,
                    "data": res_json.get("message", {}).get("content", "")
                })

        return {"filename": file.filename, "results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))