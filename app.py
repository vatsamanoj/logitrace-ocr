import os
import io
import base64
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
import httpx

app = FastAPI()
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://ollama-vlm:11434")

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("index.html", "r") as f:
        return f.read()

@app.post("/scan")
async def scan(file: UploadFile = File(...), prompt: str = Form(...)):
    # Convert image to Base64 for Qwen2.5-VL
    image_data = base64.b64encode(await file.read()).decode('utf-8')
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": "qwen2.5-vl:3b-q4_K_M",
                "messages": [{
                    "role": "user",
                    "content": prompt,
                    "images": [image_data]
                }],
                "stream": False,
                "format": "json" # Qwen2.5-VL excels at structured JSON
            }
        )
    return response.json()