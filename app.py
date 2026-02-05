import os, io, base64, json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

app = FastAPI()

# Optimized for CPU and 8GB RAM
MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.5"
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID, 
    trust_remote_code=True,
    torch_dtype=torch.float32 # Use float32 for CPU
).to("cpu").eval()

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("index.html", "r") as f:
        return f.read()

@app.post("/scan")
async def scan_document(file: UploadFile = File(...), prompt: str = Form(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Prepare inputs for PaddleOCR-VL-1.5
        inputs = processor(text=prompt, images=image, return_tensors="pt").to("cpu")
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=1024)
        
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Force JSON parsing if possible
        try:
            return json.loads(output_text)
        except:
            return {"raw_output": output_text}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))