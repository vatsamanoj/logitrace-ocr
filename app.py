import os
import io
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

app = FastAPI()

MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.5"

# Load Model globally to avoid "name not defined" errors
# Ensure 'einops' and 'torchvision' are in requirements.txt
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    trust_remote_code=True,
    dtype=torch.float32  # Updated from torch_dtype to avoid warnings
).to("cpu").eval()

@app.get("/", response_class=HTMLResponse)
async def home():
    if os.path.exists("index.html"):
        with open("index.html", "r") as f:
            return f.read()
    return "<h1>LogiTrace OCR API is Running</h1>"

@app.post("/scan")
async def scan_document(file: UploadFile = File(...), prompt: str = Form(...)):
    # Use the globally defined 'model' here
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        inputs = processor(text=prompt, images=image, return_tensors="pt").to("cpu")
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=1024)
        
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return {"result": output_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))