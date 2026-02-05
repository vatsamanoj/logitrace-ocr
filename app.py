import os
import io
import base64
import json
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

app = FastAPI()

# Model configuration
MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.5"

# Load Model and Processor (Optimized for 8GB RAM CPU)
try:
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        trust_remote_code=True,
        torch_dtype=torch.float32  # Standard for CPU stability
    ).to("cpu").eval()
except Exception as e:
    print(f"Error loading model: {e}")

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("index.html", "r") as f:
        return f.read()

@app.post("/scan")
async def scan_document(file: UploadFile = File(...), prompt: str = Form(...)):
    try:
        # Process image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Inference
        inputs = processor(text=prompt, images=image, return_tensors="pt").to("cpu")
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=1024,
                do_sample=False  # Greedy decoding for consistent OCR results
            )
        
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Return as JSON
        try:
            return json.loads(output_text)
        except:
            return {"raw_output": output_text}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))