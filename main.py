from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import openai
import os
from pydantic import BaseModel

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranscriptionRequest(BaseModel):
    language: str  # "ur", "hi", or "en"

@app.post("/transcribe")
async def transcribe(file: UploadFile, request: TranscriptionRequest):
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Transcribe with Whisper
        audio_file = open(temp_path, "rb")
        transcript = openai.Audio.transcribe(
            "whisper-1", 
            audio_file,
            language=request.language
        )
        
        # Cleanup
        os.remove(temp_path)
        
        return {"text": transcript["text"]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))