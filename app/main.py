from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from datetime import datetime
import logging
from app.utils.detector import CarcassDetector
from app.config import settings

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Carcass Detection API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar detector
detector = CarcassDetector(settings.MODEL_PATH)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": detector.is_model_loaded(),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/detect")
async def detect_carcass(file: UploadFile = File(...)):
    try:
        # Ler imagem
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        # Detectar carcaça
        result = detector.detect(image)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in detection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))