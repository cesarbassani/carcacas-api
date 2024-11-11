from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from datetime import datetime
import logging
import time
import psutil
from typing import Dict, Any
from .utils.detector import CarcassDetector
from .config import settings
from .utils.monitoring import monitor_endpoint, log_system_metrics

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Carcass Detection API",
    description="API for detecting beef carcasses using YOLOv8",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ajuste conforme necessário
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar detector
detector = CarcassDetector(settings.MODEL_PATH)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.get("/", tags=["Status"])
@monitor_endpoint("root")
async def root():
    """Root endpoint para verificar se a API está online"""
    return {
        "name": "Carcass Detection API",
        "version": "1.0.0",
        "status": "online",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", tags=["Status"])
@monitor_endpoint("health_check")
async def health_check():
    """Health check endpoint para verificar o status do sistema"""
    try:
        system_metrics = log_system_metrics()
        health_data = {
            "status": "healthy",
            "model_loaded": detector.is_model_loaded(),
            "timestamp": datetime.now().isoformat(),
            "system_info": system_metrics
        }
        return health_data
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Health check failed: {str(e)}"
        )

@app.post("/detect", tags=["Detection"])
@monitor_endpoint("detect_carcass")
async def detect_carcass(
    file: UploadFile = File(...),
) -> Dict[str, Any]:
    """
    Detecta carcaças em uma imagem
    
    - **file**: Arquivo de imagem (JPG, PNG)
    
    Returns:
        Dict contendo as detecções encontradas
    """
    try:
        # Validar tipo do arquivo
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )
        
        # Validar tamanho do arquivo (10MB max)
        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(
                status_code=400,
                detail="File size too large. Maximum size is 10MB"
            )
        
        # Processar imagem
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid image format"
            )
        
        # Detectar carcaça
        start_time = time.time()
        result = detector.detect(image)
        process_time = time.time() - start_time
        
        # Adicionar metadados
        result["metadata"] = {
            "process_time": process_time,
            "image_size": {
                "width": image.shape[1],
                "height": image.shape[0]
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return result
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in detection: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}"
        )

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}")
    return {
        "error": "Internal server error",
        "detail": str(exc),
        "path": request.url.path
    }