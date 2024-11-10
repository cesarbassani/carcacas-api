from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class CarcassDetector:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self) -> None:
        """Carrega o modelo YOLO"""
        try:
            self.model = YOLO(self.model_path)
            logger.info(f"Modelo carregado: {self.model_path}")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            raise

    def is_model_loaded(self) -> bool:
        """Verifica se o modelo está carregado"""
        return self.model is not None

    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """Detecta carcaças na imagem"""
        try:
            # Fazer predição
            results = self.model(image, conf=0.7)[0]
            
            detections = []
            for r in results.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = r
                detections.append({
                    "bbox": [float(x) for x in [x1, y1, x2, y2]],
                    "confidence": float(conf),
                    "class": int(cls)
                })
            
            return {
                "detected": len(detections) > 0,
                "detections": detections
            }
            
        except Exception as e:
            logger.error(f"Erro na detecção: {e}")
            raise