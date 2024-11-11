from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class CarcassDetector:
    # Lista de classes que o modelo pode detectar
    CLASS_NAMES = ["carcaca"]  # Adicionado aqui

    def __init__(self, model_path: str, conf_threshold: float = 0.7):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None
        self.model_version = "YOLOv8"  # Adicionado aqui
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
            results = self.model(image, conf=self.conf_threshold)[0]
            
            detections = []
            for r in results.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = r
                detections.append({
                    "bbox": [int(x) for x in [x1, y1, x2, y2]],  # Convertido para inteiros
                    "confidence": float(conf),
                    "class": int(cls),
                    "class_name": self.CLASS_NAMES[int(cls)]  # Adicionado nome da classe
                })
            
            return {
                "detected": len(detections) > 0,
                "detections": detections,
                "model_info": {  # Adicionado informações do modelo
                    "version": self.model_version,
                    "confidence_threshold": self.conf_threshold,
                    "model_path": str(self.model_path)
                }
            }
            
        except Exception as e:
            logger.error(f"Erro na detecção: {e}")
            raise