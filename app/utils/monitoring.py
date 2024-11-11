import logging
import time
from functools import wraps
import psutil
import json
from datetime import datetime

logger = logging.getLogger(__name__)

def log_system_metrics():
    """Log system metrics"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used_mb': memory.used / (1024 * 1024),
            'disk_percent': disk.percent,
            'disk_free_gb': disk.free / (1024 * 1024 * 1024)
        }
        
        logger.info(f"System metrics: {json.dumps(metrics)}")
        return metrics
    except Exception as e:
        logger.error(f"Error collecting system metrics: {e}")
        return None

def monitor_endpoint(name):
    """Decorator to monitor endpoint performance"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                status = 'success'
            except Exception as e:
                status = 'error'
                raise e
            finally:
                execution_time = (time.time() - start_time) * 1000  # em ms
                
                metrics = {
                    'endpoint': name,
                    'status': status,
                    'execution_time_ms': execution_time,
                    'timestamp': datetime.now().isoformat()
                }
                
                logger.info(f"Endpoint metrics: {json.dumps(metrics)}")
                
                # Log alert if response time is too high
                if execution_time > 2000:  # 2 segundos
                    logger.warning(f"High response time on {name}: {execution_time}ms")
                
                # Collect system metrics after processing
                log_system_metrics()
            
            return result
        return wrapper
    return decorator