import os
import sys
import pandas as pd
import numpy as np
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv
import io
from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path
import json
import pickle

# Add current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env'))

# Configuration
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
MODEL_REGISTRY_NAME = os.getenv('MODEL_REGISTRY_NAME', 'severity_detection_best_model')

# Set MLflow tracking URI
if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, preprocessor, model_metadata
    
    # Initialize Preprocessor
    try:
        from inference_preprocessor import InferencePreprocessor
        preprocessor = InferencePreprocessor()
        print("✅ Preprocessor initialized (Traffic Logic).")
    except Exception as e:
        print(f"❌ Error initializing preprocessor: {e}")
        
    # Load Model from Local Registry
    try:
        # Check if running in Docker (model_registry is mounted at /app/model_registry)
        if os.path.exists('/app/model_registry'):
            MODEL_REGISTRY_DIR = Path('/app/model_registry')
        else:
            # Local development path
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            MODEL_REGISTRY_DIR = Path(base_dir) / 'notebooks' / 'model_registry'
        
        def load_from_registry(model_name, stage="production"):
            model_dir = MODEL_REGISTRY_DIR / model_name.replace(" ", "_")
            model_path = model_dir / f"{stage}.pkl"
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found at {model_path}")
            
            with open(model_path, 'rb') as f:
                mod = pickle.load(f)
            
            # Metadata
            versions = [d for d in model_dir.iterdir() if d.is_dir()]
            meta = {}
            if versions:
                latest = sorted(versions)[-1]
                if (latest / "metadata.json").exists():
                    with open(latest / "metadata.json", 'r') as f:
                        meta = json.load(f)
            
            return mod, meta

        print(f"🔄 Loading model from local registry...")
        
        # Target Model Name (Traffic Severity)
        # Based on file listing, we expect a folder with traffic model.
        # Ideally user has 'Best_Severity_Voting_Soft' or similar. 
        # API will try to find a valid directory.
        
        candidates = ["Best Severity Voting Soft", "Best Fraud Voting Soft"] # Fallback to Fraud name if folder wasn't renamed
        target_model_name = candidates[0]
        
        found = False
        for name in candidates:
            if (MODEL_REGISTRY_DIR / name.replace(" ", "_")).exists():
                target_model_name = name
                found = True
                break
        
        if not found:
            print(f"⚠️ No model directory found in {MODEL_REGISTRY_DIR}. Predictions will fail.")
        else:
            model, metadata = load_from_registry(target_model_name, stage="production")
            print(f"✅ Model loaded: {target_model_name}")
            
            model_metadata = {
                "name": metadata.get('model_name', target_model_name),
                "loaded_at": datetime.now().isoformat(),
                "metadata": metadata
            }
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
    
    yield
    # Shutdown

app = FastAPI(
    title="Traffic Severity Detection API",
    description="API for predicting US Traffic Accident Severity (1-4)",
    version="2.0.0",
    lifespan=lifespan
)

# Global variables
model = None
preprocessor = None
model_metadata = {}

@app.get("/")
def read_root():
    return {"message": "Traffic Severity Detection API Online."}

@app.get("/health")
def health_check():
    status = "ok" if model is not None and preprocessor is not None else "degraded"
    return {"status": status}

@app.get("/model-info")
def get_model_info():
    return model_metadata

# Input Schema for US Accidents
class AccidentInput(BaseModel):
    Start_Time: str = "2025-01-01 08:30:00"
    Start_Lat: float = 37.7749
    Start_Lng: float = -122.4194
    Temperature_F: float = 65.0
    Humidity_Percent: float = 50.0
    Pressure_in: float = 29.92
    Visibility_mi: float = 10.0
    Wind_Speed_mph: float = 5.0
    Precipitation_in: float = 0.0
    Weather_Condition: str = "Clear"
    Amenity: bool = False
    Bump: bool = False
    Crossing: bool = False
    Give_Way: bool = False
    Junction: bool = False
    No_Exit: bool = False
    Railway: bool = False
    Roundabout: bool = False
    Station: bool = False
    Stop: bool = False
    Traffic_Calming: bool = False
    Traffic_Signal: bool = False
    Turning_Loop: bool = False
    Sunrise_Sunset: str = "Day"
    Wind_Direction: str = "W"
    Side: str = "R"

@app.post("/predict")
async def predict(request: Union[AccidentInput, List[AccidentInput]]):
    if not model or not preprocessor:
        raise HTTPException(status_code=503, detail="Model/Preprocessor unavailable")
    
    try:
        # Normalize input to list of dicts with mapped keys (Pydantic -> DF columns)
        # Using alias logic or manual mapping
        
        data_list = []
        inputs = request if isinstance(request, list) else [request]
        
        for item in inputs:
            # Map Pydantic friendly names to Dataset Column names
            d = item.model_dump()
            mapped = {
                'Start_Time': d['Start_Time'],
                'Start_Lat': d['Start_Lat'],
                'Start_Lng': d['Start_Lng'],
                'Temperature(F)': d['Temperature_F'],
                'Humidity(%)': d['Humidity_Percent'],
                'Pressure(in)': d['Pressure_in'],
                'Visibility(mi)': d['Visibility_mi'],
                'Wind_Speed(mph)': d['Wind_Speed_mph'],
                'Precipitation(in)': d['Precipitation_in'],
                'Weather_Condition': d['Weather_Condition'],
                'Amenity': d['Amenity'],
                'Bump': d['Bump'],
                'Crossing': d['Crossing'],
                'Give_Way': d['Give_Way'],
                'Junction': d['Junction'],
                'No_Exit': d['No_Exit'],
                'Railway': d['Railway'],
                'Roundabout': d['Roundabout'],
                'Station': d['Station'],
                'Stop': d['Stop'],
                'Traffic_Calming': d['Traffic_Calming'],
                'Traffic_Signal': d['Traffic_Signal'],
                'Turning_Loop': d['Turning_Loop'],
                'Sunrise_Sunset': d['Sunrise_Sunset'],
                'Wind_Direction': d['Wind_Direction'],
                'Side': d['Side']
            }
            data_list.append(mapped)
            
        df_input = pd.DataFrame(data_list)
        
        # Preprocess
        X = preprocessor.preprocess_inference(df_input)
        
        # Predict
        predictions = model.predict(X)
        
        return {
            "predictions": predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
            "count": len(predictions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predictCSV")
async def predict_csv(file: UploadFile = File(...)):
    if not model or not preprocessor:
        raise HTTPException(status_code=503, detail="Unavailable")
        
    try:
        contents = await file.read()
        df_input = pd.read_csv(io.BytesIO(contents))
        
        # Ensure it matches expected columns or map if possible?
        # Assuming CSV has correct Dataset columns (Start_Lat, etc)
        
        X = preprocessor.preprocess_inference(df_input)
        predictions = model.predict(X)
        
        df_result = df_input.copy()
        df_result['Predicted_Severity'] = predictions
        
        output = io.StringIO()
        df_result.to_csv(output, index=False)
        output.seek(0)
        
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=pred_{file.filename}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))