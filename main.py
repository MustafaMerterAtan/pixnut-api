import os
import json
import zipfile
import numpy as np
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import gdown

# Keras 3 kullan - TensorFlow'un kendi Keras'ını değil, standalone Keras 3'ü kullan
os.environ['TF_USE_LEGACY_KERAS'] = '0'
import keras

app = FastAPI(
    title="PixNut Food Analysis API",
    description="Yemek fotoğraflarından besin değeri tahmini yapan API",
    version="1.0.0"
)

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model ve istatistikler
MODEL = None
TARGET_STATS = None
IMG_SIZE = (384, 384)

# Google Drive dosya ID'si
DRIVE_FILE_ID = "1j-cJhJU7jBWjUpUtfzuHeNl_JGi11uWi"
MODEL_DIR = "/tmp/model"

class NutritionResponse(BaseModel):
    success: bool
    kcal: float
    carbs: float
    protein: float
    fat: float
    grams: float = 100.0
    message: str = ""

def download_model_from_drive():
    """Google Drive'dan modeli indir"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    zip_path = f"{MODEL_DIR}/model.zip"
    
    # Eğer model zaten varsa indirme
    model_path = f"{MODEL_DIR}/qiima_nutrition_regressor.keras"
    if os.path.exists(model_path):
        print("Model zaten mevcut, indirme atlanıyor...")
        return
    
    print("Model Google Drive'dan indiriliyor...")
    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    gdown.download(url, zip_path, quiet=False)
    
    print("Zip dosyası açılıyor...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(MODEL_DIR)
    
    # Zip dosyasını sil
    os.remove(zip_path)
    print("Model başarıyla indirildi!")

def load_model():
    """Model ve istatistikleri yükle"""
    global MODEL, TARGET_STATS, IMG_SIZE
    
    # Önce modeli indir
    download_model_from_drive()
    
    # Model yolları
    model_path = f"{MODEL_DIR}/qiima_nutrition_regressor.keras"
    stats_path = f"{MODEL_DIR}/target_stats.json"
    
    # Alternatif yollar (zip içinde klasör varsa)
    if not os.path.exists(model_path):
        for root, dirs, files in os.walk(MODEL_DIR):
            for file in files:
                if file.endswith('.keras'):
                    model_path = os.path.join(root, file)
                if file == 'target_stats.json':
                    stats_path = os.path.join(root, file)
    
    print(f"Model yolu: {model_path}")
    print(f"Stats yolu: {stats_path}")
    
    # İstatistikleri yükle
    with open(stats_path, 'r') as f:
        TARGET_STATS = json.load(f)
    
    IMG_SIZE = tuple(TARGET_STATS['img_size'])
    
    # Custom loss function
    import tensorflow as tf
    
    @keras.saving.register_keras_serializable()
    def weighted_huber(y_true, y_pred, delta=1.0):
        error = y_true - y_pred
        abs_error = tf.abs(error)
        quadratic = tf.minimum(abs_error, delta)
        linear = abs_error - quadratic
        return tf.reduce_mean(0.5 * quadratic ** 2 + delta * linear)
    
    # Modeli yükle
    MODEL = keras.models.load_model(
        model_path,
        custom_objects={'weighted_huber': weighted_huber}
    )
    print(f"Model yüklendi!")
    print(f"Image size: {IMG_SIZE}")

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Görüntüyü model için hazırla"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(IMG_SIZE, Image.Resampling.LANCZOS)
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def inverse_transform(predictions: np.ndarray) -> dict:
    """Model çıktısını gerçek değerlere dönüştür"""
    import numpy as np
    cols = TARGET_STATS['cols']
    z_mean = TARGET_STATS['z_mean']
    z_std = TARGET_STATS['z_std']
    space = TARGET_STATS['space']
    
    result = {}
    for i, col in enumerate(cols):
        if i >= predictions.shape[1]:
            break
        val = predictions[0][i] * z_std[col] + z_mean[col]
        if space == 'log1p':
            val = np.expm1(val)
        result[col] = max(0, float(val))
    return result

@app.on_event("startup")
async def startup_event():
    """Uygulama başladığında modeli yükle"""
    try:
        load_model()
    except Exception as e:
        print(f"Model yükleme hatası: {e}")
        import traceback
        traceback.print_exc()

@app.get("/")
async def root():
    return {"message": "PixNut Food Analysis API", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "img_size": IMG_SIZE
    }

@app.post("/predict", response_model=NutritionResponse)
async def predict_nutrition(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model henüz yüklenmedi")
    
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Sadece resim dosyaları kabul edilir")
    
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        img_array = preprocess_image(image)
        predictions = MODEL.predict(img_array, verbose=0)
        nutrition = inverse_transform(predictions)
        
        return NutritionResponse(
            success=True,
            kcal=round(nutrition.get('kcal', 0), 1),
            carbs=round(nutrition.get('carbs', 0), 1),
            protein=round(nutrition.get('protein', 0), 1),
            fat=round(nutrition.get('fat', 0), 1),
            grams=round(nutrition.get('grams', 100), 1),
            message="Besin değerleri başarıyla tahmin edildi"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tahmin hatası: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
