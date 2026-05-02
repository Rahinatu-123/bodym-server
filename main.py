import asyncio
import numpy as np
import onnxruntime as ort
import cv2
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import os

#  Config 
IMG_SIZE      = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SCALER_MEAN  = np.array([
    24.21320478835926, 49.46577138623756, 30.586734237952296,
    37.52832512044313, 102.0268291703783, 26.57868651015905,
    171.9552159465475, 103.04196008178373, 78.14934779862315,
    35.80862319255106, 64.98503210930409, 54.37324460611304,
    90.25416387309123, 16.69412272599723
], dtype=np.float32)

SCALER_SCALE = np.array([
    1.8864193484051586, 3.139496217615162, 4.030189258683643,
    3.278421741585196, 11.368133927829547, 2.692806669173353,
    9.454396900297526, 9.948770493654228, 4.9364423313710475,
    2.6867897558397607, 4.947719724374547, 5.548706630529534,
    13.611109068075107, 1.46575005562859
], dtype=np.float32)

MEASUREMENT_COLS = [
    'ankle', 'arm-length', 'bicep', 'calf', 'chest', 'forearm',
    'height', 'hip', 'leg-length', 'shoulder-breadth',
    'shoulder-to-crotch', 'thigh', 'waist', 'wrist'
]

#  Load model 
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'bodym_model.onnx')
print("Loading measurement model...")
measurement_session = ort.InferenceSession(MODEL_PATH)
print("✓ Measurement model loaded")

# App 
app = FastAPI(title="Body Measurement API", version="1.0.0")

# Keep-alive task 
async def keep_alive():
    import httpx
    url = "https://bodym-server.onrender.com/health"
    while True:
        await asyncio.sleep(600)
        try:
            async with httpx.AsyncClient() as client:
                await client.get(url, timeout=10)
            print("✓ Keep-alive ping sent")
        except Exception as e:
            print(f"Keep-alive failed: {e}")

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(keep_alive())
    print("✓ Keep-alive task started")

#  Segmentation 
def extract_silhouette(image_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_cv is None:
        raise ValueError("Could not decode image")

    h, w     = img_cv.shape[:2]
    mask     = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    margin_x = int(w * 0.1)
    margin_y = int(h * 0.05)
    rect     = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

    cv2.grabCut(img_cv, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    fg_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel)

    return cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2RGB)


def preprocess_silhouette(silhouette: np.ndarray) -> np.ndarray:
    resized = cv2.resize(silhouette, (IMG_SIZE, IMG_SIZE))
    image   = resized.astype(np.float32) / 255.0
    image   = (image - IMAGENET_MEAN) / IMAGENET_STD
    return image.transpose(2, 0, 1)  # (3, 224, 224)


#  Routes 
@app.get("/")
def root():
    return {"status": "ok", "message": "Body Measurement API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
async def predict(
    front_image: UploadFile = File(...),
    side_image:  UploadFile = File(...),
    gender:      str        = Form(...),
    height_cm:   float      = Form(...),
    weight_kg:   float      = Form(...),
):
    try:
        front_bytes = await front_image.read()
        side_bytes  = await side_image.read()

        front_sil = extract_silhouette(front_bytes)
        side_sil  = extract_silhouette(side_bytes)

        front_np  = preprocess_silhouette(front_sil)
        side_np   = preprocess_silhouette(side_sil)

        gender_val = 1.0 if gender.lower() == 'female' else 0.0
        aux_vals   = np.array([gender_val, height_cm, weight_kg], dtype=np.float32)
        aux_tiled  = np.tile(aux_vals[:, None, None], (1, IMG_SIZE, IMG_SIZE))

        input_tensor = np.concatenate([front_np, side_np, aux_tiled], axis=0)
        input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)

        input_name = measurement_session.get_inputs()[0].name
        raw_output = measurement_session.run(None, {input_name: input_tensor})[0][0]

        predicted_cm = (raw_output * SCALER_SCALE) + SCALER_MEAN

        measurements = {
            col: round(float(val), 1)
            for col, val in zip(MEASUREMENT_COLS, predicted_cm)
        }

        return JSONResponse(content={
            "success":      True,
            "measurements": measurements,
            "gender":       gender,
            "height_cm":    height_cm,
            "weight_kg":    weight_kg,
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )