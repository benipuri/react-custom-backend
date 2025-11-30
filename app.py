import os
import io
import json
import base64
import numpy as np
import cv2

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from simple_lama_inpainting import SimpleLama
from segment_anything import sam_model_registry, SamPredictor

# ============================================================
# FASTAPI APP + CORS (for React frontend later)
# ============================================================
app = FastAPI(title="Sticker Remover – Points/Boxes Version")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # you can later restrict to your Hostinger domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# MODEL SETUP (SAM + LaMa)
# ============================================================
DEVICE = "cuda" if (os.environ.get("USE_CUDA", "1") == "1" and
                    hasattr(__import__("torch"), "cuda") and
                    __import__("torch").cuda.is_available()) else "cpu"

MODEL_TYPE = "vit_b"
SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"  # downloaded in Dockerfile

print("==========================================")
print(" Loading SAM + LaMa models for sticker removal")
print(f" DEVICE: {DEVICE}")
print(f" SAM checkpoint: {SAM_CHECKPOINT}")
print("==========================================")

sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(DEVICE)
sam.eval()
predictor = SamPredictor(sam)

lama = SimpleLama(device=DEVICE)
print("✅ Models loaded and ready.")


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def load_rgb_bytes(data: bytes) -> np.ndarray:
    """Load image bytes as RGB numpy array."""
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(img)


def build_mask(img: np.ndarray, points_json: str, boxes_json: str, dilate_px: int = 7) -> np.ndarray:
    """
    Build a binary mask (0/255) using SAM, based on points + boxes from frontend.
    points_json / boxes_json are JSON strings from form fields.
    """
    predictor.set_image(img)

    try:
        pts = json.loads(points_json or "[]")
    except Exception:
        pts = []

    try:
        bxs = json.loads(boxes_json or "[]")
    except Exception:
        bxs = []

    mask_total = np.zeros(img.shape[:2], dtype=np.uint8)

    # ---------- Points ----------
    for p in pts:
        x, y = float(p["x"]), float(p["y"])
        masks, scores, _ = predictor.predict(
            point_coords=np.array([[x, y]], dtype=np.float32),
            point_labels=np.array([1], dtype=np.int32),
            multimask_output=True,
        )
        best_mask = masks[np.argmax(scores)]
        mask_total = np.logical_or(mask_total, best_mask)

    # ---------- Boxes ----------
    for b in bxs:
        box = np.array(
            [float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])],
            dtype=np.float32,
        )
        masks, scores, _ = predictor.predict(
            box=box[None, :],
            multimask_output=True,
        )
        best_mask = masks[np.argmax(scores)]
        mask_total = np.logical_or(mask_total, best_mask)

    # Convert to 0–255 uint8
    mask_total = (mask_total.astype(np.uint8) * 255)

    # Optional dilation to cover edges
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        mask_total = cv2.dilate(mask_total, k, iterations=1)

    return mask_total


def inpaint_image(img: np.ndarray, mask: np.ndarray, mode: str = "roi") -> np.ndarray:
    """
    Apply LaMa inpainting:
      - mode = 'roi'  → crop around mask (fast)
      - mode = 'full' → full image (slower)
    Returns RGB numpy array.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("No mask found")

    if mode == "roi":
        pad = 20
        x0, x1 = max(0, xs.min() - pad), min(img.shape[1] - 1, xs.max() + pad)
        y0, y1 = max(0, ys.min() - pad), min(img.shape[0] - 1, ys.max() + pad)

        roi_img = img[y0:y1 + 1, x0:x1 + 1]
        roi_mask = mask[y0:y1 + 1, x0:x1 + 1]

        roi_out = lama(roi_img, roi_mask)
        if isinstance(roi_out, Image.Image):
            roi_out = np.array(roi_out)

        # sanity: match shapes
        if roi_out.shape[:2] != roi_img.shape[:2]:
            roi_out = cv2.resize(roi_out, (roi_img.shape[1], roi_img.shape[0]))

        out = img.copy()
        out[y0:y1 + 1, x0:x1 + 1] = roi_out

    else:
        out = lama(img, mask)
        if isinstance(out, Image.Image):
            out = np.array(out)

    return out


def encode_webp(img_rgb: np.ndarray) -> bytes:
    """
    Encode an RGB image to WebP bytes (no temp files).
    """
    img_bgr = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(".webp", img_bgr)
    if not success:
        raise RuntimeError("Failed to encode WebP")
    return encoded.tobytes()


# ============================================================
# HEALTH ENDPOINTS
# ============================================================
@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.get("/")
def home():
    return {"message": "Sticker Remover API (points/boxes) ready"}


# ============================================================
# MAIN PROCESS ENDPOINT
# ============================================================
@app.post("/process")
async def process_endpoint(
    file: UploadFile = File(...),
    points: str = Form("[]"),
    boxes: str = Form("[]"),
    mode: str = Form("roi"),
    dilate_px: int = Form(7),
):
    """
    Accepts:
      - file: image file (JPG/PNG/WEBP)
      - points: JSON string: [{ "x": ..., "y": ... }, ...]
      - boxes:  JSON string: [{ "x1":..., "y1":..., "x2":..., "y2":... }, ...]
      - mode:   "roi" (fast) or "full" (slower)
      - dilate_px: int, optional mask dilation

    Returns:
      - 200: WebP binary image (media_type="image/webp")
      - 4xx/5xx: JSON error
    """
    try:
        img_bytes = await file.read()
        img = load_rgb_bytes(img_bytes)

        # build mask from user selections
        mask = build_mask(img, points, boxes, dilate_px=int(dilate_px))

        # ensure something was selected
        if not np.any(mask > 0):
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "No mask found from points/boxes"},
            )

        # run LaMa
        mode = (mode or "roi").lower()
        if mode not in ("roi", "full"):
            mode = "roi"

        out_img = inpaint_image(img, mask, mode=mode)

        # encode to WebP
        webp_bytes = encode_webp(out_img)

        return Response(
            content=webp_bytes,
            media_type="image/webp",
            headers={"X-Status": "success", "X-Mode": mode},
        )

    except Exception as e:
        import traceback
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e),
                "trace": traceback.format_exc(),
            },
        )


# ============================================================
# MAIN (for local dev; RunPod uses this too via CMD ["python3", "app.py"])
# ============================================================
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 80))
    uvicorn.run(app, host="0.0.0.0", port=port)
