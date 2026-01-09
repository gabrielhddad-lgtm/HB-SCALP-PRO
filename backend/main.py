from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io

VIP_KEY = "HBVIP2026"  # 🔑 TROQUE ESTA SENHA

app = FastAPI(title="HB Signals AI VIP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def analyze_chart(image):
    image = image.resize((400, int(400 * image.height / image.width)))
    img = np.array(image.convert("L"))

    h, w = img.shape
    price = np.array([h - img[:, x].argmin() for x in range(w)])

    smooth = np.convolve(price, np.ones(15)/15, mode="valid")
    x_axis = np.arange(len(smooth))

    slope = np.polyfit(x_axis, smooth, 1)[0]
    delta = smooth[-1] - smooth[0]
    volatility = np.std(np.diff(smooth))

    recent = smooth[-12:]
    pullback = recent.max() - recent.min()

    return slope, delta, volatility, pullback

@app.post("/predict")
async def predict(file: UploadFile = File(...), x_vip_key: str = Header(None)):
    if x_vip_key != VIP_KEY:
        raise HTTPException(status_code=401, detail="Acesso VIP inválido")

    image = Image.open(io.BytesIO(await file.read()))
    slope, delta, volatility, pullback = analyze_chart(image)

    lateral = abs(slope) < 0.015 and abs(delta) < 6 and pullback > 7
    pro = abs(slope) > 0.035 and abs(delta) > 14 and volatility > 2.5 and pullback < 5

    score = 50
    if abs(slope) > 0.02: score += 20
    if abs(delta) > 8: score += 15
    if volatility > 2: score += 10
    if pullback < 6: score += 10

    if lateral:
        signal = "🚫 MERCADO LATERAL"
        score = 35
    elif slope > 0.025 and delta > 10:
        signal = "📈 CALL COMPRA"
    elif slope < -0.025 and delta < -10:
        signal = "📉 CALL VENDA"
    else:
        signal = "⏸️ SEM ENTRADA"

    if pro and not lateral:
        signal += " 💎 PRO TRADER"
        score = min(score + 12, 99)

    return {"signal": signal, "score": score}
