from enum import Enum
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import torch, io, tempfile, asyncio
from inference import submit_request, batching_loop

class ClothType(str, Enum):
    upper = "upper"
    lower = "lower"
    overall = "overall"

class ShowType(str, Enum):
    result_only = "result only"
    all = "all"

class MixedPrecision(str, Enum):
    fp16 = "fp16"
    fp32 = "fp32"
    bf16 = "bf16"

app = FastAPI(title="CatVTON Try-On API")

# Start dynamic-batching loop
@app.on_event("startup")
async def on_startup():
    asyncio.create_task(batching_loop())

@app.post("/tryon/")
async def tryon(
    person_image: UploadFile = File(...),
    cloth_image: UploadFile = File(...),
    cloth_type: ClothType = Form(...),
    num_inference_steps: int = Form(25),
    guidance_scale: float = Form(3.5),
    seed: int = Form(42),
    show_type: ShowType = Form(...),
    mixed_precision: MixedPrecision = Form(MixedPrecision.fp16)
):
    # Load and save images
    try:
        person_pil = Image.open(person_image.file).convert("RGB")
        cloth_pil  = Image.open(cloth_image.file).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # Save temp files
    with tempfile.NamedTemporaryFile(suffix=".png") as p_tmp, tempfile.NamedTemporaryFile(suffix=".png") as c_tmp:
        person_pil.save(p_tmp.name)
        cloth_pil.save(c_tmp.name)
        person_dict = {"background": p_tmp.name, "layers": [p_tmp.name]}

        # Invoke dynamic batching
        result_img = await submit_request(
            person_dict,
            c_tmp.name,
            cloth_type.value,
            num_inference_steps,
            guidance_scale,
            seed,
            show_type.value
        )

    # Stream back PNG
    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")