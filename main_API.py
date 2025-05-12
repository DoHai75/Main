# main.py

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import torch, io, tempfile
from model.cloth_masker import AutoMasker, vis_mask       # ví dụ nếu bạn giữ model/ ở gốc project
from app import submit_function
from utils import init_weight_dtype, resize_and_crop, resize_and_padding
app = FastAPI(title="CatVTON Try-On API")

@app.post("/tryon/")
async def tryon(
    person_image: UploadFile = File(...),
    cloth_image: UploadFile = File(...),
    cloth_type: str = Form(..., regex="^(upper|lower|overall)$"),
    num_inference_steps: int = Form(20),
    guidance_scale: float = Form(2.5),
    seed: int = Form(42),
    show_type: str = Form(..., regex="^(result only|all)$"),
    mixed_precision: str = Form("fp16", regex="^(fp16|fp32)$")
):
    # 1. Load images into PIL
    try:
        person_pil = Image.open(person_image.file).convert("RGB")
        cloth_pil  = Image.open(cloth_image.file).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # 2. Set dtype
    use_fp16 = mixed_precision == "fp16"
    dtype = torch.float16 if use_fp16 else torch.float32
    torch.set_default_dtype(dtype)

    # 3. Save to temp files
    with tempfile.NamedTemporaryFile(suffix=".png") as p_tmp, tempfile.NamedTemporaryFile(suffix=".png") as c_tmp:
        person_pil.save(p_tmp.name); cloth_pil.save(c_tmp.name)
        person_dict = {"background": p_tmp.name, "layers": [p_tmp.name]}

        # 4. Run inference
        with torch.cuda.amp.autocast(enabled=use_fp16, dtype=dtype):
            result_img = submit_function(
                person_dict,
                c_tmp.name,
                cloth_type,
                num_inference_steps,
                guidance_scale,
                seed,
                show_type
            )

    # 5. Stream back PNG
    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
