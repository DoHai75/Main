from enum import Enum
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import torch, io, tempfile
from model.cloth_masker import AutoMasker, vis_mask       
from app import submit_function
from utils import init_weight_dtype, resize_and_crop, resize_and_padding
from fastapi.concurrency import run_in_threadpool  

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
    # 1. Load images into PIL
    try:
        person_pil = Image.open(person_image.file).convert("RGB")
        cloth_pil  = Image.open(cloth_image.file).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # 2. Set dtype
    if mixed_precision == MixedPrecision.fp16:
        dtype = torch.float16
    elif mixed_precision == MixedPrecision.bf16:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    torch.set_default_dtype(dtype)

    # 3. Save to temp files
    with tempfile.NamedTemporaryFile(suffix=".png") as p_tmp, tempfile.NamedTemporaryFile(suffix=".png") as c_tmp:
        person_pil.save(p_tmp.name); cloth_pil.save(c_tmp.name)
        person_dict = {"background": p_tmp.name, "layers": [p_tmp.name]}

        def do_inference():
            with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
                return submit_function(
                    person_dict,
                    c_tmp.name,
                    cloth_type.value,
                    num_inference_steps,
                    guidance_scale,
                    seed,
                    show_type.value
                )

        result_img = await run_in_threadpool(do_inference)  

        '''
        # 4. Run inference
        with torch.cuda.amp.autocast(enabled=use_fp16, dtype=dtype):
            result_img = submit_function(
                person_dict,
                c_tmp.name,
                cloth_type.value,
                num_inference_steps,
                guidance_scale,
                seed,
                show_type.value
            )
        '''

    # 5. Stream back PNG
    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")