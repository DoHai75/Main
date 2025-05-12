import torch
from PIL import Image
import os
from datetime import datetime
import numpy as np

# Đảm bảo các hàm này đã được import từ app.py
from app import submit_function, resize_and_crop, resize_and_padding, automasker, mask_processor, pipeline, vis_mask, image_grid

# Đường dẫn tới ảnh người và ảnh quần áo để test
person_image_path = "/home/ubuntu/CVT_main/OIP (4).jpg"  # Thay bằng đường dẫn tới ảnh người thật
cloth_image_path = "/home/ubuntu/CVT_main/OIP.jpg"    # Thay bằng đường dẫn tới ảnh quần áo thật

# Đảm bảo thư mục output tồn tại
output_dir = "test_output"
os.makedirs(output_dir, exist_ok=True)

# Xóa cache bộ nhớ GPU
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Đặt chế độ precision (fp16 để tối ưu VRAM)
torch.set_default_dtype(torch.float16)

# Đầu vào cho hàm submit_function
person_image = {
    "background": person_image_path,
    "layers": [person_image_path]
}

cloth_image = cloth_image_path
cloth_type = "upper"
num_inference_steps = 20
guidance_scale = 2.5
seed = 42
show_type = "result only"

try:
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        result_image = submit_function(
            person_image,
            cloth_image,
            cloth_type,
            num_inference_steps,
            guidance_scale,
            seed,
            show_type
        )
except torch.cuda.OutOfMemoryError:
    print("Lỗi bộ nhớ: Đang thử lại với kích thước nhỏ hơn.")
    torch.cuda.empty_cache()

    person_image = resize_and_crop(Image.open(person_image_path).convert("RGB"), (512, 512))
    cloth_image = resize_and_padding(Image.open(cloth_image_path).convert("RGB"), (512, 512))

    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        result_image = submit_function(
            {"background": person_image, "layers": [person_image]},
            cloth_image,
            cloth_type,
            num_inference_steps,
            guidance_scale,
            seed,
            show_type
        )

# Lưu kết quả
result_save_path = os.path.join(output_dir, "test_result_optimized.png")
result_image.save(result_save_path)
print(f"Kết quả đã được lưu tại: {result_save_path}")