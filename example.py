import os
import io
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from trellis.utils.postprocessing_utils import simplify_gs
import imageio
import subprocess
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['ATTN_BACKEND'] = 'xformers'
os.environ['SPCONV_ALGO'] = 'native'
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9'

app = FastAPI()
@app.get("/")
async def root():
    return "root"

@app.post("/upload")
async def generate_model(img: UploadFile = File(...), email: str = Form(...)):
    try:
        # 이미지 파일을 메모리에서 바로 읽기
        image_bytes = await img.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 이메일을 이용해 고유 파일명 생성
        file_stem = email.replace("@", "_").replace(".", "_")

        print("GenerateModel 시작")
        start_time = time.time()

        generated_glb_path, generated_video_path = await GenerateModel(image, email, start_time)
        print("GenerateModel 완료:", generated_glb_path)

        # Blender 자동 실행
        # if os.path.exists(generated_glb_path):
        #     blender_cmd = [
        #         "/opt/blender/blender", "--background", "--python", "trellis/bpy_test.py", "--",
        #         email
        #     ]
        #     subprocess.run(blender_cmd, check=True)

        if os.path.exists(generated_glb_path) and os.path.exists(generated_video_path):
            return FileResponse(
                path=generated_glb_path,
                filename=f"{email}.glb",
                media_type='application/octet-stream',
                headers={"X-Video-Path": generated_video_path}
            )
        else:
            return JSONResponse(
                status_code=500,
                content={"error": "GLB or video file not found after generation"}
            )

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


async def GenerateModel(pil_image: Image.Image, user_email: str, start_time: float):
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils import render_utils, postprocessing_utils

    user_dir = f"C:/Users/hyoun/TRELLIS/glb_models/{user_email}"
    os.makedirs(user_dir, exist_ok=True)

    pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    pipeline.cuda()

    outputs = pipeline.run(
        pil_image,
        seed=20,
        sparse_structure_sampler_params={"steps": 12, "cfg_strength": 7.5},
        slat_sampler_params={"steps": 12, "cfg_strength": 3.5},
    )

    video = render_utils.render_video(outputs['gaussian'][0])['color']
    mesh = outputs['mesh'][0]
    print(f"Mesh vertices: {len(mesh.vertices)}")
    print(f"Mesh faces: {len(mesh.faces)}")
    print(f"모델 생성 시간 : {((time.time() - start_time)/60):.2f} 분")

    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        simplify=0.8,
        texture_size=1024,
    )
    print(f"모델 변환 시간 into *.glb: {((time.time() - start_time)/60):.2f} 분")

    glb_path = f"{user_dir}/{user_email}.glb"
    video_path = f"{user_dir}/{user_email}_v.mp4"
    glb.export(glb_path)
    imageio.mimsave(video_path, video, fps=30)

    return glb_path, video_path

