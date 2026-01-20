import os
import torch
import tempfile
import soundfile as sf
import asyncio
from typing import Optional
from fastapi import FastAPI, Form, UploadFile, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from concurrent.futures import ThreadPoolExecutor

import voxcpm
import uvicorn

os.environ["TOKENIZERS_PARALLELISM"] = "false"
if os.environ.get("HF_REPO_ID", "").strip() == "":
    os.environ["HF_REPO_ID"] = "/models/Voice/VoxCPM/VoxCPM1.5/"


# ========== 模型类 ==========
class VoxCPMDemo:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Running on device: {self.device}")
        self.voxcpm_model: Optional[voxcpm.VoxCPM] = None
        self.default_local_model_dir = "/models/Voice/VoxCPM/VoxCPM1.5/"

    def _resolve_model_dir(self) -> str:
        if os.path.isdir(self.default_local_model_dir):
            return self.default_local_model_dir
        return "models"

    def get_or_load_voxcpm(self) -> voxcpm.VoxCPM:
        if self.voxcpm_model is None:
            print("🔄 Loading VoxCPM model...")
            model_dir = self._resolve_model_dir()
            self.voxcpm_model = voxcpm.VoxCPM(voxcpm_model_path=model_dir)
            print("✅ VoxCPM model loaded.")
        return self.voxcpm_model

    def tts_generate(
        self,
        text: str,
        prompt_wav_path: Optional[str] = None,
        prompt_text: Optional[str] = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        do_normalize: bool = True,
        denoise: bool = True,
    ) -> str:
        model = self.get_or_load_voxcpm()
        wav = model.generate(
            text=text,
            prompt_text=prompt_text,
            prompt_wav_path=prompt_wav_path,
            cfg_value=float(cfg_value),
            inference_timesteps=int(inference_timesteps),
            normalize=do_normalize,
            denoise=denoise,
        )
        tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(tmp_wav.name, wav, 44100)
        torch.cuda.empty_cache()
        return tmp_wav.name


# ========== FastAPI ==========
app = FastAPI(title="VoxCPM API", version="1.0.0")
demo = VoxCPMDemo()

executor = ThreadPoolExecutor(max_workers=2)  # CPU线程池，执行GPU任务
gpu_queue = asyncio.Queue()  # GPU并发队列
MAX_GPU_CONCURRENT = 1  # 单GPU同时最多1个任务，可以调整


# GPU队列消费者协程
async def gpu_worker():
    while True:
        future, func, args, kwargs = await gpu_queue.get()
        loop = asyncio.get_running_loop()
        try:
            # 在线程池中执行GPU任务
            result = await loop.run_in_executor(executor, func, *args, **kwargs)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
        finally:
            gpu_queue.task_done()


# 启动GPU消费者协程
async def start_gpu_workers():
    for _ in range(MAX_GPU_CONCURRENT):
        asyncio.create_task(gpu_worker())


@app.on_event("startup")
async def startup_event():
    await start_gpu_workers()


# 封装GPU队列任务
async def submit_to_gpu(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    await gpu_queue.put((future, func, args, kwargs))
    return await future


# ---------- TTS API ----------
@app.post("/generate_tts")
async def generate_tts(
    background_tasks: BackgroundTasks,
    text: str = Form(...),
    prompt_text: Optional[str] = Form(None),
    cfg_value: float = Form(2.0),
    inference_timesteps: int = Form(10),
    do_normalize: bool = Form(True),
    denoise: bool = Form(True),
    prompt_wav: Optional[UploadFile] = None,
):
    try:
        prompt_path = None
        if prompt_wav:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(await prompt_wav.read())
                prompt_path = tmp.name
                background_tasks.add_task(os.remove, tmp.name)

        # 提交到GPU队列
        output_path = await submit_to_gpu(
            demo.tts_generate,
            text,
            prompt_path,
            prompt_text,
            cfg_value,
            inference_timesteps,
            do_normalize,
            denoise
        )

        # 后台删除生成的文件
        background_tasks.add_task(os.remove, output_path)

        return StreamingResponse(
            open(output_path, "rb"),
            media_type="audio/wav",
            headers={"Content-Disposition": 'attachment; filename="output.wav"'}
        )

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/")
async def root():
    return {"message": "VoxCPM API running 🚀", "endpoints": ["/generate_tts"]}

if __name__ == "__main__":
    uvicorn.run("api_concurrent:app", host="0.0.0.0", port=5000, workers=8)