import os
import torch
import tempfile
import soundfile as sf
import numpy as np
import wave
from io import BytesIO
import asyncio
from typing import Optional
from fastapi import FastAPI, Form, UploadFile, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from concurrent.futures import ThreadPoolExecutor

import voxcpm
import uvicorn

os.environ["TOKENIZERS_PARALLELISM"] = "false"
if os.environ.get("HF_REPO_ID", "").strip() == "":
    os.environ["HF_REPO_ID"] = "/models/VoxCPM1.5/"


# ========== 模型类 ==========
class VoxCPMDemo:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Running on device: {self.device}")
        self.voxcpm_model: Optional[voxcpm.VoxCPM] = None
        self.default_local_model_dir = "/models/VoxCPM1.5/"

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
        retry_badcase: bool = True,
        retry_badcase_max_times: int = 3,
        retry_badcase_ratio_threshold: float = 6.0,
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
            retry_badcase=retry_badcase,
            retry_badcase_max_times=retry_badcase_max_times,
            retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
        )
        tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(tmp_wav.name, wav, model.tts_model.sample_rate)
        torch.cuda.empty_cache()
        return tmp_wav.name

    def tts_generate_streaming(
        self,
        text: str,
        prompt_wav_path: Optional[str] = None,
        prompt_text: Optional[str] = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        do_normalize: bool = True,
        denoise: bool = True,
        retry_badcase: bool = True,
        retry_badcase_max_times: int = 3,
        retry_badcase_ratio_threshold: float = 6.0,
    ):
        """Generates audio and yields it as a stream of WAV chunks."""
        model = self.get_or_load_voxcpm()

        # 1. Yield a WAV header first.
        # The size fields will be 0, which is standard for streaming.
        SAMPLE_RATE = model.tts_model.sample_rate
        CHANNELS = 1
        SAMPLE_WIDTH = 2  # 16-bit

        header_buf = BytesIO()
        with wave.open(header_buf, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)

        yield header_buf.getvalue()

        # 2. Generate and yield audio chunks.
        # NOTE: We assume a `generate_stream` method exists on the model that yields audio chunks.
        # You may need to change `generate_stream` to the actual method name in your version of voxcpm.
        try:
            stream = model.generate_streaming(
                text=text,
                prompt_text=prompt_text,
                prompt_wav_path=prompt_wav_path,
                cfg_value=float(cfg_value),
                inference_timesteps=int(inference_timesteps),
                normalize=do_normalize,
                denoise=denoise,
                retry_badcase=retry_badcase,
                retry_badcase_max_times=retry_badcase_max_times,
                retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
            )
            for chunk_np in stream:  # Assuming it yields numpy arrays
                # Ensure audio is in 16-bit PCM format for streaming
                if chunk_np.dtype in [np.float32, np.float64]:
                    chunk_np = (chunk_np * 32767).astype(np.int16)
                yield chunk_np.tobytes()
        finally:
            torch.cuda.empty_cache()


# ========== FastAPI ==========
app = FastAPI(title="VoxCPM API", version="1.0.0")
demo = VoxCPMDemo()

# --- Concurrency Control ---
# Use a semaphore to limit concurrent GPU tasks.
MAX_GPU_CONCURRENT = int(os.environ.get("MAX_GPU_CONCURRENT", "1"))
gpu_semaphore = asyncio.Semaphore(MAX_GPU_CONCURRENT)

# Use a thread pool for running blocking (CPU/GPU-bound) code.
executor = ThreadPoolExecutor(max_workers=2)

@app.on_event("shutdown")
def shutdown_event():
    print("Shutting down thread pool executor...")
    executor.shutdown(wait=True)


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
    retry_badcase: bool = Form(True),
    retry_badcase_max_times: int = Form(3),
    retry_badcase_ratio_threshold: float = Form(6.0),
    prompt_wav: Optional[UploadFile] = None,
):
    try:
        prompt_path = None
        if prompt_wav:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(await prompt_wav.read())
                prompt_path = tmp.name
                background_tasks.add_task(os.remove, tmp.name)

        # Submit to GPU via semaphore and executor
        loop = asyncio.get_running_loop()
        async with gpu_semaphore:
            output_path = await loop.run_in_executor(
                executor,
                demo.tts_generate,
                text,
                prompt_path,
                prompt_text,
                cfg_value,
                inference_timesteps,
                do_normalize,
                denoise,
                retry_badcase,
                retry_badcase_max_times,
                retry_badcase_ratio_threshold,
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


@app.post("/generate_tts_streaming")
async def generate_tts_streaming(
    background_tasks: BackgroundTasks,
    text: str = Form(...),
    prompt_text: Optional[str] = Form(None),
    cfg_value: float = Form(2.0),
    inference_timesteps: int = Form(10),
    do_normalize: bool = Form(True),
    denoise: bool = Form(True),
    retry_badcase: bool = Form(True),
    retry_badcase_max_times: int = Form(3),
    retry_badcase_ratio_threshold: float = Form(6.0),
    prompt_wav: Optional[UploadFile] = None,
):
    try:
        prompt_path = None
        if prompt_wav:
            # Save uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(await prompt_wav.read())
                prompt_path = tmp.name
                # Ensure the temp file is deleted after the request is finished
                background_tasks.add_task(os.remove, prompt_path)

        async def stream_generator():
            # This async generator consumes from a queue populated by a sync generator in a thread.
            q = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def producer():
                # This runs in the executor thread and produces chunks.
                try:
                    # This is a sync generator
                    for chunk in demo.tts_generate_streaming(
                        text, prompt_path, prompt_text, cfg_value,
                        inference_timesteps, do_normalize, denoise,
                        retry_badcase, retry_badcase_max_times, retry_badcase_ratio_threshold
                    ):
                        loop.call_soon_threadsafe(q.put_nowait, chunk)
                except Exception as e:
                    # Put the exception in the queue to be re-raised in the consumer
                    loop.call_soon_threadsafe(q.put_nowait, e)
                finally:
                    # Signal the end of the stream
                    loop.call_soon_threadsafe(q.put_nowait, None)
            
            # Acquire the GPU semaphore before starting the producer thread.
            async with gpu_semaphore:
                loop.run_in_executor(executor, producer)
                while True:
                    chunk = await q.get()
                    if chunk is None:
                        break
                    if isinstance(chunk, Exception):
                        raise chunk
                    yield chunk
        
        return StreamingResponse(stream_generator(), media_type="audio/wav")

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/")
async def root():
    return {"message": "VoxCPM API running 🚀", "endpoints": ["/generate_tts"]}

if __name__ == "__main__":
    uvicorn.run("api_concurrent:app", host="0.0.0.0", port=5000, workers=4)