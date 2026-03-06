"""
FastAPI main application entry point.
All routers are mounted here, CORS configured, and DB initialized on startup.
"""
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import settings
from database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_directories()
    await init_db()
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "CPU"
    except Exception:
        cuda_ok, gpu_name = False, "CPU"
    hw_label = f"CUDA ({gpu_name})" if cuda_ok else "CPU (Intel QSV/CPU fallback)"
    print(f"✅ DB ready | Device: {settings.device} | Whisper: {settings.whisper_model} | HW: {hw_label}")

    # Pre-warm faster-whisper model so the first analysis request doesn't hang on download
    try:
        from faster_whisper import WhisperModel
        print(f"[Startup] Pre-loading faster-whisper '{settings.whisper_model}' model...")
        WhisperModel(settings.whisper_model, device="cpu", compute_type="int8")
        print(f"[Startup] faster-whisper '{settings.whisper_model}' ready ✅")
    except Exception as e:
        print(f"[Startup] faster-whisper pre-load skipped: {e}")

    yield


app = FastAPI(
    title="AI Video Summarizer API",
    description="Fully local AI-powered video summarization and editing API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

storage_path = Path(settings.storage_dir)
storage_path.mkdir(parents=True, exist_ok=True)
app.mount("/storage", StaticFiles(directory=str(storage_path)), name="storage")

from modules.video_input import router as video_router
from modules.analysis import router as analysis_router
from modules.summarization import router as summarization_router
from modules.tts import router as tts_router
from modules.visual_match import router as match_router
from modules.auto_editor import router as auto_edit_router
from modules.manual_editor import router as manual_edit_router
from modules.aspect_ratio import router as aspect_ratio_router
from modules.export import router as export_router

app.include_router(video_router, prefix="/api", tags=["Video Input"])
app.include_router(analysis_router, prefix="/api", tags=["Analysis"])
app.include_router(summarization_router, prefix="/api", tags=["Summarization"])
app.include_router(tts_router, prefix="/api", tags=["TTS"])
app.include_router(match_router, prefix="/api", tags=["Visual Match"])
app.include_router(auto_edit_router, prefix="/api", tags=["Auto Editor"])
app.include_router(manual_edit_router, prefix="/api", tags=["Manual Editor"])
app.include_router(aspect_ratio_router, prefix="/api", tags=["Aspect Ratio"])
app.include_router(export_router, prefix="/api", tags=["Export"])


@app.get("/api/health")
async def health():
    return {"status": "ok", "device": settings.device, "whisper_model": settings.whisper_model}



@app.get("/api/gpu-info")
async def gpu_info():
    import subprocess
    info = {
        "cuda_available": False,
        "gpu_name": None,
        "vram_gb": None,
        "cuda_version": None,
        "intel_qsv": False,
        "faster_whisper": False,
        "device": settings.device,
        "whisper_model": settings.whisper_model,
    }
    try:
        import torch
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
            vram = torch.cuda.get_device_properties(0).total_memory
            info["vram_gb"] = round(vram / (1024 ** 3), 1)
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=5
        )
        info["intel_qsv"] = "h264_qsv" in result.stdout
    except Exception:
        pass
    try:
        import faster_whisper
        info["faster_whisper"] = True
    except ImportError:
        pass
    if info["cuda_available"]:
        info["label"] = f"🟢 GPU: {info['gpu_name']}"
    elif info["intel_qsv"]:
        info["label"] = "🟡 Intel QuickSync + CPU"
    else:
        info["label"] = "⚪ CPU Only"
    return info


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=True)