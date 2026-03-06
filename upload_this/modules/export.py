"""
Module 9: Export Module
FFmpeg-based high-quality video export with GPU acceleration (NVIDIA NVENC).
Falls back to CPU if GPU not available.
"""
import os
import asyncio
import subprocess
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from config import settings
from database import get_db
from models import Project, Timeline

router = APIRouter()

_export_progress: dict[int, dict] = {}

RESOLUTIONS = {
    "480p":  (854, 480),
    "720p":  (1280, 720),
    "1080p": (1920, 1080),
    "2k":    (2560, 1440),
    "4k":    (3840, 2160),
}

BITRATE_PRESETS = {
    "low":    {"video": "1M",  "audio": "96k",  "preset": "ultrafast"},
    "medium": {"video": "4M",  "audio": "128k", "preset": "ultrafast"},
    "high":   {"video": "8M",  "audio": "192k", "preset": "ultrafast"},
}


class ExportRequest(BaseModel):
    resolution: str = "1080p"
    bitrate: str = "medium"
    format: str = "mp4"


def do_ffmpeg_export(
    input_path: str,
    output_path: str,
    width: int,
    height: int,
    video_bitrate: str,
    audio_bitrate: str,
    preset: str = "ultrafast"
):
    """
    Try NVIDIA GPU encoding first (h264_nvenc).
    Falls back to CPU (libx264) if GPU fails.
    """
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
    )

    # --- GPU command (NVIDIA NVENC) ---
    cmd_gpu = [
        "ffmpeg", "-y",
        "-hwaccel", "cuda",
        "-i", input_path,
        "-vf", vf,
        "-c:v", "h264_nvenc",
        "-preset", "p1",        # p1 = fastest NVENC preset
        "-b:v", video_bitrate,
        "-c:a", "aac",
        "-b:a", audio_bitrate,
        "-movflags", "+faststart",
        output_path
    ]

    # --- CPU fallback command ---
    cmd_cpu = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", preset,
        "-b:v", video_bitrate,
        "-c:a", "aac",
        "-b:a", audio_bitrate,
        "-movflags", "+faststart",
        output_path
    ]

    # Try GPU first
    try:
        result = subprocess.run(cmd_gpu, capture_output=True, text=True, timeout=3600)
        if result.returncode == 0:
            return  # GPU encoding success!
    except Exception:
        pass

    # CPU fallback
    result = subprocess.run(cmd_cpu, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg export failed: {result.stderr[-1000:]}")


async def run_export(project_id: int, resolution: str, bitrate: str):
    """Background export task."""
    from database import AsyncSessionLocal

    _export_progress[project_id] = {"step": "starting", "progress": 0}

    async with AsyncSessionLocal() as db:
        tl_result = await db.execute(
            select(Timeline).where(Timeline.project_id == project_id, Timeline.is_current == True)
        )
        timeline = tl_result.scalar_one_or_none()
        if not timeline or not timeline.draft_video_path:
            _export_progress[project_id] = {"step": "error", "error": "Draft video not found. Run Auto Edit first."}
            return

        if not os.path.exists(timeline.draft_video_path):
            _export_progress[project_id] = {"step": "error", "error": "Draft video file missing on disk."}
            return

        w, h = RESOLUTIONS.get(resolution, (1280, 720))
        brates = BITRATE_PRESETS.get(bitrate, BITRATE_PRESETS["medium"])

        exports_dir = Path(settings.exports_dir)
        exports_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(exports_dir / f"project_{project_id}_{resolution}_{bitrate}.mp4")

        _export_progress[project_id] = {"step": "encoding", "progress": 10}

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, do_ffmpeg_export,
                timeline.draft_video_path,
                output_path,
                w, h,
                brates["video"],
                brates["audio"],
                brates["preset"],
            )

            timeline.export_video_path = output_path
            p_result = await db.execute(select(Project).where(Project.id == project_id))
            project = p_result.scalar_one_or_none()
            if project:
                project.status = "exported"
            await db.commit()

            _export_progress[project_id] = {"step": "done", "progress": 100, "output": output_path}

        except Exception as e:
            _export_progress[project_id] = {"step": "error", "progress": 0, "error": str(e)}
            raise


@router.post("/export/{project_id}")
async def start_export(
    project_id: int,
    request: ExportRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Start the final export pipeline."""
    if request.resolution not in RESOLUTIONS:
        raise HTTPException(status_code=400, detail=f"Invalid resolution. Choose: {list(RESOLUTIONS.keys())}")
    if request.bitrate not in BITRATE_PRESETS:
        raise HTTPException(status_code=400, detail=f"Invalid bitrate. Choose: {list(BITRATE_PRESETS.keys())}")

    current = _export_progress.get(project_id, {})
    if current.get("step") == "encoding":
        raise HTTPException(status_code=409, detail="Export already in progress.")

    background_tasks.add_task(run_export, project_id, request.resolution, request.bitrate)
    return {
        "message": "Export started",
        "project_id": project_id,
        "resolution": request.resolution,
        "bitrate": request.bitrate,
        "target_size": RESOLUTIONS[request.resolution],
    }


@router.get("/export/{project_id}/status")
async def get_export_status(project_id: int):
    """Poll export progress."""
    return _export_progress.get(project_id, {"step": "not_started", "progress": 0})


@router.get("/export/{project_id}/download")
async def download_export(project_id: int, db: AsyncSession = Depends(get_db)):
    """Download the exported MP4."""
    tl_result = await db.execute(
        select(Timeline).where(Timeline.project_id == project_id, Timeline.is_current == True)
    )
    timeline = tl_result.scalar_one_or_none()
    if not timeline or not timeline.export_video_path or not os.path.exists(timeline.export_video_path):
        raise HTTPException(status_code=404, detail="Exported video not ready")
    return FileResponse(
        timeline.export_video_path,
        media_type="video/mp4",
        filename=Path(timeline.export_video_path).name,
        headers={"Content-Disposition": f"attachment; filename={Path(timeline.export_video_path).name}"}
    )