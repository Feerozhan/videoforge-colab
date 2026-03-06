"""
Module 1: Video Input
Handles file upload, FFmpeg extraction of audio/metadata, frame sampling.
"""
import os
import json
import subprocess
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from config import settings
from database import get_db
from models import Project, Timeline

router = APIRouter()

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def get_video_metadata(video_path: str) -> dict:
    """Use ffprobe to extract video metadata."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        video_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        data = json.loads(result.stdout)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ffprobe error: {e}")

    metadata = {"duration": None, "fps": None, "width": None, "height": None, "file_size": None}
    fmt = data.get("format", {})
    metadata["duration"] = float(fmt.get("duration", 0) or 0)
    metadata["file_size"] = int(fmt.get("size", 0) or 0)

    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            metadata["width"] = stream.get("width")
            metadata["height"] = stream.get("height")
            fps_str = stream.get("r_frame_rate", "0/1")
            try:
                num, den = fps_str.split("/")
                metadata["fps"] = round(int(num) / int(den), 2)
            except Exception:
                metadata["fps"] = 0.0
            break
    return metadata


def extract_audio(video_path: str, audio_dir: str, project_id: int) -> str:
    """Extract audio from video using FFmpeg."""
    audio_path = os.path.join(audio_dir, f"project_{project_id}_audio.wav")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        audio_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Audio extraction failed: {result.stderr}")
    return audio_path


@router.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """Upload a video file, extract metadata and audio."""
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {ALLOWED_EXTENSIONS}"
        )

    # Ensure directories exist
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = Path(settings.audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Save file temporarily to get ID
    temp_project = Project(
        name=Path(file.filename).stem,
        original_filename=file.filename,
        video_path="",
        status="uploading"
    )
    db.add(temp_project)
    await db.commit()
    await db.refresh(temp_project)

    # Save video with project ID in filename
    safe_name = f"project_{temp_project.id}{ext}"
    video_path = upload_dir / safe_name
    contents = await file.read()
    with open(video_path, "wb") as f:
        f.write(contents)

    # Extract metadata
    metadata = get_video_metadata(str(video_path))

    # Extract audio
    try:
        audio_path = extract_audio(str(video_path), str(audio_dir), temp_project.id)
    except Exception as e:
        audio_path = None

    # Update project record
    temp_project.video_path = str(video_path)
    temp_project.audio_path = audio_path
    temp_project.duration = metadata["duration"]
    temp_project.fps = metadata["fps"]
    temp_project.width = metadata["width"]
    temp_project.height = metadata["height"]
    temp_project.file_size = metadata["file_size"]
    temp_project.status = "uploaded"
    await db.commit()
    await db.refresh(temp_project)

    return {
        "project_id": temp_project.id,
        "name": temp_project.name,
        "filename": temp_project.original_filename,
        "duration": temp_project.duration,
        "fps": temp_project.fps,
        "width": temp_project.width,
        "height": temp_project.height,
        "file_size": temp_project.file_size,
        "status": temp_project.status,
        "video_url": f"/storage/uploads/{safe_name}",
    }


@router.get("/projects")
async def list_projects(db: AsyncSession = Depends(get_db)):
    """List all projects."""
    result = await db.execute(select(Project).order_by(Project.created_at.desc()))
    projects = result.scalars().all()
    return [
        {
            "id": p.id,
            "name": p.name,
            "filename": p.original_filename,
            "duration": p.duration,
            "status": p.status,
            "created_at": p.created_at.isoformat() if p.created_at else None,
        }
        for p in projects
    ]


@router.get("/projects/{project_id}")
async def get_project(project_id: int, db: AsyncSession = Depends(get_db)):
    """Get a single project by ID with step completion flags."""
    from sqlalchemy.orm import selectinload
    result = await db.execute(
        select(Project)
        .options(
            selectinload(Project.transcript),
            selectinload(Project.timelines).selectinload(Timeline.clips)
        )
        .where(Project.id == project_id)
    )
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    has_transcript = project.transcript is not None and bool(project.transcript.segments)
    has_summary = project.transcript is not None and bool(project.transcript.summary_text)
    
    current_timeline = next((t for t in project.timelines if t.is_current), None)
    has_tts = current_timeline is not None and bool(current_timeline.tts_audio_path)
    has_match = current_timeline is not None and len(current_timeline.clips) > 0
    has_draft = current_timeline is not None and bool(current_timeline.draft_video_path)
    has_export = current_timeline is not None and bool(current_timeline.export_video_path)

    return {
        "id": project.id,
        "name": project.name,
        "filename": project.original_filename,
        "duration": project.duration,
        "fps": project.fps,
        "width": project.width,
        "height": project.height,
        "file_size": project.file_size,
        "status": project.status,
        "created_at": project.created_at.isoformat() if project.created_at else None,
        "has_transcript": has_transcript,
        "has_summary": has_summary,
        "has_tts": has_tts,
        "has_match": has_match,
        "has_draft": has_draft,
        "has_export": has_export,
    }


@router.delete("/projects/{project_id}")
async def delete_project(project_id: int, db: AsyncSession = Depends(get_db)):
    """Delete a project and its files."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Delete files
    for path in [project.video_path, project.audio_path]:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass

    await db.delete(project)
    await db.commit()
    return {"message": "Project deleted"}
