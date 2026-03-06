"""
Module 8: Aspect Ratio Module
Smart crop/resize using OpenCV face detection for subject centering.
"""
import os
import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from config import settings
from database import get_db
from models import Timeline, Clip, Project

router = APIRouter()

RATIO_SIZES = {
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "1:1": (720, 720),
    "4:5": (720, 900),
}


class AspectRatioRequest(BaseModel):
    ratio: str


def smart_crop_frame(frame_path: str, target_w: int, target_h: int, output_path: str) -> bool:
    import cv2
    img = cv2.imread(frame_path)
    if img is None:
        return False

    h, w = img.shape[:2]
    target_ratio = target_w / target_h
    src_ratio = w / h

    if src_ratio > target_ratio:
        new_w = int(h * target_ratio)
        cx = w // 2
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            face_cascade = cv2.CascadeClassifier(cascade_path)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                largest = max(faces, key=lambda r: r[2] * r[3])
                fx, fy, fw, fh = largest
                cx = fx + fw // 2
        except Exception:
            pass
        x1 = max(0, min(cx - new_w // 2, w - new_w))
        crop = img[:, x1:x1 + new_w]
    else:
        new_h = int(w / target_ratio)
        crop = img[:new_h, :]

    resized = cv2.resize(crop, (target_w, target_h))
    cv2.imwrite(output_path, resized)
    return True


# IMPORTANT: /ratios route MUST be defined BEFORE /{project_id}
# Otherwise FastAPI matches "ratios" as a project_id integer and returns 422
@router.get("/aspect/ratios")
async def list_ratios():
    """List available aspect ratios."""
    return [
        {
            "ratio": r,
            "width": w,
            "height": h,
            "name": {
                "16:9": "YouTube Landscape",
                "9:16": "Shorts / TikTok",
                "1:1": "Instagram Square",
                "4:5": "Instagram Reels"
            }.get(r, r)
        }
        for r, (w, h) in RATIO_SIZES.items()
    ]


@router.post("/aspect/{project_id}")
async def set_aspect_ratio(
    project_id: int,
    request: AspectRatioRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Set the aspect ratio for the project timeline."""
    if request.ratio not in RATIO_SIZES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid ratio. Choose from: {list(RATIO_SIZES.keys())}"
        )

    tl_result = await db.execute(
        select(Timeline).where(Timeline.project_id == project_id, Timeline.is_current == True)
    )
    timeline = tl_result.scalar_one_or_none()
    if not timeline:
        raise HTTPException(status_code=404, detail="Timeline not found. Run Visual Match first.")

    timeline.aspect_ratio = request.ratio
    await db.commit()

    target_w, target_h = RATIO_SIZES[request.ratio]
    background_tasks.add_task(crop_all_keyframes, project_id, target_w, target_h)

    return {
        "message": f"Aspect ratio set to {request.ratio}",
        "target_size": RATIO_SIZES[request.ratio]
    }


@router.get("/aspect/{project_id}")
async def get_aspect_ratio(project_id: int, db: AsyncSession = Depends(get_db)):
    """Get current aspect ratio for a project."""
    tl_result = await db.execute(
        select(Timeline).where(Timeline.project_id == project_id, Timeline.is_current == True)
    )
    timeline = tl_result.scalar_one_or_none()
    if not timeline:
        raise HTTPException(status_code=404, detail="Timeline not found")
    return {
        "ratio": timeline.aspect_ratio or "16:9",
        "target_size": RATIO_SIZES.get(timeline.aspect_ratio or "16:9", (1280, 720))
    }


async def crop_all_keyframes(project_id: int, target_w: int, target_h: int):
    """Crop all keyframes in the project to target aspect ratio."""
    from database import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        clips_result = await db.execute(
            select(Clip).where(Clip.project_id == project_id)
        )
        clips = clips_result.scalars().all()

        loop = asyncio.get_event_loop()
        for clip in clips:
            safe_keyframe = clip.keyframe_path.replace('\\', '/') if clip.keyframe_path else None
            if safe_keyframe and os.path.exists(safe_keyframe):
                cropped_dir = Path(safe_keyframe).parent / "cropped"
                cropped_dir.mkdir(parents=True, exist_ok=True)
                cropped_path = str(cropped_dir / Path(safe_keyframe).name)
                await loop.run_in_executor(
                    None, smart_crop_frame,
                    safe_keyframe, target_w, target_h, cropped_path
                )