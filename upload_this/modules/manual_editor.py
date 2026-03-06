"""
Module 7: Manual Editing
Timeline mutation endpoints for split, trim, delete, insert, reorder, adjust, filter.
"""
import json
import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from database import get_db
from models import Timeline, Clip, Project

router = APIRouter()


# ---- Request models ----

class SplitRequest(BaseModel):
    clip_id: int
    split_at: float  # seconds from beginning of clip

class TrimRequest(BaseModel):
    clip_id: int
    new_start: float
    new_end: float

class DeleteRequest(BaseModel):
    clip_id: int

class InsertClipRequest(BaseModel):
    project_id: int
    insert_at_index: int
    source_start: float
    source_end: float
    duration: float

class ReorderRequest(BaseModel):
    clip_ids: List[int]  # ordered list of all clip IDs in new order

class AdjustRequest(BaseModel):
    clip_id: int
    brightness: float = 0    # -100 to 100
    contrast: float = 0
    saturation: float = 0
    sharpness: float = 0

class FilterRequest(BaseModel):
    clip_id: int
    filter_name: str  # none, grayscale, warm, cool, vintage

class ReplaceAudioRequest(BaseModel):
    audio_url: str


# ---- Helper ----

async def get_timeline_for_project(project_id: int, db: AsyncSession) -> Timeline:
    result = await db.execute(
        select(Timeline).where(Timeline.project_id == project_id, Timeline.is_current == True)
    )
    timeline = result.scalar_one_or_none()
    if not timeline:
        raise HTTPException(status_code=404, detail="Timeline not found")
    return timeline


async def reindex_clips(timeline_id: int, db: AsyncSession):
    """Re-compute clip_index and timeline_start/end after mutations."""
    result = await db.execute(
        select(Clip).where(Clip.timeline_id == timeline_id).order_by(Clip.clip_index)
    )
    clips = result.scalars().all()
    current_pos = 0.0
    for i, clip in enumerate(clips):
        clip.clip_index = i
        duration = clip.source_end - clip.source_start
        clip.timeline_start = current_pos
        clip.timeline_end = current_pos + duration
        current_pos += duration
    await db.commit()


# ---- Endpoints ----

@router.post("/edit/{project_id}/split")
async def split_clip(project_id: int, req: SplitRequest, db: AsyncSession = Depends(get_db)):
    """Split a clip into two at a given time offset."""
    timeline = await get_timeline_for_project(project_id, db)
    result = await db.execute(select(Clip).where(Clip.id == req.clip_id))
    clip = result.scalar_one_or_none()
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    split_source = clip.source_start + req.split_at
    if split_source <= clip.source_start or split_source >= clip.source_end:
        raise HTTPException(status_code=400, detail="Split point out of range")

    # Second half
    new_clip = Clip(
        timeline_id=timeline.id,
        project_id=project_id,
        clip_index=clip.clip_index + 1,
        source_start=split_source,
        source_end=clip.source_end,
        timeline_start=clip.timeline_start + req.split_at,
        timeline_end=clip.timeline_end,
        script_paragraph=clip.script_paragraph,
        keyframe_path=clip.keyframe_path,
        adjustments=clip.adjustments,
        transition_in="fade",
        transition_out=clip.transition_out,
    )
    # First half
    clip.source_end = split_source
    clip.timeline_end = clip.timeline_start + req.split_at

    db.add(new_clip)
    await db.commit()
    await reindex_clips(timeline.id, db)
    return {"message": "Clip split successfully"}


@router.post("/edit/{project_id}/trim")
async def trim_clip(project_id: int, req: TrimRequest, db: AsyncSession = Depends(get_db)):
    """Trim a clip to new source timestamps."""
    result = await db.execute(select(Clip).where(Clip.id == req.clip_id))
    clip = result.scalar_one_or_none()
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    if req.new_start >= req.new_end:
        raise HTTPException(status_code=400, detail="new_start must be less than new_end")
    clip.source_start = req.new_start
    clip.source_end = req.new_end
    await db.commit()
    timeline = await get_timeline_for_project(project_id, db)
    await reindex_clips(timeline.id, db)
    return {"message": "Clip trimmed"}


@router.post("/edit/{project_id}/delete")
async def delete_clip(project_id: int, req: DeleteRequest, db: AsyncSession = Depends(get_db)):
    """Delete a clip from the timeline."""
    timeline = await get_timeline_for_project(project_id, db)
    result = await db.execute(select(Clip).where(Clip.id == req.clip_id))
    clip = result.scalar_one_or_none()
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    await db.delete(clip)
    await db.commit()
    await reindex_clips(timeline.id, db)
    return {"message": "Clip deleted"}


@router.post("/edit/{project_id}/insert")
async def insert_clip(project_id: int, req: InsertClipRequest, db: AsyncSession = Depends(get_db)):
    """Insert a new clip at a given index."""
    timeline = await get_timeline_for_project(project_id, db)
    # Bump indices of existing clips
    result = await db.execute(
        select(Clip).where(
            Clip.timeline_id == timeline.id,
            Clip.clip_index >= req.insert_at_index
        )
    )
    for c in result.scalars().all():
        c.clip_index += 1

    new_clip = Clip(
        timeline_id=timeline.id,
        project_id=project_id,
        clip_index=req.insert_at_index,
        source_start=req.source_start,
        source_end=req.source_end,
        timeline_start=0,
        timeline_end=req.duration,
    )
    db.add(new_clip)
    await db.commit()
    await reindex_clips(timeline.id, db)
    return {"message": "Clip inserted"}


@router.post("/edit/{project_id}/reorder")
async def reorder_clips(project_id: int, req: ReorderRequest, db: AsyncSession = Depends(get_db)):
    """Reorder clips by providing the new ordered list of clip IDs."""
    timeline = await get_timeline_for_project(project_id, db)
    for new_index, clip_id in enumerate(req.clip_ids):
        result = await db.execute(select(Clip).where(Clip.id == clip_id))
        clip = result.scalar_one_or_none()
        if clip:
            clip.clip_index = new_index
    await db.commit()
    await reindex_clips(timeline.id, db)
    return {"message": "Clips reordered"}


@router.post("/edit/{project_id}/adjust")
async def adjust_clip(project_id: int, req: AdjustRequest, db: AsyncSession = Depends(get_db)):
    """Adjust brightness, contrast, saturation, sharpness of a clip."""
    result = await db.execute(select(Clip).where(Clip.id == req.clip_id))
    clip = result.scalar_one_or_none()
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    adj = clip.adjustments_dict
    adj.update({
        "brightness": req.brightness,
        "contrast": req.contrast,
        "saturation": req.saturation,
        "sharpness": req.sharpness,
    })
    clip.adjustments = json.dumps(adj)
    await db.commit()
    return {"message": "Adjustments saved"}


@router.post("/edit/{project_id}/filter")
async def apply_filter(project_id: int, req: FilterRequest, db: AsyncSession = Depends(get_db)):
    """Apply a named filter preset to a clip."""
    result = await db.execute(select(Clip).where(Clip.id == req.clip_id))
    clip = result.scalar_one_or_none()
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    adj = clip.adjustments_dict
    adj["filter"] = req.filter_name
    clip.adjustments = json.dumps(adj)
    await db.commit()
    return {"message": f"Filter '{req.filter_name}' applied"}


@router.post("/edit/{project_id}/replace-audio")
async def replace_audio(project_id: int, req: ReplaceAudioRequest, db: AsyncSession = Depends(get_db)):
    """Set a new TTS/voiceover audio path for the timeline."""
    timeline = await get_timeline_for_project(project_id, db)
    timeline.tts_audio_path = req.audio_url
    await db.commit()
    return {"message": "Audio replaced"}
