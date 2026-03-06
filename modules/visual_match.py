"""
Module 5: Visual Matching
Uses Sentence Transformers + CLIP to match script paragraphs to video scenes/keyframes.
"""
import os
import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from config import settings
from database import get_db
from models import Project, Transcript, Scene, Timeline, Clip

router = APIRouter()

_match_progress: dict[int, dict] = {}


def compute_clip_match(summary_segments: list, scenes: list, device: str) -> list:
    """
    Match each summary segment to the most visually-relevant scene using CLIP.
    Falls back to evenly-distributed scene selection if CLIP unavailable.
    """
    texts = [seg["text"] for seg in summary_segments]
    n_segs = len(texts)
    n_scenes = len(scenes)

    if not scenes or not texts:
        return [0] * n_segs

    # --- Try CLIP matching ---
    try:
        import clip
        import torch
        from PIL import Image

        keyframe_paths = [s["keyframe_path"] for s in scenes if s.get("keyframe_path")]
        if not keyframe_paths:
            raise ValueError("No keyframes")

        model, preprocess = clip.load("ViT-B/32", device=device)

        text_tokens = clip.tokenize(texts, truncate=True).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_features_list = []
        valid_indices = []
        for idx, kf_path in enumerate(keyframe_paths):
            try:
                image = preprocess(Image.open(kf_path)).unsqueeze(0).to(device)
                with torch.no_grad():
                    feat = model.encode_image(image)
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                image_features_list.append(feat)
                valid_indices.append(idx)
            except Exception:
                continue

        if not image_features_list:
            raise ValueError("No images encoded")

        import torch as _torch
        image_features = _torch.cat(image_features_list, dim=0)
        similarity = (text_features @ image_features.T).cpu().numpy()
        matched = similarity.argmax(axis=1).tolist()
        return [valid_indices[int(m)] for m in matched]

    except Exception:
        pass

    # --- Fallback: evenly distribute scenes across segments ---
    # This ensures variety — no clip repeats back-to-back
    import math
    result = []
    for i in range(n_segs):
        # Spread segments evenly across available scenes
        scene_idx = math.floor(i * n_scenes / n_segs)
        scene_idx = min(scene_idx, n_scenes - 1)
        result.append(scene_idx)
    return result


async def run_visual_matching(project_id: int):
    """Background visual matching and timeline building."""
    from database import AsyncSessionLocal

    _match_progress[project_id] = {"step": "starting", "progress": 0}

    async with AsyncSessionLocal() as db:
        # Fetch transcript
        t_result = await db.execute(select(Transcript).where(Transcript.project_id == project_id))
        transcript = t_result.scalar_one_or_none()
        if not transcript or not transcript.summary_segments:
            _match_progress[project_id] = {"step": "error", "error": "No summarized script found"}
            return

        # Fetch scenes
        s_result = await db.execute(
            select(Scene).where(Scene.project_id == project_id).order_by(Scene.start_time)
        )
        scenes_orm = s_result.scalars().all()
        scenes = [
            {
                "id": s.id,
                "scene_index": s.scene_index,
                "start_time": s.start_time,
                "end_time": s.end_time,
                "keyframe_path": s.keyframe_path,
            }
            for s in scenes_orm
        ]

        if not scenes:
            _match_progress[project_id] = {"step": "error", "error": "No scenes found. Run analysis first."}
            return

        summary_segments = transcript.summary_segments_list

        if not summary_segments:
            _match_progress[project_id] = {"step": "error", "error": "No summary segments found."}
            return

        _match_progress[project_id] = {"step": "matching", "progress": 30}

        try:
            loop = asyncio.get_event_loop()
            matched_scene_indices = await loop.run_in_executor(
                None, compute_clip_match, summary_segments, scenes, settings.device
            )

            # Get or create current timeline
            tl_result = await db.execute(
                select(Timeline).where(Timeline.project_id == project_id, Timeline.is_current == True)
            )
            timeline = tl_result.scalar_one_or_none()
            if not timeline:
                timeline = Timeline(project_id=project_id)
                db.add(timeline)
                await db.flush()

            # Compute TTS audio duration for pacing
            tts_path = Path(settings.tts_dir) / f"project_{project_id}_tts.wav"
            tts_duration = None
            if tts_path.exists():
                try:
                    import wave
                    with wave.open(str(tts_path), "r") as wf:
                        tts_duration = wf.getnframes() / wf.getframerate()
                except Exception:
                    pass

            # Delete old clips
            old_clips = await db.execute(select(Clip).where(Clip.timeline_id == timeline.id))
            for c in old_clips.scalars().all():
                await db.delete(c)
            await db.flush()

            # Build timeline clips — one clip per summary segment
            n_segs = len(summary_segments)
            total_timeline_duration = tts_duration or sum(
                max(0.1, seg.get("end", 0) - seg.get("start", 0)) for seg in summary_segments
            )
            current_timeline_pos = 0.0

            for i, (seg, scene_idx) in enumerate(zip(summary_segments, matched_scene_indices)):
                scene = scenes[min(scene_idx, len(scenes) - 1)]
                source_start = float(scene["start_time"])
                source_end = float(scene["end_time"])

                # Clip duration proportional to segment length in TTS
                if tts_duration and summary_segments[-1].get("end", 0) > 0:
                    seg_dur = tts_duration * (
                        (seg.get("end", 0) - seg.get("start", 0)) /
                        max(0.01, summary_segments[-1]["end"])
                    )
                else:
                    seg_dur = total_timeline_duration / max(1, n_segs)

                seg_dur = max(1.0, seg_dur)  # minimum 1 second per clip

                db.add(Clip(
                    timeline_id=timeline.id,
                    project_id=project_id,
                    clip_index=i,
                    source_start=source_start,
                    source_end=min(source_end, source_start + seg_dur),
                    timeline_start=current_timeline_pos,
                    timeline_end=current_timeline_pos + seg_dur,
                    script_paragraph=seg.get("text", ""),
                    keyframe_path=scene["keyframe_path"] if scene else None,
                    transition_in="fade",
                    transition_out="fade",
                ))
                current_timeline_pos += seg_dur

            timeline.total_duration = current_timeline_pos
            await db.commit()

            _match_progress[project_id] = {"step": "done", "progress": 100}

        except Exception as e:
            _match_progress[project_id] = {"step": "error", "progress": 0, "error": str(e)}
            raise


@router.post("/match/{project_id}")
async def start_visual_match(
    project_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Match summarized script paragraphs to scenes visually."""
    background_tasks.add_task(run_visual_matching, project_id)
    return {"message": "Visual matching started", "project_id": project_id}


@router.get("/match/{project_id}/status")
async def get_match_status(project_id: int):
    """Poll visual matching progress."""
    return _match_progress.get(project_id, {"step": "not_started", "progress": 0})


@router.get("/timeline/{project_id}")
async def get_timeline(project_id: int, db: AsyncSession = Depends(get_db)):
    """Get the current timeline with all clips."""
    tl_result = await db.execute(
        select(Timeline).where(Timeline.project_id == project_id, Timeline.is_current == True)
    )
    timeline = tl_result.scalar_one_or_none()
    if not timeline:
        raise HTTPException(status_code=404, detail="Timeline not found")

    clips_result = await db.execute(
        select(Clip).where(Clip.timeline_id == timeline.id).order_by(Clip.clip_index)
    )
    clips = clips_result.scalars().all()

    response_clips = []
    for c in clips:
        keyframe_name = Path(c.keyframe_path.replace('\\', '/')).name if c.keyframe_path else None
        response_clips.append({
            "id": c.id,
            "clip_index": c.clip_index,
            "source_start": c.source_start,
            "source_end": c.source_end,
            "timeline_start": c.timeline_start,
            "timeline_end": c.timeline_end,
            "script_paragraph": c.script_paragraph,
            "keyframe_url": f"/storage/frames/project_{project_id}/{keyframe_name}" if keyframe_name else None,
            "adjustments": c.adjustments_dict,
            "transition_in": c.transition_in,
            "transition_out": c.transition_out,
        })

    return {
        "timeline_id": timeline.id,
        "project_id": project_id,
        "total_duration": timeline.total_duration,
        "aspect_ratio": timeline.aspect_ratio,
        "tts_audio_url": (
            f"/storage/tts/project_{project_id}_tts.wav"
            if timeline.tts_audio_path else None
        ),
        "clips": response_clips
    }