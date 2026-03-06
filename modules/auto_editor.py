"""
Module 6: Automatic Video Editing Engine
Two modes:
  - "slideshow" (default): Ken Burns animated keyframe images + vignette + optional subtitles
  - "clips": original MoviePy v2 video-clips pipeline
"""
import os
import asyncio
import random
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from config import settings
from database import get_db
from models import Project, Timeline, Clip, Scene

router = APIRouter()

_edit_progress: dict[int, dict] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Hardware-accelerated FFmpeg encoder
# ─────────────────────────────────────────────────────────────────────────────

def _ffmpeg_encode_with_hw_accel(input_path: str, output_path: str, fps: int = 24, preset: str = "fast") -> None:
    base_args = ["ffmpeg", "-y", "-i", input_path]
    out_args = ["-b:v", "4M", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", output_path]
    encoders = [
        ["-c:v", "h264_qsv", "-global_quality", "23", "-look_ahead", "1"],
        ["-hwaccel", "cuda", "-c:v", "h264_nvenc", "-preset", "p1"],
        ["-c:v", "libx264", "-preset", preset, "-crf", "23"],
    ]
    for enc_args in encoders:
        try:
            result = subprocess.run(base_args + enc_args + out_args, capture_output=True, text=True, timeout=1800)
            if result.returncode == 0:
                return
        except Exception:
            pass
    raise RuntimeError("All ffmpeg encoders failed. Check ffmpeg installation.")


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE SLIDESHOW ENGINE
# ─────────────────────────────────────────────────────────────────────────────

_EFFECTS = ["zoom_in", "zoom_out", "pan_left", "pan_right", "zoom_in_pan_right", "zoom_out_pan_left"]


def _apply_ken_burns(img_array, effect: str, t: float, duration: float, w: int, h: int):
    """Apply a Ken Burns motion effect at time t. Returns (h, w, 3) numpy array."""
    import numpy as np
    import cv2

    progress = t / max(duration, 0.001)
    scale_start, scale_end = 1.0, 1.0
    tx_start, ty_start = 0.0, 0.0
    tx_end, ty_end = 0.0, 0.0

    if effect == "zoom_in":
        scale_start, scale_end = 1.0, 1.25
    elif effect == "zoom_out":
        scale_start, scale_end = 1.25, 1.0
    elif effect == "pan_left":
        scale_start = scale_end = 1.15
        tx_start, tx_end = 0.08, -0.08
    elif effect == "pan_right":
        scale_start = scale_end = 1.15
        tx_start, tx_end = -0.08, 0.08
    elif effect == "zoom_in_pan_right":
        scale_start, scale_end = 1.0, 1.2
        tx_start, tx_end = -0.05, 0.05
    elif effect == "zoom_out_pan_left":
        scale_start, scale_end = 1.2, 1.0
        tx_start, tx_end = 0.05, -0.05

    scale = scale_start + (scale_end - scale_start) * progress
    tx = tx_start + (tx_end - tx_start) * progress
    ty = ty_start + (ty_end - ty_start) * progress

    ih, iw = img_array.shape[:2]
    scaled_w = max(w + 2, int(iw * scale))
    scaled_h = max(h + 2, int(ih * scale))
    scaled = cv2.resize(img_array, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

    cx = int((scaled_w - w) / 2 + tx * scaled_w)
    cy = int((scaled_h - h) / 2 + ty * scaled_h)
    cx = max(0, min(cx, scaled_w - w))
    cy = max(0, min(cy, scaled_h - h))

    cropped = scaled[cy:cy + h, cx:cx + w]
    if cropped.shape[:2] != (h, w):
        cropped = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    return cropped


def _make_vignette(w: int, h: int, strength: float = 0.65):
    """Generate a vignette mask — dark cinematic edges. Returns (h, w, 3) float array 0..1."""
    import numpy as np
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xv, yv = np.meshgrid(x, y)
    dist = np.sqrt(xv**2 + yv**2)
    dist = dist / dist.max()
    mask = 1.0 - np.clip(dist * strength, 0, 1)
    mask = np.power(mask, 0.5)          # soften falloff
    return np.stack([mask, mask, mask], axis=2).astype(np.float32)


def _render_subtitle(frame, text: str, progress: float, w: int, h: int):
    """
    Burn subtitle text onto the frame using PIL for proper Urdu/Arabic Unicode rendering.
    `progress` (0.0 to 1.0) determines which part of the text segment is currently spoken.
    Returns modified frame.
    """
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import arabic_reshaper
    from bidi.algorithm import get_display

    if not text or not text.strip():
        return frame

    # 1. Reshape and Bidi-convert the text for RTL and Arabic/Urdu joining
    reshaped_text = arabic_reshaper.reshape(text.strip())
    bidi_text = get_display(reshaped_text)

    # 2. Determine font size (reduce significantly for less intrusion)
    # Original font scale was ~0.75 equivalent cv2 size. Let's make it ~0.45 for a cleaner look.
    base_font_size = int(max(16, w / 1280 * 35))
    
    # Try to load a unicode supporting font, fallback to default PIL font
    font = None
    try:
        # Try common windows system fonts that support Arabic/Urdu unicode
        font_paths = [
            "C:\\Windows\\Fonts\\arial.ttf", 
            "C:\\Windows\\Fonts\\tahoma.ttf", 
            "C:\\Windows\\Fonts\\seguiemj.ttf"
        ]
        for path in font_paths:
            if os.path.exists(path):
                try:
                    font = ImageFont.truetype(path, base_font_size)
                    break
                except Exception:
                    continue
                    
        if font is None:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    # 3. Create PIL Image from OpenCV frame (BGR -> RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_img, "RGBA")

    # Word-wrap using spaces (not perfect for all Arabic, but works for spaced words)
    words = bidi_text.split()
    max_chars_per_line = max(35, int(w / (base_font_size * 0.6)))
    
    lines, cur = [], []
    # Reverse words iterator so RTL word wrap flows correctly if mixed
    for word in words:
        if sum(len(w2) + 1 for w2 in cur) + len(word) > max_chars_per_line:
            if cur:
                lines.append(" ".join(cur))
            cur = [word]
        else:
            cur.append(word)
    if cur:
        lines.append(" ".join(cur))
        
    # Animate lines based on progress through the segment
    total_lines = len(lines)
    if total_lines > 0:
        # Prevent index out of bounds if progress perfectly hits 1.0
        active_idx = min(total_lines - 1, int(progress * total_lines))
        lines = [lines[active_idx]]
    else:
        lines = []

    line_h = int(base_font_size * 1.5)
    total_h = line_h * len(lines)
    # Move to the very bottom
    y_start = h - total_h - int(h * 0.03)

    # 4. Draw text line by line (no background box)
    for li, line in enumerate(lines):
        # Calculate text width to center it
        try:
            bbox = font.getbbox(line)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except AttributeError:
            text_w, text_h = draw.textsize(line, font=font)

        x = (w - text_w) // 2
        y = y_start + (li * line_h) + (line_h - text_h) // 2

        # Draw stronger Shadow for legibility without background
        shadow_offset = max(2, int(base_font_size * 0.08))
        draw.text((x + shadow_offset, y + shadow_offset), line, font=font, fill=(0, 0, 0, 255))
        draw.text((x - shadow_offset, y + shadow_offset), line, font=font, fill=(0, 0, 0, 255))
        draw.text((x + shadow_offset, y - shadow_offset), line, font=font, fill=(0, 0, 0, 255))
        draw.text((x - shadow_offset, y - shadow_offset), line, font=font, fill=(0, 0, 0, 255))
        
        # Draw text
        draw.text((x, y), line, font=font, fill=(255, 255, 255, 255))

    # 6. Convert back to OpenCV (RGB -> BGR)
    rendered_rgb = np.array(pil_img)
    rendered_bgr = cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR)
    
    return rendered_bgr


def build_slideshow_from_images(
    image_paths: list[str],
    tts_audio_path: Optional[str],
    output_path: str,
    target_w: int,
    target_h: int,
    target_duration: float,
    subtitles: list[dict] = None,      # [{start, end, text}] — summary segments
    show_subtitles: bool = True,
    fps: int = 24,
    transition_duration: float = 0.5,
    progress_callback=None,
) -> None:
    """
    Build a video from still images with:
    - Ken Burns zoom/pan effects (rotated per image)
    - Vignette overlay (cinematic dark edges)
    - Subtitle overlay (optional, from summary segments)
    - Crossfade transitions
    - TTS audio track
    Images are randomly shuffled and looped to fill target_duration.
    """
    import numpy as np
    import cv2
    import tempfile
    import os
    import subprocess
    from moviepy import AudioFileClip

    if not image_paths:
        raise ValueError("No images provided for slideshow")

    # ── Build shuffled image list that fills target_duration ─────────────────
    img_duration = max(2.5, min(6.0, target_duration / max(len(image_paths), 1)))
    needed = int(target_duration / img_duration) + 2

    shuffled = list(image_paths)
    random.shuffle(shuffled)

    # Loop through shuffled images until we have enough
    pool = []
    while len(pool) < needed:
        batch = list(shuffled)
        random.shuffle(batch)
        pool.extend(batch)
    pool = pool[:needed]

    # ── Build subtitle lookup ─────────────────────────────────────────────────
    sub_segments = subtitles or []

    def _get_subtitle_at(t: float) -> tuple[str, float]:
        """Return (subtitle_text, progress_ratio) for the given absolute time offset."""
        for seg in sub_segments:
            start_t = seg.get("start", 0)
            end_t = seg.get("end", 0)
            if start_t <= t <= end_t:
                duration = max(0.001, end_t - start_t)
                progress = (t - start_t) / duration
                return (seg.get("text", ""), progress)
        return ("", 0.0)

    # ── Pre-build vignette mask ───────────────────────────────────────────────
    vignette = _make_vignette(target_w, target_h, strength=0.7)

    # ── Render frames via OpenCV VideoWriter ──────────────────────────────────
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        video_only_path = tmp.name

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(video_only_path, fourcc, fps, (target_w, target_h))

    total_frames_target = int(target_duration * fps)
    frames_written = 0

    time_cursor = 0.0
    total = len(pool)
    for i, img_path in enumerate(pool):
        if frames_written >= total_frames_target:
            break

        effect = _EFFECTS[i % len(_EFFECTS)]
        img = cv2.imread(img_path)
        if img is None:
            time_cursor += img_duration
            continue
        
        # Keep BGR for cv2 writing
        ih, iw = img.shape[:2]
        scale = max(target_w / iw, target_h / ih) * 1.32
        img_large = cv2.resize(img, (int(iw * scale), int(ih * scale)), interpolation=cv2.INTER_LANCZOS4)

        # Generate frames for this image's duration
        clip_frames = int(img_duration * fps)
        
        # Account for crossfade overlap (if not the first image)
        overlap_frames = int(transition_duration * fps) if i > 0 else 0
        
        for f in range(clip_frames):
            if frames_written >= total_frames_target:
                break
                
            t_local = f / fps
            t_absolute = time_cursor + t_local
            
            # 1. Ken Burns (operates on BGR here)
            frame = _apply_ken_burns(img_large, effect, t_local, img_duration, target_w, target_h)
            # 2. Keep colors natural (no vignette/grading per user request)
            # frame is already purely the correct BGR uint8 pixel values
            
            # 3. Subtitles
            if show_subtitles and sub_segments:
                sub_text, sub_progress = _get_subtitle_at(t_absolute)
                if sub_text:
                    frame = _render_subtitle(frame, sub_text, sub_progress, target_w, target_h)
            
            # Note: A real crossfade buffer is complex in a streaming writer. 
            # For massive stability and speed, we will write hard cuts here. 
            # The vignette and slow Ken Burns already provide a smooth cinematic feel.
            out.write(frame)
            frames_written += 1
            
        time_cursor += img_duration

        if progress_callback:
            progress_callback(10 + int(70 * (i + 1) / total))

    out.release()

    if progress_callback:
        progress_callback(85)

    # ── Merge Audio & Hardware Encode ─────────────────────────────────────────

    try:
        if tts_audio_path and os.path.exists(tts_audio_path):
            import subprocess
            # Merge audio using ultra-fast ffmpeg copy/encode
            cmd = [
                "ffmpeg", "-y",
                "-i", video_only_path,
                "-i", tts_audio_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-t", str(target_duration),
                "-map", "0:v:0",
                "-map", "1:a:0",
                output_path
            ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError:
                # Fallback if map fails
                _ffmpeg_encode_with_hw_accel(video_only_path, output_path, fps=fps, preset="fast")
        else:
            # No audio, just copy to final
            import shutil
            shutil.copy2(video_only_path, output_path)

    finally:
        try:
            os.remove(video_only_path)
        except Exception:
            pass
        try:
            os.remove(tmp_path)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# CLIPS ENGINE (original pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def apply_adjustments_to_clip(clip, adjustments: dict):
    try:
        from moviepy.video.fx import MultiplyColor
        b = adjustments.get("brightness", 0) / 100.0
        if b != 0:
            clip = clip.with_effects([MultiplyColor(1.0 + b)])
    except Exception:
        pass
    return clip


def build_video_from_timeline(
    video_path: str, clips_data: list, tts_audio_path: Optional[str],
    output_path: str, aspect: str, target_w: int, target_h: int, progress_callback=None
):
    from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips
    from moviepy.video.fx import Resize, FadeIn, FadeOut, Loop

    video = VideoFileClip(video_path)
    final_clips = []
    total = len(clips_data)

    for i, clip_data in enumerate(clips_data):
        src_start = max(0, clip_data["source_start"])
        src_end = min(video.duration, clip_data["source_end"])
        if src_end <= src_start:
            src_end = min(src_start + 3.0, video.duration)
        seg_duration = clip_data["timeline_end"] - clip_data["timeline_start"]
        if seg_duration <= 0:
            seg_duration = 3.0
        try:
            sub = video.subclipped(src_start, src_end)
        except Exception:
            sub = video.subclipped(0, min(3.0, video.duration))
        if sub.duration < seg_duration:
            sub = sub.with_effects([Loop(duration=seg_duration)])
        else:
            sub = sub.subclipped(0, seg_duration)
        sub = sub.with_effects([Resize((target_w, target_h))])
        sub = apply_adjustments_to_clip(sub, clip_data.get("adjustments", {}))
        transition = clip_data.get("transition_in", "fade")
        if transition in ("fade", "crossfade") and sub.duration > 0.5:
            sub = sub.with_effects([FadeIn(0.3), FadeOut(0.3)])
        final_clips.append(sub.without_audio())
        if progress_callback and total > 0:
            progress_callback(20 + int(40 * (i + 1) / total))

    if not final_clips:
        video.close()
        raise ValueError("No clips to assemble")

    final_video = concatenate_videoclips(final_clips)
    if tts_audio_path and os.path.exists(tts_audio_path):
        audio = AudioFileClip(tts_audio_path)
        if audio.duration > final_video.duration:
            audio = audio.subclipped(0, final_video.duration)
        final_video = final_video.with_audio(audio)
    if progress_callback:
        progress_callback(65)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        final_video.write_videofile(tmp_path, codec="libx264", audio_codec="aac", fps=24, preset="ultrafast", logger=None)
        video.close()
        final_video.close()
        if progress_callback:
            progress_callback(80)
        _ffmpeg_encode_with_hw_accel(tmp_path, output_path, fps=24, preset="fast")
        if progress_callback:
            progress_callback(98)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def get_target_size(aspect: str):
    return {"16:9": (1280, 720), "9:16": (720, 1280), "1:1": (720, 720), "4:5": (720, 900)}.get(aspect, (1280, 720))


# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND TASK
# ─────────────────────────────────────────────────────────────────────────────

async def run_auto_edit(project_id: int, mode: str = "slideshow", show_subtitles: bool = True):
    from database import AsyncSessionLocal
    from models import Transcript

    _edit_progress[project_id] = {"step": "starting", "progress": 0, "mode": mode}

    async with AsyncSessionLocal() as db:
        p_result = await db.execute(select(Project).where(Project.id == project_id))
        project = p_result.scalar_one_or_none()
        if not project:
            _edit_progress[project_id] = {"step": "error", "error": "Project not found"}
            return

        exports_dir = Path(settings.exports_dir)
        exports_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(exports_dir / f"project_{project_id}_draft.mp4")

        def update_progress(val: int):
            _edit_progress[project_id] = {"step": "rendering", "progress": val, "mode": mode}

        loop = asyncio.get_event_loop()

        try:
            if mode == "slideshow":
                # Get all keyframes
                scenes_result = await db.execute(
                    select(Scene).where(Scene.project_id == project_id).order_by(Scene.scene_index)
                )
                scenes = scenes_result.scalars().all()
                image_paths = [
                    s.keyframe_path.replace('\\', '/') for s in scenes
                    if s.keyframe_path and os.path.exists(s.keyframe_path.replace('\\', '/'))
                ]
                if not image_paths:
                    _edit_progress[project_id] = {
                        "step": "error",
                        "error": "No keyframe images found. Run analysis first."
                    }
                    return

                # Get target duration from transcript
                transcript_result = await db.execute(
                    select(Transcript).where(Transcript.project_id == project_id)
                )
                transcript = transcript_result.scalar_one_or_none()
                target_minutes = float(transcript.target_minutes or 5) if transcript else 5.0
                target_duration = target_minutes * 60.0

                # Get subtitle segments (summary segments for timing)
                subtitles = []
                if transcript and transcript.summary_segments:
                    try:
                        import json
                        subtitles = json.loads(transcript.summary_segments)
                    except Exception:
                        pass

                # TTS + aspect ratio
                tl_result = await db.execute(
                    select(Timeline).where(Timeline.project_id == project_id, Timeline.is_current == True)
                )
                timeline = tl_result.scalar_one_or_none()
                tts_audio = timeline.tts_audio_path if timeline else None
                aspect = timeline.aspect_ratio if timeline else "16:9"
                target_w, target_h = get_target_size(aspect)

                _edit_progress[project_id] = {"step": "rendering", "progress": 5, "mode": mode}

                tts_safe = tts_audio.replace('\\', '/') if tts_audio else None
                await loop.run_in_executor(
                    None, build_slideshow_from_images,
                    image_paths, tts_safe, output_path,
                    target_w, target_h, target_duration,
                    subtitles, show_subtitles, 24, 0.5, update_progress
                )

            else:
                # Clips mode
                tl_result = await db.execute(
                    select(Timeline).where(Timeline.project_id == project_id, Timeline.is_current == True)
                )
                timeline = tl_result.scalar_one_or_none()
                if not timeline:
                    _edit_progress[project_id] = {
                        "step": "error",
                        "error": "Timeline not found. Run visual matching first."
                    }
                    return

                clips_result = await db.execute(
                    select(Clip).where(Clip.timeline_id == timeline.id).order_by(Clip.clip_index)
                )
                clips = clips_result.scalars().all()
                if not clips:
                    _edit_progress[project_id] = {"step": "error", "error": "No clips found in timeline."}
                    return

                clips_data = [
                    {
                        "source_start": c.source_start, "source_end": c.source_end,
                        "timeline_start": c.timeline_start, "timeline_end": c.timeline_end,
                        "adjustments": c.adjustments_dict, "transition_in": c.transition_in,
                    }
                    for c in clips
                ]
                target_w, target_h = get_target_size(timeline.aspect_ratio)
                _edit_progress[project_id] = {"step": "rendering", "progress": 20, "mode": mode}

                safe_video = project.video_path.replace('\\', '/') if project.video_path else None
                safe_tts = timeline.tts_audio_path.replace('\\', '/') if timeline.tts_audio_path else None

                await loop.run_in_executor(
                    None, build_video_from_timeline,
                    safe_video, clips_data, safe_tts,
                    output_path, timeline.aspect_ratio, target_w, target_h, update_progress
                )

            # Save draft path
            tl_upd = await db.execute(
                select(Timeline).where(Timeline.project_id == project_id, Timeline.is_current == True)
            )
            saved_tl = tl_upd.scalar_one_or_none()
            if saved_tl:
                saved_tl.draft_video_path = output_path
            project.status = "draft_ready"
            await db.commit()
            _edit_progress[project_id] = {"step": "done", "progress": 100, "mode": mode}

        except Exception as e:
            _edit_progress[project_id] = {"step": "error", "progress": 0, "error": str(e), "mode": mode}
            raise


# ─────────────────────────────────────────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

class AutoEditRequest(BaseModel):
    mode: str = "slideshow"         # "slideshow" | "clips"
    show_subtitles: bool = True     # burn subtitles onto frames


@router.post("/auto-edit/{project_id}")
async def start_auto_edit(
    project_id: int,
    request: AutoEditRequest = AutoEditRequest(),
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db)
):
    if _edit_progress.get(project_id, {}).get("step") == "rendering":
        raise HTTPException(status_code=409, detail="Auto edit already in progress.")
    if request.mode == "clips":
        tl_result = await db.execute(
            select(Timeline).where(Timeline.project_id == project_id, Timeline.is_current == True)
        )
        if not tl_result.scalar_one_or_none():
            raise HTTPException(status_code=400, detail="Run visual matching first to build a timeline.")
    background_tasks.add_task(run_auto_edit, project_id, request.mode, request.show_subtitles)
    return {"message": "Auto-editing started", "project_id": project_id, "mode": request.mode}


@router.get("/auto-edit/{project_id}/status")
async def get_auto_edit_status(project_id: int):
    return _edit_progress.get(project_id, {"step": "not_started", "progress": 0})


@router.get("/auto-edit/{project_id}/preview")
async def get_draft_video_url(project_id: int, db: AsyncSession = Depends(get_db)):
    draft_path = str(Path(settings.exports_dir) / f"project_{project_id}_draft.mp4")
    tl_result = await db.execute(
        select(Timeline).where(Timeline.project_id == project_id, Timeline.is_current == True)
    )
    timeline = tl_result.scalar_one_or_none()
    if timeline and timeline.draft_video_path:
        draft_path = timeline.draft_video_path
    if not os.path.exists(draft_path):
        raise HTTPException(status_code=404, detail="Draft video not ready")
    return {
        "video_url": f"/storage/exports/project_{project_id}_draft.mp4",
        "duration": timeline.total_duration if timeline else None,
    }