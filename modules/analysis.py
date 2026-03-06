"""
Module 2: Video Analysis
Whisper speech-to-text, PySceneDetect scene detection, OpenCV keyframe extraction.
"""
import os
import json
import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from config import settings
from database import get_db
from models import Project, Transcript, Scene

router = APIRouter()

_analysis_progress: dict[int, dict] = {}


# Languages that use non-Latin scripts — tiny/base Whisper models struggle with these
_NON_LATIN_LANGUAGES = {"ur", "hi", "ar", "pa", "fa", "zh", "ja", "ko", "ru", "uk",
                        "el", "he", "am", "th", "ka", "hy", "bn", "gu", "te", "ta"}
_SMALL_MODELS = {"tiny", "tiny.en"}   # Only truly tiny models get upgraded; small/base are fine for Urdu


def transcribe_audio(audio_path: str, model_size: str, device: str, language: str = None) -> dict:
    """
    Run Whisper transcription — tries faster-whisper first (4× faster on CPU via CTranslate2
    with INT8 quantization), then falls back to original openai-whisper.
    - language: ISO-639-1 code, e.g. 'ur', 'hi', 'en'. Forces language detection.
    - task is always 'transcribe' (never 'translate') so output stays in original script.
    - For non-Latin scripts like Urdu/Hindi/Arabic, auto-upgrades from tiny/base → medium.
    """
    # Auto-upgrade model size for complex scripts when using a small model
    effective_model = model_size
    if language in _NON_LATIN_LANGUAGES and model_size in _SMALL_MODELS:
        effective_model = "medium"  # medium has solid Urdu/Arabic/Hindi support

    # --- Try faster-whisper first (CTranslate2 — 4× faster on CPU, lower memory) ---
    try:
        from faster_whisper import WhisperModel

        # Choose compute type: int8 for CPU (fastest), float16 for CUDA
        compute_type = "float16" if device == "cuda" else "int8"
        fw_device = device if device == "cuda" else "cpu"

        model = WhisperModel(
            effective_model,
            device=fw_device,
            compute_type=compute_type,
            num_workers=4,          # parallel audio chunk processing
            cpu_threads=0,          # 0 = use all available CPU cores
        )
        segments_iter, info = model.transcribe(
            audio_path,
            language=language,                  # None = auto-detect
            task="transcribe",                  # never translate
            beam_size=1,                        # greedy (fastest) — minimal quality loss
            condition_on_previous_text=False,   # skip re-encoding between segments
            word_timestamps=False,              # we don't need word-level, segment only
            vad_filter=True,                    # skip silence — big speedup on dramas
            vad_parameters={"min_silence_duration_ms": 300},
        )

        segments = []
        full_text_parts = []
        for seg in segments_iter:      # generator — processes lazily
            segments.append({"start": round(seg.start, 3), "end": round(seg.end, 3), "text": seg.text.strip()})
            full_text_parts.append(seg.text.strip())

        return {
            "text": " ".join(full_text_parts),
            "language": info.language or language or "en",
            "segments": segments,
            "engine": "faster-whisper",
        }

    except ImportError:
        print("[Whisper] faster-whisper not installed, falling back to openai-whisper")
        pass  # fall through to standard whisper
    except Exception as fw_err:
        print(f"[Whisper] faster-whisper error: {fw_err!r} — falling back to openai-whisper")
        pass  # fall through to standard whisper

    # --- Fallback: original openai-whisper ---
    import whisper
    model = whisper.load_model(effective_model, device=device)
    options = {
        "verbose": False,
        "task": "transcribe",     # NEVER translate — keep original language script
    }
    if language:
        options["language"] = language

    result = model.transcribe(audio_path, **options)
    return {
        "text": result["text"],
        "language": result.get("language", language or "en"),
        "segments": [
            {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
            for seg in result["segments"]
        ],
        "engine": "openai-whisper",
    }


def detect_scenes(video_path: str, duration: float) -> list[dict]:
    detected = []

    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import AdaptiveDetector
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(AdaptiveDetector(adaptive_threshold=2.0, min_scene_len=8))
        scene_manager.detect_scenes(video, show_progress=False)
        scene_list = scene_manager.get_scene_list()
        for start, end in scene_list:
            detected.append((start.get_seconds(), end.get_seconds()))
    except Exception:
        pass

    if len(detected) < 8:
        try:
            from scenedetect import open_video, SceneManager
            from scenedetect.detectors import ContentDetector
            video = open_video(video_path)
            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector(threshold=10.0, min_scene_len=5))
            scene_manager.detect_scenes(video, show_progress=False)
            scene_list = scene_manager.get_scene_list()
            if len(scene_list) > len(detected):
                detected = [(s.get_seconds(), e.get_seconds()) for s, e in scene_list]
        except Exception:
            pass

    duration = duration or 30.0
    chunk_size = max(2.0, duration / 25)
    time_cuts = set()
    t = 0.0
    while t < duration:
        time_cuts.add(round(t, 2))
        t += chunk_size

    all_starts = sorted(set([s for s, e in detected] + list(time_cuts)))

    scenes = []
    for i, start in enumerate(all_starts):
        end = all_starts[i + 1] if i + 1 < len(all_starts) else duration
        if end - start < 0.5:
            continue
        scenes.append({
            "index": len(scenes),
            "start_time": round(start, 3),
            "end_time": round(end, 3),
        })

    return scenes if scenes else [{"index": 0, "start_time": 0.0, "end_time": duration}]


def extract_keyframe(video_path: str, timestamp: float, output_path: str) -> bool:
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_number = int(timestamp * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if ret:
        cv2.imwrite(output_path, frame)
        return True
    return False


async def run_analysis(project_id: int, language: str = None):
    from database import AsyncSessionLocal

    _analysis_progress[project_id] = {"step": "starting", "progress": 0}

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Project).where(Project.id == project_id))
        project = result.scalar_one_or_none()
        if not project:
            _analysis_progress[project_id] = {"step": "error", "progress": 0, "error": "Project not found"}
            return

        project.status = "analyzing"
        await db.commit()

        try:
            # Step 1: Transcription — force language if provided
            _analysis_progress[project_id] = {"step": "transcribing", "progress": 10}
            loop = asyncio.get_event_loop()
            transcript_data = await loop.run_in_executor(
                None, transcribe_audio,
                project.audio_path, settings.whisper_model, settings.device, language
            )

            existing = await db.execute(select(Transcript).where(Transcript.project_id == project_id))
            existing_transcript = existing.scalar_one_or_none()
            if existing_transcript:
                existing_transcript.full_text = transcript_data["text"]
                existing_transcript.segments = json.dumps(transcript_data["segments"])
                existing_transcript.language = transcript_data["language"]
            else:
                db.add(Transcript(
                    project_id=project_id,
                    full_text=transcript_data["text"],
                    segments=json.dumps(transcript_data["segments"]),
                    language=transcript_data["language"],
                ))
            await db.commit()

            # Step 2: Scene detection
            _analysis_progress[project_id] = {"step": "detecting_scenes", "progress": 50}
            scenes_data = await loop.run_in_executor(
                None, detect_scenes, project.video_path, project.duration or 60.0
            )

            # Step 3: Keyframe extraction
            _analysis_progress[project_id] = {"step": "extracting_keyframes", "progress": 70}
            frames_dir = Path(settings.frames_dir) / f"project_{project_id}"
            frames_dir.mkdir(parents=True, exist_ok=True)

            old_scenes = await db.execute(select(Scene).where(Scene.project_id == project_id))
            for s in old_scenes.scalars().all():
                await db.delete(s)
            await db.commit()

            total = len(scenes_data)
            # Ensure cross-platform path compatibility for video_path
            safe_video_path = project.video_path.replace('\\', '/') if project.video_path else None
            
            for i, scene_info in enumerate(scenes_data):
                mid_time = (scene_info["start_time"] + scene_info["end_time"]) / 2
                keyframe_path = str(frames_dir / f"scene_{scene_info['index']:04d}.jpg").replace('\\', '/')
                success = await loop.run_in_executor(
                    None, extract_keyframe, safe_video_path, mid_time, keyframe_path
                )
                db.add(Scene(
                    project_id=project_id,
                    scene_index=scene_info["index"],
                    start_time=scene_info["start_time"],
                    end_time=scene_info["end_time"],
                    keyframe_path=keyframe_path if success else None,
                ))
                _analysis_progress[project_id] = {
                    "step": "extracting_keyframes",
                    "progress": 70 + int(25 * (i + 1) / max(1, total))
                }

            await db.commit()
            project.status = "analyzed"
            await db.commit()
            _analysis_progress[project_id] = {"step": "done", "progress": 100}

        except Exception as e:
            project.status = "error"
            await db.commit()
            _analysis_progress[project_id] = {"step": "error", "progress": 0, "error": str(e)}
            raise


@router.post("/analyze/{project_id}")
async def start_analysis(
    project_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    language: str = Query(default=None, description="Force language: ur, hi, en, ar, pa")
):
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    safe_audio_path = project.audio_path.replace('\\', '/') if project.audio_path else None
    
    if not safe_audio_path:
        raise HTTPException(status_code=400, detail="Audio file not found in DB. Upload a video first.")
        
    resolved_path = Path(safe_audio_path).resolve()
    if not resolved_path.exists():
        raise HTTPException(status_code=400, detail=f"Audio file not found on disk at {resolved_path}. Upload a video first.")

    background_tasks.add_task(run_analysis, project_id, language)
    return {"message": "Analysis started", "project_id": project_id, "language": language}


@router.get("/analyze/{project_id}/status")
async def get_analysis_status(project_id: int):
    progress = _analysis_progress.get(project_id, {"step": "not_started", "progress": 0})
    return {"project_id": project_id, **progress}


@router.get("/transcript/{project_id}")
async def get_transcript(project_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Transcript).where(Transcript.project_id == project_id))
    transcript = result.scalar_one_or_none()
    if not transcript:
        raise HTTPException(status_code=404, detail="Transcript not found")
    return {
        "project_id": project_id,
        "full_text": transcript.full_text,
        "language": transcript.language,
        "segments": transcript.segments_list,
        "summary_text": transcript.summary_text,
    }


@router.get("/scenes/{project_id}")
async def get_scenes(project_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Scene).where(Scene.project_id == project_id).order_by(Scene.start_time)
    )
    scenes = result.scalars().all()
    response_scenes = []
    for s in scenes:
        keyframe_name = Path(s.keyframe_path.replace('\\', '/')).name if s.keyframe_path else None
        response_scenes.append({
            "id": s.id,
            "scene_index": s.scene_index,
            "start_time": s.start_time,
            "end_time": s.end_time,
            "duration": round(s.end_time - s.start_time, 2),
            "keyframe_url": f"/storage/frames/project_{project_id}/{keyframe_name}" if keyframe_name else None,
        })
    return response_scenes