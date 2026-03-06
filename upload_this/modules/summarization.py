"""
Module 3: Script Summarization
Uses BART (distilbart) for initial summarization, then Groq (Llama 3.3 70B)
to rewrite it as a professional Hook → Intro → Body script.
"""
import json
import asyncio
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from config import settings
from database import get_db
from models import Project, Transcript

router = APIRouter()

_summarize_progress: dict[int, dict] = {}
AVG_WORDS_PER_MINUTE = 130


class SummarizeRequest(BaseModel):
    target_minutes: float = 5.0
    style_prompt: Optional[str] = "YouTube drama review, engaging and conversational"


class ScriptUpdateRequest(BaseModel):
    summary_text: str


def chunk_text(text: str, max_tokens: int = 900) -> list[str]:
    """Split text into chunks that fit within model token limits."""
    words = text.split()
    chunks = []
    current = []
    current_len = 0
    for word in words:
        current.append(word)
        current_len += 1
        if current_len >= max_tokens:
            chunks.append(" ".join(current))
            current = []
            current_len = 0
    if current:
        chunks.append(" ".join(current))
    return chunks


def summarize_text(text: str, target_words: int, model_name: str, device: str) -> str:
    """
    Summarize text using BART/distilBART from HuggingFace.
    Uses AutoTokenizer + AutoModelForSeq2SeqLM directly (compatible with
    transformers v5.x where the 'summarization' pipeline task was removed).
    """
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch

    words = text.split()
    total_words = len(words)

    if total_words == 0:
        return text

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    torch_device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(torch_device)

    def _summarize_chunk(chunk_text: str, max_len: int, min_len: int) -> str:
        inputs = tokenizer(
            chunk_text, return_tensors="pt", max_length=1024, truncation=True
        ).to(torch_device)
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_len,
            min_length=min_len,
            num_beams=4,
            do_sample=False,
            length_penalty=2.0,
        )
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    chunks = chunk_text(text, max_tokens=400)
    target_per_chunk = max(30, target_words // max(1, len(chunks)))
    max_per_chunk = min(target_per_chunk * 2, 512)

    summaries = []
    for chunk in chunks:
        chunk_words = len(chunk.split())
        if chunk_words < 30:
            summaries.append(chunk)
            continue
        try:
            summary = _summarize_chunk(
                chunk,
                max_len=max_per_chunk,
                min_len=max(20, target_per_chunk // 2),
            )
            summaries.append(summary)
        except Exception:
            summaries.append(chunk[:200])

    combined = " ".join(summaries)

    # Second pass if still too long
    if len(combined.split()) > target_words * 1.2:
        try:
            combined = _summarize_chunk(
                combined[:2000],
                max_len=target_words,
                min_len=int(target_words * 0.7),
            )
        except Exception:
            pass

    return combined


def generate_structured_script(
    raw_summary: str,
    video_title: str,
    target_minutes: float,
    api_key: str,
    language: str = "en",
) -> dict:
    """
    Use Groq (Llama 3.3 70B) to rewrite the raw summary into a
    professional structured script: Hook → Intro → Body.
    Returns dict with keys: hook, intro, body, full_script.
    """
    from groq import Groq

    client = Groq(api_key=api_key)
    target_words = int(target_minutes * AVG_WORDS_PER_MINUTE)

    # Map ISO language code to a human-readable name for the prompt
    lang_names = {
        "ur": "Urdu", "en": "English", "ar": "Arabic", "hi": "Hindi",
        "tr": "Turkish", "fa": "Persian (Farsi)", "pa": "Punjabi",
        "fr": "French", "es": "Spanish", "de": "German", "zh": "Chinese",
        "ja": "Japanese", "ko": "Korean", "ru": "Russian", "pt": "Portuguese",
        "it": "Italian", "nl": "Dutch", "pl": "Polish", "bn": "Bengali",
    }
    lang_code = (language or "en").lower()[:2]
    lang_name = lang_names.get(lang_code, f"the same language as the content ({lang_code})")
    is_rtl = lang_code in {"ur", "ar", "fa", "he"}
    rtl_note = " This is a right-to-left language — write naturally in its proper script." if is_rtl else ""

    system_prompt = f"""LANGUAGE RULE: Write 100% in {lang_name}.{rtl_note} Every single word must be in {lang_name}. Do NOT switch to English.

YOU ARE A SCRIPT EDITOR FOR DRAMATIC YOUTUBE REVIEWS.

Your only job is to take the user's existing video transcript and reformat it exactly into two sections, matching a specific dramatic storytelling style.

ABSOLUTE RULES YOU MUST NEVER BREAK:
- You MUST NOT invent, add, or imagine any topic, character, event, or fact not already in the user's script.
- You are enhancing and structuring the user's words. You are NOT creating new content.
- Every fact in your output must be traceable to a line in the provided transcript.
- You must match the dramatic, emotional tone of a high-quality drama review.

SECTION 1: **Hook + Intro**
- Write 1 paragraph bridging a powerful hook and the introduction to the video.
- Start with an emotional or suspenseful setup.
- Introduce the core conflict or emotional stakes using the characters/events from the transcript.
- End this section with 1 or 2 dramatic questions to build curiosity.

SECTION 2: **Story (Summary)**
- Write multiple paragraphs summarizing the actual narrative from the transcript.
- Focus on the character emotions, the plot progression, and the conflicts.
- Make it flow naturally like a professional storyteller narrating a drama sequence.
- End the final paragraph with an open question about what will happen next, creating a cliffhanger.

Format your response EXACTLY like this (use these exact bold headers):
**Hook + Intro**
<your hook and intro paragraph>

**Story (Summary)**
<your story summary paragraphs>"""

    user_prompt = f"""Video Title: "{video_title}"
Content Language: {lang_name}
Target Word Count: approximately {target_words} words
Target Duration: {target_minutes} minutes when spoken aloud

━━━ THIS IS THE ONLY CONTENT YOU MAY USE ━━━
{raw_summary}
━━━ END — DO NOT USE CONTENT FROM OUTSIDE THIS ━━━

Your task:
- Read the transcript above carefully.
- Write the **Hook + Intro** and **Story (Summary)** that restructure and retell THE ABOVE CONTENT ONLY.
- Match the emotional, suspenseful storytelling style of a dramatic review.
- Do NOT write about any other topic. Do NOT add new information. Do NOT replace the content.
- Every sentence you write must come from something mentioned in the transcript above.
- Write entirely in {lang_name}.
- Target: {target_words} words total across both sections.

Output the structured script now exactly using the bold headers:"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=target_words * 2,
        )

        content = response.choices[0].message.content.strip()

        # Parse sections
        hook_intro = ""
        body = ""
        current_section = None

        for line in content.splitlines():
            line_stripped = line.strip()
            
            # Identify headers (case sensitive or insensitive)
            if "**Hook + Intro**" in line_stripped or "[HOOK]" in line_stripped:
                current_section = "hook"
                continue
            elif "**Story (Summary)**" in line_stripped or "[BODY]" in line_stripped:
                current_section = "body"
                continue
            
            if current_section == "hook":
                hook_intro += line + "\n"
            elif current_section == "body":
                body += line + "\n"

        hook_intro = hook_intro.strip()
        body = body.strip()

        # Fallback: if parsing failed, treat the whole response as body
        if not hook_intro and not body:
            hook_intro = content[:250]
            body = content

        full_script = f"{hook_intro}\n\n{body}".strip()

        return {
            "hook": hook_intro,
            "intro": "",
            "body": body,
            "full_script": full_script,
        }

    except Exception as e:
        # Fallback: return raw summary structured manually
        sentences = raw_summary.split(". ")
        hook_text = ". ".join(sentences[:2]) + "." if len(sentences) >= 2 else raw_summary[:200]
        intro_text = ". ".join(sentences[2:4]) + "." if len(sentences) >= 4 else ""
        body_text = ". ".join(sentences[4:]) if len(sentences) > 4 else raw_summary

        return {
            "hook": hook_text,
            "intro": intro_text,
            "body": body_text,
            "full_script": raw_summary,
            "error": str(e),
        }


async def run_summarization(project_id: int, target_minutes: float, style_prompt: str):
    """Background summarization task."""
    from database import AsyncSessionLocal

    _summarize_progress[project_id] = {"step": "starting", "progress": 0}

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Transcript).where(Transcript.project_id == project_id))
        transcript = result.scalar_one_or_none()
        if not transcript:
            _summarize_progress[project_id] = {"step": "error", "error": "Transcript not found"}
            return

        project_result = await db.execute(select(Project).where(Project.id == project_id))
        project = project_result.scalar_one_or_none()

        target_words = int(target_minutes * AVG_WORDS_PER_MINUTE)
        _summarize_progress[project_id] = {"step": "summarizing", "progress": 10}

        # Languages that BART cannot handle (English-only model — hallucinates on these)
        BART_UNSUPPORTED = {"ur", "ar", "hi", "pa", "fa", "zh", "ja", "ko", "ru",
                            "tr", "bn", "gu", "te", "ta", "uk", "el", "he", "am"}
        detected_lang = (transcript.language or "en").lower()[:2]
        use_bart = detected_lang not in BART_UNSUPPORTED

        try:
            loop = asyncio.get_event_loop()
            video_title = project.name if project else "Video"

            if use_bart:
                # Step 1a: BART raw summarization (English/Latin-script only)
                _summarize_progress[project_id] = {"step": "summarizing", "progress": 15}
                raw_summary = await loop.run_in_executor(
                    None, summarize_text,
                    transcript.full_text,
                    target_words,
                    settings.summarization_model,
                    settings.device
                )
                _summarize_progress[project_id] = {"step": "structuring", "progress": 60}
            else:
                # Step 1b: Non-English — skip BART entirely.
                # BART is English-only; feeding Urdu/Arabic/Hindi into it causes hallucinations.
                # Pass the real transcript directly to Groq which handles all languages natively.
                print(f"[Summarize] Language '{detected_lang}' — skipping BART, using raw transcript for Groq")
                raw_summary = transcript.full_text
                _summarize_progress[project_id] = {"step": "structuring", "progress": 30}

            # Step 2: Groq structured script (Hook → Intro → Body)
            structured = {"hook": "", "intro": "", "body": raw_summary, "full_script": raw_summary}

            if settings.groq_api_key:
                _summarize_progress[project_id] = {"step": "structuring", "progress": 50}
                structured = await loop.run_in_executor(
                    None, generate_structured_script,
                    raw_summary,
                    video_title,
                    target_minutes,
                    settings.groq_api_key,
                    transcript.language or "en",
                )
            _summarize_progress[project_id] = {"step": "saving", "progress": 85}

            # Use full structured script as the primary summary_text (for TTS)
            final_script = structured.get("full_script") or raw_summary

            # Step 3: Create segment mapping proportionally
            original_segs = transcript.segments_list
            total_seg_duration = sum(s["end"] - s["start"] for s in original_segs) if original_segs else 60.0

            import re
            sentences = re.split(r'(?<=[.!?])\s+', final_script.strip())
            if not sentences:
                sentences = [final_script]

            summary_segs = []
            n = len(sentences)
            for i, sent in enumerate(sentences):
                start_ratio = i / n
                end_ratio = (i + 1) / n
                summary_segs.append({
                    "start": round(total_seg_duration * start_ratio, 2),
                    "end": round(total_seg_duration * end_ratio, 2),
                    "text": sent.strip()
                })

            # Save to DB
            transcript.summary_text = final_script
            transcript.summary_segments = json.dumps(summary_segs)
            transcript.target_minutes = int(target_minutes)
            transcript.hook = structured.get("hook", "")
            transcript.intro = structured.get("intro", "")
            transcript.structured_script = json.dumps({
                "hook": structured.get("hook", ""),
                "intro": structured.get("intro", ""),
                "body": structured.get("body", ""),
                "full_script": final_script,
            })

            if project:
                project.status = "summarized"
            await db.commit()

            _summarize_progress[project_id] = {"step": "done", "progress": 100}

        except Exception as e:
            _summarize_progress[project_id] = {"step": "error", "progress": 0, "error": str(e)}
            raise


@router.post("/summarize/{project_id}")
async def start_summarization(
    project_id: int,
    request: SummarizeRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Start the summarization pipeline."""
    result = await db.execute(select(Transcript).where(Transcript.project_id == project_id))
    transcript = result.scalar_one_or_none()
    if not transcript:
        raise HTTPException(status_code=404, detail="Transcript not found. Run analysis first.")

    background_tasks.add_task(
        run_summarization, project_id, request.target_minutes, request.style_prompt
    )
    return {
        "message": "Summarization started",
        "project_id": project_id,
        "target_minutes": request.target_minutes,
        "target_words": int(request.target_minutes * AVG_WORDS_PER_MINUTE),
        "groq_enabled": bool(settings.groq_api_key),
    }


@router.get("/summarize/{project_id}/status")
async def get_summarization_status(project_id: int):
    """Poll summarization progress."""
    return _summarize_progress.get(project_id, {"step": "not_started", "progress": 0})


@router.get("/script/{project_id}")
async def get_script(project_id: int, db: AsyncSession = Depends(get_db)):
    """Get the summarized script including structured sections."""
    result = await db.execute(select(Transcript).where(Transcript.project_id == project_id))
    transcript = result.scalar_one_or_none()
    if not transcript:
        raise HTTPException(status_code=404, detail="Transcript not found")

    # Parse structured script if available
    structured = {}
    if transcript.structured_script:
        try:
            structured = json.loads(transcript.structured_script)
        except Exception:
            pass

    return {
        "project_id": project_id,
        "summary_text": transcript.summary_text,
        "summary_segments": transcript.summary_segments_list,
        "target_minutes": transcript.target_minutes,
        "word_count": len(transcript.summary_text.split()) if transcript.summary_text else 0,
        "hook": structured.get("hook") or transcript.hook or "",
        "intro": structured.get("intro") or transcript.intro or "",
        "body": structured.get("body") or "",
        "structured_script": structured,
        "groq_enhanced": bool(transcript.hook),
    }


@router.put("/script/{project_id}")
async def update_script(
    project_id: int,
    request: ScriptUpdateRequest,
    db: AsyncSession = Depends(get_db)
):
    """Allow user to manually edit the summarized script."""
    result = await db.execute(select(Transcript).where(Transcript.project_id == project_id))
    transcript = result.scalar_one_or_none()
    if not transcript:
        raise HTTPException(status_code=404, detail="Transcript not found")
    transcript.summary_text = request.summary_text
    await db.commit()
    return {"message": "Script updated", "word_count": len(request.summary_text.split())}
