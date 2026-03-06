"""
SQLAlchemy ORM models for all entities.
"""
import json
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Text, Boolean,
    DateTime, ForeignKey, JSON
)
from sqlalchemy.orm import relationship
from database import Base


class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    original_filename = Column(String(512), nullable=False)
    video_path = Column(String(512), nullable=False)
    audio_path = Column(String(512), nullable=True)
    duration = Column(Float, nullable=True)        # seconds
    fps = Column(Float, nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    file_size = Column(Integer, nullable=True)     # bytes
    status = Column(String(64), default="uploaded")  # uploaded|analyzing|summarizing|editing|done
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    transcript = relationship("Transcript", back_populates="project", uselist=False, cascade="all, delete")
    scenes = relationship("Scene", back_populates="project", cascade="all, delete", order_by="Scene.start_time")
    timelines = relationship("Timeline", back_populates="project", cascade="all, delete")


class Transcript(Base):
    __tablename__ = "transcripts"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    full_text = Column(Text, nullable=False)
    # JSON list of {"start": float, "end": float, "text": str}
    segments = Column(Text, nullable=False, default="[]")
    language = Column(String(10), nullable=True)
    # Summarized version
    summary_text = Column(Text, nullable=True)
    summary_segments = Column(Text, nullable=True)
    target_minutes = Column(Integer, nullable=True)
    # Groq-structured script sections
    hook = Column(Text, nullable=True)          # attention-grabbing opener
    intro = Column(Text, nullable=True)          # engaging context-setter
    structured_script = Column(Text, nullable=True)  # full hook+intro+body script
    created_at = Column(DateTime, default=datetime.utcnow)

    project = relationship("Project", back_populates="transcript")

    @property
    def segments_list(self):
        return json.loads(self.segments) if self.segments else []

    @property
    def summary_segments_list(self):
        return json.loads(self.summary_segments) if self.summary_segments else []


class Scene(Base):
    __tablename__ = "scenes"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    scene_index = Column(Integer, nullable=False)
    start_time = Column(Float, nullable=False)   # seconds
    end_time = Column(Float, nullable=False)     # seconds
    keyframe_path = Column(String(512), nullable=True)

    project = relationship("Project", back_populates="scenes")


class Clip(Base):
    __tablename__ = "clips"

    id = Column(Integer, primary_key=True, index=True)
    timeline_id = Column(Integer, ForeignKey("timelines.id"), nullable=False)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    clip_index = Column(Integer, nullable=False)
    source_start = Column(Float, nullable=False)  # source video time
    source_end = Column(Float, nullable=False)
    timeline_start = Column(Float, nullable=False)  # position on timeline
    timeline_end = Column(Float, nullable=False)
    script_paragraph = Column(Text, nullable=True)
    keyframe_path = Column(String(512), nullable=True)
    video_segment_path = Column(String(512), nullable=True)
    # Adjustments (stored as JSON)
    adjustments = Column(Text, default='{"brightness":0,"contrast":0,"saturation":0,"sharpness":0,"filter":"none"}')
    transition_in = Column(String(64), default="fade")
    transition_out = Column(String(64), default="fade")
    layer = Column(Integer, default=0)  # 0=video, 1=overlay

    timeline = relationship("Timeline", back_populates="clips")

    @property
    def adjustments_dict(self):
        return json.loads(self.adjustments) if self.adjustments else {}


class Timeline(Base):
    __tablename__ = "timelines"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    version = Column(Integer, default=1)
    tts_audio_path = Column(String(512), nullable=True)
    aspect_ratio = Column(String(16), default="16:9")
    total_duration = Column(Float, nullable=True)
    draft_video_path = Column(String(512), nullable=True)
    export_video_path = Column(String(512), nullable=True)
    is_current = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    project = relationship("Project", back_populates="timelines")
    clips = relationship("Clip", back_populates="timeline", cascade="all, delete", order_by="Clip.clip_index")
