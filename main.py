#!/usr/bin/env python3
"""
Twitch-to-YouTube Shorts Automation Pipeline
============================================

A production-ready script that automates the entire lifecycle of converting
Twitch clips into YouTube Shorts with face tracking, caption burning, and
automatic uploading.

Author: Twitch Clips Automation
License: MIT
"""

import os
import sys
import json
import logging
import tempfile
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from collections import deque

import requests
import numpy as np
from dotenv import load_dotenv

# Video Processing
import yt_dlp
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
import whisper

# Face Recognition
import face_recognition
import cv2

# YouTube API
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

# ============================================================================
# CONFIGURATION
# ============================================================================

# Load environment variables
load_dotenv()

# Twitch API Configuration
TWITCH_CLIENT_ID = os.getenv("TWITCH_CLIENT_ID", "")
TWITCH_CLIENT_SECRET = os.getenv("TWITCH_CLIENT_SECRET", "")

# YouTube API Configuration
YOUTUBE_CLIENT_SECRETS_FILE = os.getenv("YOUTUBE_CLIENT_SECRETS_FILE", "client_secrets.json")
YOUTUBE_TOKEN_FILE = os.getenv("YOUTUBE_TOKEN_FILE", "token.json")
YOUTUBE_SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

# OpenAI Whisper Configuration
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")  # tiny, base, small, medium, large

# Processing Configuration
TARGET_GAME_ID = os.getenv("TARGET_GAME_ID", "509658")  # Default: Just Chatting
BROADCASTER_IDS = os.getenv("BROADCASTER_IDS", "").split(",") if os.getenv("BROADCASTER_IDS") else []
CLIP_DURATION = int(os.getenv("CLIP_DURATION", "30"))  # Max duration for shorts
CLIPS_TO_FETCH = int(os.getenv("CLIPS_TO_FETCH", "10"))  # Number of clips to fetch

# Upload Configuration
DAILY_UPLOAD_LIMIT = int(os.getenv("DAILY_UPLOAD_LIMIT", "6"))  # YouTube quota protection
DEFAULT_PRIVACY_STATUS = os.getenv("DEFAULT_PRIVACY_STATUS", "private")  # private, public, unlisted

# File Paths
HISTORY_FILE = Path(os.getenv("HISTORY_FILE", "history.json"))
UPLOAD_HISTORY_FILE = Path(os.getenv("UPLOAD_HISTORY_FILE", "upload_history.json"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))
TEMP_DIR = Path(os.getenv("TEMP_DIR", "temp"))

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "pipeline.log")

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("TwitchToShorts")


# ============================================================================
# HISTORY MANAGEMENT
# ============================================================================

class HistoryManager:
    """Manages clip processing and upload history for deduplication."""
    
    def __init__(self, history_file: Path, upload_history_file: Path):
        self.history_file = history_file
        self.upload_history_file = upload_history_file
        self._ensure_files_exist()
    
    def _ensure_files_exist(self):
        """Create history files if they don't exist."""
        if not self.history_file.exists():
            self.history_file.write_text(json.dumps({"processed": [], "metadata": {}}))
        if not self.upload_history_file.exists():
            self.upload_history_file.write_text(json.dumps({"uploads": {}}))
    
    def load_history(self) -> Dict[str, List]:
        """Load processing history from file."""
        try:
            return json.loads(self.history_file.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            return {"processed": [], "uploads": []}
    
    def save_history(self, history: Dict[str, List]):
        """Save processing history to file."""
        self.history_file.write_text(json.dumps(history, indent=2))
    
    def is_processed(self, clip_id: str) -> bool:
        """Check if a clip has already been processed."""
        history = self.load_history()
        return clip_id in history.get("processed", [])
    
    def mark_processed(self, clip_id: str, metadata: Optional[Dict] = None):
        """Mark a clip as processed."""
        history = self.load_history()
        if clip_id not in history["processed"]:
            history["processed"].append(clip_id)
            if metadata:
                if "metadata" not in history:
                    history["metadata"] = {}
                history["metadata"][clip_id] = metadata
            self.save_history(history)
            logger.info(f"Marked clip {clip_id} as processed")
    
    def get_today_upload_count(self) -> int:
        """Get the number of uploads performed today."""
        try:
            upload_history = json.loads(self.upload_history_file.read_text())
            # Handle corrupted or incorrectly structured files
            if not isinstance(upload_history, dict):
                upload_history = {"uploads": {}}
                self.upload_history_file.write_text(json.dumps(upload_history))
            uploads = upload_history.get("uploads", {})
            if not isinstance(uploads, dict):
                uploads = {}
        except (json.JSONDecodeError, FileNotFoundError):
            return 0
        
        today = datetime.now(timezone.utc).date().isoformat()
        return uploads.get(today, 0)
    
    def increment_upload_count(self):
        """Increment today's upload count."""
        try:
            upload_history = json.loads(self.upload_history_file.read_text())
            if not isinstance(upload_history, dict):
                upload_history = {"uploads": {}}
        except (json.JSONDecodeError, FileNotFoundError):
            upload_history = {"uploads": {}}
        
        # Ensure uploads is a dict
        if "uploads" not in upload_history or not isinstance(upload_history["uploads"], dict):
            upload_history["uploads"] = {}
        
        today = datetime.now(timezone.utc).date().isoformat()
        upload_history["uploads"][today] = upload_history["uploads"].get(today, 0) + 1
        self.upload_history_file.write_text(json.dumps(upload_history, indent=2))


# ============================================================================
# TWITCH API MODULE
# ============================================================================

class TwitchAPI:
    """Handles all Twitch API interactions."""
    
    BASE_URL = "https://api.twitch.tv/helix"
    AUTH_URL = "https://id.twitch.tv/oauth2/token"
    
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token: Optional[str] = None
        self.token_expires: Optional[datetime] = None
    
    def authenticate(self) -> bool:
        """Authenticate with Twitch and get OAuth token."""
        try:
            response = requests.post(
                self.AUTH_URL,
                params={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "grant_type": "client_credentials"
                },
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            self.access_token = data["access_token"]
            self.token_expires = datetime.now() + timedelta(seconds=data["expires_in"])
            
            logger.info("Successfully authenticated with Twitch API")
            return True
            
        except requests.RequestException as e:
            logger.error(f"Failed to authenticate with Twitch: {e}")
            return False
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        if not self.access_token or (self.token_expires and datetime.now() >= self.token_expires):
            self.authenticate()
        
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Client-Id": self.client_id
        }
    
    def get_clips(
        self,
        game_id: Optional[str] = None,
        broadcaster_ids: Optional[List[str]] = None,
        hours: int = 24,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Fetch top clips from Twitch.
        
        Args:
            game_id: Filter by specific game
            broadcaster_ids: List of broadcaster IDs to fetch clips from
            hours: Number of hours to look back
            limit: Maximum number of clips to fetch
        
        Returns:
            List of clip data dictionaries
        """
        clips = []
        
        # Calculate time range
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours)
        
        params = {
            "first": min(limit, 100),
            "started_at": start_time.isoformat(),
            "ended_at": end_time.isoformat()
        }
        
        try:
            # Fetch by broadcaster IDs if provided
            if broadcaster_ids:
                for broadcaster_id in broadcaster_ids:
                    if not broadcaster_id.strip():
                        continue
                    params["broadcaster_id"] = broadcaster_id.strip()
                    response = requests.get(
                        f"{self.BASE_URL}/clips",
                        headers=self._get_headers(),
                        params=params,
                        timeout=30
                    )
                    response.raise_for_status()
                    data = response.json()
                    clips.extend(data.get("data", []))
            
            # Fetch by game ID
            elif game_id:
                params["game_id"] = game_id
                response = requests.get(
                    f"{self.BASE_URL}/clips",
                    headers=self._get_headers(),
                    params=params,
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()
                clips.extend(data.get("data", []))
            
            # Sort by view count and limit
            clips.sort(key=lambda x: x.get("view_count", 0), reverse=True)
            clips = clips[:limit]
            
            logger.info(f"Fetched {len(clips)} clips from Twitch")
            return clips
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch clips from Twitch: {e}")
            return []
    
    def get_game_info(self, game_id: str) -> Optional[Dict[str, Any]]:
        """Get game information by ID."""
        try:
            response = requests.get(
                f"{self.BASE_URL}/games",
                headers=self._get_headers(),
                params={"id": game_id},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get("data"):
                return data["data"][0]
            return None
            
        except requests.RequestException as e:
            logger.error(f"Failed to get game info: {e}")
            return None


# ============================================================================
# VIDEO PROCESSING MODULE (The "TikTokifier")
# ============================================================================

class VideoProcessor:
    """Handles video downloading, processing, and caption burning."""
    
    def __init__(self, temp_dir: Path, output_dir: Path, whisper_model: str = "base"):
        self.temp_dir = temp_dir
        self.output_dir = output_dir
        self.whisper_model_name = whisper_model
        self.whisper_model = None
        
        # Ensure directories exist
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_whisper_model(self):
        """Lazy load whisper model."""
        if self.whisper_model is None:
            logger.info(f"Loading Whisper model: {self.whisper_model_name}")
            self.whisper_model = whisper.load_model(self.whisper_model_name)
    
    def download_clip(self, clip_url: str, clip_id: str) -> Optional[Path]:
        """
        Download a Twitch clip using yt-dlp.
        
        Args:
            clip_url: URL of the clip
            clip_id: Unique clip identifier
        
        Returns:
            Path to downloaded file or None if failed
        """
        output_path = self.temp_dir / f"{clip_id}.mp4"
        
        ydl_opts = {
            "format": "best[ext=mp4]/best",
            "outtmpl": str(output_path),
            "quiet": True,
            "no_warnings": True,
            "extractaudio": False,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([clip_url])
            
            if output_path.exists():
                logger.info(f"Successfully downloaded clip: {clip_id}")
                return output_path
            else:
                logger.error(f"Download completed but file not found: {output_path}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to download clip {clip_id}: {e}")
            return None
    
    def transcribe_audio(self, video_path: Path) -> List[Dict[str, Any]]:
        """
        Transcribe audio from video using Whisper.
        
        Args:
            video_path: Path to video file
        
        Returns:
            List of transcription segments with timing
        """
        self._load_whisper_model()
        
        try:
            # Extract audio to temp file
            audio_path = video_path.with_suffix(".wav")
            
            video = VideoFileClip(str(video_path))
            video.audio.write_audiofile(str(audio_path), verbose=False, logger=None)
            video.close()
            
            # Transcribe
            result = self.whisper_model.transcribe(str(audio_path), word_timestamps=True)
            
            # Clean up
            audio_path.unlink(missing_ok=True)
            
            segments = []
            for segment in result.get("segments", []):
                segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"].strip()
                })
            
            logger.info(f"Transcribed {len(segments)} segments")
            return segments
            
        except Exception as e:
            logger.error(f"Failed to transcribe audio: {e}")
            return []
    
    def detect_faces_in_frame(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in a single frame.
        
        Args:
            frame: Video frame as numpy array (RGB)
        
        Returns:
            List of face locations as (top, right, bottom, left)
        """
        try:
            # Resize for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            
            # Detect faces
            face_locations = face_recognition.face_locations(small_frame, model="hog")
            
            # Scale back to original size
            scaled_locations = [
                (top * 4, right * 4, bottom * 4, left * 4)
                for (top, right, bottom, left) in face_locations
            ]
            
            return scaled_locations
            
        except Exception as e:
            logger.warning(f"Face detection failed: {e}")
            return []
    
    def sample_video_frames(
        self,
        video_path: Path,
        num_samples: int = 10
    ) -> List[Tuple[float, np.ndarray]]:
        """
        Sample frames at evenly spaced timestamps throughout the video.
        
        Args:
            video_path: Path to video file
            num_samples: Number of frames to sample
        
        Returns:
            List of (timestamp, frame) tuples
        """
        samples = []
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            if total_frames < num_samples:
                num_samples = max(1, total_frames)
            
            # Calculate evenly spaced frame indices
            frame_indices = [
                int(i * (total_frames - 1) / (num_samples - 1)) if num_samples > 1 else 0
                for i in range(num_samples)
            ]
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    timestamp = frame_idx / fps if fps > 0 else 0
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    samples.append((timestamp, rgb_frame))
            
            cap.release()
            logger.info(f"Sampled {len(samples)} frames from video")
            
        except Exception as e:
            logger.error(f"Failed to sample video frames: {e}")
        
        return samples
    
    def find_optimal_crop_position(
        self,
        video_path: Path,
        target_width: int,
        frame_width: int,
        num_samples: int = 10
    ) -> int:
        """
        Sample video frames and find the optimal static crop position using median.
        
        Args:
            video_path: Path to video file
            target_width: Target crop width
            frame_width: Original frame width
            num_samples: Number of frames to sample
        
        Returns:
            Optimal X position for crop (clamped to valid range)
        """
        # Sample frames at evenly spaced intervals
        samples = self.sample_video_frames(video_path, num_samples)
        
        if not samples:
            # Fallback to center crop
            logger.warning("No samples available, using center crop")
            return (frame_width - target_width) // 2
        
        # Detect faces and collect center X positions
        face_x_positions = []
        
        for timestamp, frame in samples:
            face_locations = self.detect_faces_in_frame(frame)
            
            if face_locations:
                # Use the first (most prominent) face
                top, right, bottom, left = face_locations[0]
                face_center_x = (left + right) // 2
                face_x_positions.append(face_center_x)
                logger.debug(f"Frame at {timestamp:.2f}s: face detected at x={face_center_x}")
            else:
                logger.debug(f"Frame at {timestamp:.2f}s: no face detected")
        
        # Calculate crop position
        if face_x_positions:
            # Use MEDIAN to filter outliers
            median_x = int(np.median(face_x_positions))
            
            # Center the crop window on the median face position
            crop_x = median_x - (target_width // 2)
            
            logger.info(f"Found {len(face_x_positions)} faces across samples, median X: {median_x}")
        else:
            # No faces found - use center crop
            crop_x = (frame_width - target_width) // 2
            logger.warning("No faces detected in any samples, using center crop")
        
        # CLAMP to ensure crop window stays within video bounds
        crop_x = max(0, min(crop_x, frame_width - target_width))
        
        logger.info(f"Optimal crop position: x={crop_x} (clamped to 0-{frame_width - target_width})")
        return crop_x
    
    def smart_crop_video(
        self,
        video_path: Path,
        output_path: Path,
        target_aspect: Tuple[int, int] = (9, 16),
        num_samples: int = 10
    ) -> bool:
        """
        Convert 16:9 video to 9:16 using Sample & Lock static crop.
        
        This method samples ~10 frames, detects faces, calculates the median
        position, and applies a single static crop for smooth, jitter-free output.
        
        Args:
            video_path: Path to input video
            output_path: Path for output video
            target_aspect: Target aspect ratio (width, height)
            num_samples: Number of frames to sample for face detection
        
        Returns:
            True if successful
        """
        try:
            logger.info(f"Starting Sample & Lock crop for: {video_path}")
            
            # Load video with MoviePy for easier processing
            video = VideoFileClip(str(video_path))
            frame_width = video.w
            frame_height = video.h
            
            # Calculate target dimensions for 9:16
            target_height = frame_height
            target_width = int(target_height * target_aspect[0] / target_aspect[1])
            
            if target_width > frame_width:
                target_width = frame_width
                target_height = int(target_width * target_aspect[1] / target_aspect[0])
            
            logger.info(f"Original: {frame_width}x{frame_height} -> Target: {target_width}x{target_height}")
            
            # Find optimal crop position using sampling
            crop_x = self.find_optimal_crop_position(
                video_path,
                target_width,
                frame_width,
                num_samples
            )
            
            # Calculate Y position (center vertically)
            crop_y = (frame_height - target_height) // 2
            
            # Apply single static crop using MoviePy
            # crop(x1, y1, x2, y2) - coordinates of the crop region
            cropped_video = video.crop(
                x1=crop_x,
                y1=crop_y,
                x2=crop_x + target_width,
                y2=crop_y + target_height
            )
            
            # Write output
            cropped_video.write_videofile(
                str(output_path),
                codec="libx264",
                audio_codec="aac",
                verbose=False,
                logger=None
            )
            
            # Cleanup
            video.close()
            cropped_video.close()
            
            logger.info(f"Sample & Lock crop completed: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Smart crop failed: {e}")
            return False
    
    def burn_captions(
        self,
        video_path: Path,
        segments: List[Dict[str, Any]],
        output_path: Path
    ) -> bool:
        """
        Burn TikTok-style captions onto video.
        
        Args:
            video_path: Path to input video
            segments: List of transcription segments
            output_path: Path for output video
        
        Returns:
            True if successful
        """
        try:
            video = VideoFileClip(str(video_path))
            
            # Create text clips for each segment
            text_clips = []
            
            for segment in segments:
                # Split long text into multiple lines
                text = segment["text"]
                max_chars = 30
                
                if len(text) > max_chars:
                    words = text.split()
                    lines = []
                    current_line = []
                    
                    for word in words:
                        if len(" ".join(current_line + [word])) <= max_chars:
                            current_line.append(word)
                        else:
                            if current_line:
                                lines.append(" ".join(current_line))
                            current_line = [word]
                    
                    if current_line:
                        lines.append(" ".join(current_line))
                    
                    text = "\n".join(lines)
                
                # Create TikTok-style text clip
                txt_clip = TextClip(
                    text,
                    fontsize=40,
                    font="Arial-Bold",
                    color="white",
                    stroke_color="black",
                    stroke_width=3,
                    method="caption",
                    size=(video.w - 40, None),
                    align="center"
                )
                
                # Position at center-bottom
                txt_clip = txt_clip.set_position(("center", video.h * 0.75))
                txt_clip = txt_clip.set_start(segment["start"])
                txt_clip = txt_clip.set_duration(segment["end"] - segment["start"])
                
                text_clips.append(txt_clip)
            
            # Composite video with captions
            final_video = CompositeVideoClip([video] + text_clips)
            
            final_video.write_videofile(
                str(output_path),
                codec="libx264",
                audio_codec="aac",
                verbose=False,
                logger=None
            )
            
            # Cleanup
            video.close()
            final_video.close()
            for clip in text_clips:
                clip.close()
            
            logger.info(f"Captions burned successfully: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to burn captions: {e}")
            return False
    
    def process_clip(
        self,
        clip_url: str,
        clip_id: str,
        max_duration: int = 30
    ) -> Optional[Path]:
        """
        Full processing pipeline for a single clip.
        
        Args:
            clip_url: URL of the clip
            clip_id: Unique clip identifier
            max_duration: Maximum duration in seconds
        
        Returns:
            Path to processed video or None if failed
        """
        logger.info(f"Processing clip: {clip_id}")
        
        # Step 1: Download
        downloaded_path = self.download_clip(clip_url, clip_id)
        if not downloaded_path:
            return None
        
        # Step 2: Trim if needed
        try:
            video = VideoFileClip(str(downloaded_path))
            duration = video.duration
            
            if duration > max_duration:
                logger.info(f"Trimming video from {duration:.1f}s to {max_duration}s")
                video = video.subclip(0, max_duration)
                trimmed_path = self.temp_dir / f"{clip_id}_trimmed.mp4"
                video.write_videofile(str(trimmed_path), verbose=False, logger=None)
                video.close()
                downloaded_path.unlink(missing_ok=True)
                downloaded_path = trimmed_path
            else:
                video.close()
        except Exception as e:
            logger.error(f"Failed to check/trim video: {e}")
            return None
        
        # Step 3: Transcribe
        segments = self.transcribe_audio(downloaded_path)
        
        # Step 4: Smart crop to 9:16
        cropped_path = self.temp_dir / f"{clip_id}_cropped.mp4"
        if not self.smart_crop_video(downloaded_path, cropped_path):
            return None
        
        # Cleanup downloaded file
        downloaded_path.unlink(missing_ok=True)
        
        # Step 5: Burn captions
        final_path = self.output_dir / f"{clip_id}_final.mp4"
        
        if segments:
            if not self.burn_captions(cropped_path, segments, final_path):
                # If caption burning fails, use cropped video without captions
                shutil.copy(cropped_path, final_path)
        else:
            # No segments, just use cropped video
            shutil.copy(cropped_path, final_path)
        
        # Cleanup
        cropped_path.unlink(missing_ok=True)
        
        logger.info(f"Clip processing complete: {final_path}")
        return final_path


# ============================================================================
# YOUTUBE UPLOAD MODULE
# ============================================================================

class YouTubeUploader:
    """Handles YouTube API authentication and video uploads."""
    
    def __init__(
        self,
        client_secrets_file: str,
        token_file: str,
        scopes: List[str]
    ):
        self.client_secrets_file = client_secrets_file
        self.token_file = Path(token_file)
        self.scopes = scopes
        self.youtube = None
    
    def authenticate(self) -> bool:
        """
        Authenticate with YouTube API using OAuth 2.0.
        Saves credentials to token.json for subsequent headless runs.
        
        Returns:
            True if authentication successful
        """
        creds = None
        
        # Load existing credentials
        if self.token_file.exists():
            try:
                creds = Credentials.from_authorized_user_file(str(self.token_file), self.scopes)
            except Exception as e:
                logger.warning(f"Failed to load existing credentials: {e}")
        
        # Refresh or get new credentials
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as e:
                    logger.warning(f"Failed to refresh credentials: {e}")
                    creds = None
            
            if not creds:
                if not Path(self.client_secrets_file).exists():
                    logger.error(f"Client secrets file not found: {self.client_secrets_file}")
                    return False
                
                try:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.client_secrets_file,
                        self.scopes
                    )
                    creds = flow.run_local_server(port=0)
                except Exception as e:
                    logger.error(f"OAuth flow failed: {e}")
                    return False
            
            # Save credentials for future runs
            self.token_file.write_text(creds.to_json())
        
        try:
            self.youtube = build("youtube", "v3", credentials=creds)
            logger.info("Successfully authenticated with YouTube API")
            return True
        except Exception as e:
            logger.error(f"Failed to build YouTube service: {e}")
            return False
    
    def upload_video(
        self,
        video_path: Path,
        title: str,
        description: str,
        tags: List[str],
        privacy_status: str = "private",
        category_id: str = "20"  # Gaming category
    ) -> Optional[str]:
        """
        Upload a video to YouTube.
        
        Args:
            video_path: Path to video file
            title: Video title (will append #Shorts)
            description: Video description
            tags: List of tags
            privacy_status: private, public, or unlisted
            category_id: YouTube category ID
        
        Returns:
            Video ID if successful, None otherwise
        """
        if not self.youtube:
            if not self.authenticate():
                return None
        
        # Ensure title ends with #Shorts for Shorts recognition
        if "#Shorts" not in title:
            title = f"{title} #Shorts"
        
        # Truncate title if too long (100 char limit)
        if len(title) > 100:
            title = title[:96] + "..."
        
        body = {
            "snippet": {
                "title": title,
                "description": description,
                "tags": tags,
                "categoryId": category_id
            },
            "status": {
                "privacyStatus": privacy_status,
                "selfDeclaredMadeForKids": False
            }
        }
        
        try:
            media = MediaFileUpload(
                str(video_path),
                mimetype="video/mp4",
                resumable=True
            )
            
            request = self.youtube.videos().insert(
                part="snippet,status",
                body=body,
                media_body=media
            )
            
            response = None
            while response is None:
                status, response = request.next_chunk()
                if status:
                    logger.debug(f"Upload progress: {int(status.progress() * 100)}%")
            
            video_id = response.get("id")
            logger.info(f"Successfully uploaded video: https://youtube.com/shorts/{video_id}")
            return video_id
            
        except HttpError as e:
            if "quotaExceeded" in str(e):
                logger.error("YouTube API quota exceeded. Please try again tomorrow.")
            else:
                logger.error(f"YouTube API error: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to upload video: {e}")
            return None


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class TwitchToShortsPipeline:
    """Main orchestrator for the Twitch-to-Shorts pipeline."""
    
    def __init__(self):
        self.history_manager = HistoryManager(HISTORY_FILE, UPLOAD_HISTORY_FILE)
        self.twitch_api = TwitchAPI(TWITCH_CLIENT_ID, TWITCH_CLIENT_SECRET)
        self.video_processor = VideoProcessor(TEMP_DIR, OUTPUT_DIR, WHISPER_MODEL)
        self.youtube_uploader = YouTubeUploader(
            YOUTUBE_CLIENT_SECRETS_FILE,
            YOUTUBE_TOKEN_FILE,
            YOUTUBE_SCOPES
        )
        self.game_info: Optional[Dict[str, Any]] = None
    
    def check_configuration(self) -> bool:
        """Validate configuration before running."""
        errors = []
        
        if not TWITCH_CLIENT_ID:
            errors.append("TWITCH_CLIENT_ID not set")
        if not TWITCH_CLIENT_SECRET:
            errors.append("TWITCH_CLIENT_SECRET not set")
        if not Path(YOUTUBE_CLIENT_SECRETS_FILE).exists():
            errors.append(f"YouTube client secrets file not found: {YOUTUBE_CLIENT_SECRETS_FILE}")
        
        if errors:
            for error in errors:
                logger.error(f"Configuration error: {error}")
            return False
        
        return True
    
    def run(self) -> Dict[str, Any]:
        """
        Run the full pipeline.
        
        Returns:
            Summary of pipeline execution
        """
        logger.info("="*60)
        logger.info("Starting Twitch-to-Shorts Pipeline")
        logger.info("="*60)
        
        summary = {
            "started_at": datetime.now().isoformat(),
            "clips_fetched": 0,
            "clips_processed": 0,
            "clips_uploaded": 0,
            "errors": []
        }
        
        # Validate configuration
        if not self.check_configuration():
            summary["errors"].append("Configuration validation failed")
            return summary
        
        # Check YouTube quota
        today_uploads = self.history_manager.get_today_upload_count()
        if today_uploads >= DAILY_UPLOAD_LIMIT:
            logger.warning(f"Daily upload limit reached ({today_uploads}/{DAILY_UPLOAD_LIMIT})")
            summary["errors"].append("Daily upload limit reached")
            return summary
        
        remaining_uploads = DAILY_UPLOAD_LIMIT - today_uploads
        logger.info(f"Remaining uploads today: {remaining_uploads}")
        
        # Authenticate with Twitch
        if not self.twitch_api.authenticate():
            summary["errors"].append("Twitch authentication failed")
            return summary
        
        # Get game info for tags
        if TARGET_GAME_ID:
            self.game_info = self.twitch_api.get_game_info(TARGET_GAME_ID)
            if self.game_info:
                logger.info(f"Fetching clips for game: {self.game_info['name']}")
        
        # Fetch clips
        clips = self.twitch_api.get_clips(
            game_id=TARGET_GAME_ID if not BROADCASTER_IDS else None,
            broadcaster_ids=BROADCASTER_IDS if BROADCASTER_IDS else None,
            hours=24,
            limit=CLIPS_TO_FETCH
        )
        
        summary["clips_fetched"] = len(clips)
        
        if not clips:
            logger.warning("No clips found")
            return summary
        
        # Process clips
        uploads_this_run = 0
        
        for clip in clips:
            # Check if we've hit the limit
            if uploads_this_run >= remaining_uploads:
                logger.info("Daily upload limit reached during this run")
                break
            
            clip_id = clip["id"]
            clip_url = clip["url"]
            clip_title = clip["title"]
            broadcaster_name = clip["broadcaster_name"]
            
            # Skip if already processed
            if self.history_manager.is_processed(clip_id):
                logger.info(f"Skipping already processed clip: {clip_id}")
                continue
            
            logger.info(f"Processing clip: {clip_title} by {broadcaster_name}")
            
            try:
                # Process the clip
                processed_path = self.video_processor.process_clip(
                    clip_url,
                    clip_id,
                    max_duration=CLIP_DURATION
                )
                
                if not processed_path:
                    logger.error(f"Failed to process clip: {clip_id}")
                    summary["errors"].append(f"Processing failed: {clip_id}")
                    continue
                
                summary["clips_processed"] += 1
                
                # Prepare upload metadata
                game_name = self.game_info["name"] if self.game_info else "Gaming"
                
                description = (
                    f"Credit: {broadcaster_name} | {clip['url']}\n\n"
                    f"#twitch #gaming #{game_name.replace(' ', '').lower()} #shorts"
                )
                
                tags = ["Shorts", "Twitch", "Gaming", game_name, broadcaster_name]
                
                # Upload to YouTube
                video_id = self.youtube_uploader.upload_video(
                    processed_path,
                    clip_title,
                    description,
                    tags,
                    privacy_status=DEFAULT_PRIVACY_STATUS
                )
                
                if video_id:
                    summary["clips_uploaded"] += 1
                    uploads_this_run += 1
                    self.history_manager.increment_upload_count()
                    
                    # Mark as processed with metadata
                    self.history_manager.mark_processed(clip_id, {
                        "title": clip_title,
                        "broadcaster": broadcaster_name,
                        "youtube_id": video_id,
                        "processed_at": datetime.now().isoformat()
                    })
                    
                    logger.info(f"Successfully uploaded: {clip_title}")
                else:
                    summary["errors"].append(f"Upload failed: {clip_id}")
                
                # Cleanup processed file
                processed_path.unlink(missing_ok=True)
                
            except Exception as e:
                logger.error(f"Error processing clip {clip_id}: {e}")
                summary["errors"].append(f"Error: {clip_id} - {str(e)}")
        
        # Cleanup temp directory
        try:
            for file in TEMP_DIR.iterdir():
                file.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp directory: {e}")
        
        summary["completed_at"] = datetime.now().isoformat()
        
        logger.info("="*60)
        logger.info("Pipeline Complete")
        logger.info(f"  Clips fetched: {summary['clips_fetched']}")
        logger.info(f"  Clips processed: {summary['clips_processed']}")
        logger.info(f"  Clips uploaded: {summary['clips_uploaded']}")
        logger.info(f"  Errors: {len(summary['errors'])}")
        logger.info("="*60)
        
        return summary


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """Main entry point for the script."""
    try:
        pipeline = TwitchToShortsPipeline()
        summary = pipeline.run()
        
        # Exit with error code if there were failures
        if summary["errors"]:
            sys.exit(1)
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Fatal error in pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
