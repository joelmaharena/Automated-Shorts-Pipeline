# 🎬 Twitch-to-YouTube Shorts Pipeline

Automated pipeline that discovers viral Twitch clips, converts them to vertical format with face tracking and burned captions, then uploads to YouTube Shorts.

## ✨ Features

- **Smart Discovery** - Fetches top clips from any game/broadcaster in the last 24 hours
- **Face Tracking** - Intelligent 9:16 crop that follows the streamer's face with smooth camera movement
- **Auto Captions** - Whisper-powered transcription with TikTok-style burned subtitles
- **YouTube Integration** - Automatic upload with proper metadata and Shorts optimization
- **Deduplication** - History tracking prevents reprocessing clips
- **Quota Protection** - Built-in daily upload limits to avoid API quota issues
- **CRON Ready** - Designed for scheduled, headless execution

## 📋 Prerequisites

- Python 3.9+
- FFmpeg installed (`brew install ffmpeg` on macOS)
- Twitch Developer Account
- Google Cloud Project with YouTube Data API v3 enabled

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Note: face_recognition requires dlib which needs cmake
# On macOS: brew install cmake
# On Ubuntu: sudo apt-get install cmake
```

### 2. Configure Credentials

```bash
# Copy example config
cp .env.example .env

# Edit with your credentials
nano .env
```

#### Twitch API Setup
1. Go to [Twitch Developer Console](https://dev.twitch.tv/console/apps)
2. Create a new application
3. Copy **Client ID** and **Client Secret** to `.env`

#### YouTube API Setup
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Enable **YouTube Data API v3**
4. Create **OAuth 2.0 Client ID** (Desktop Application)
5. Download credentials as `client_secrets.json`

### 3. First Run (Authorization)

```bash
python main.py
```

On first run, a browser window will open for YouTube OAuth authorization. After authorizing, credentials are saved to `token.json` for future headless runs.

### 4. Schedule with CRON

```bash
# Edit crontab
crontab -e

# Run every 6 hours
0 */6 * * * cd /path/to/twitch_clips && /path/to/venv/bin/python main.py >> cron.log 2>&1
```

## ⚙️ Configuration Options

| Variable | Description | Default |
|----------|-------------|---------|
| `TARGET_GAME_ID` | Twitch game ID to fetch clips from | `509658` (Just Chatting) |
| `BROADCASTER_IDS` | Comma-separated broadcaster IDs (overrides game) | - |
| `CLIP_DURATION` | Max clip duration in seconds | `30` |
| `CLIPS_TO_FETCH` | Number of clips to fetch per run | `10` |
| `DAILY_UPLOAD_LIMIT` | Max uploads per day (quota protection) | `6` |
| `DEFAULT_PRIVACY_STATUS` | Upload privacy: private/public/unlisted | `private` |
| `WHISPER_MODEL` | Transcription model: tiny/base/small/medium/large | `base` |

### Common Game IDs

| Game | ID |
|------|-----|
| Just Chatting | `509658` |
| Call of Duty: Warzone | `512710` |
| Fortnite | `33214` |
| League of Legends | `21779` |
| Valorant | `516575` |
| Minecraft | `27471` |
| GTA V | `32982` |

## 📁 Project Structure

```
twitch_clips/
├── main.py              # Main pipeline script
├── requirements.txt     # Python dependencies
├── .env                 # Your configuration (not committed)
├── .env.example         # Configuration template
├── client_secrets.json  # YouTube OAuth credentials
├── token.json           # Saved YouTube auth token
├── history.json         # Processed clips history
├── upload_history.json  # Daily upload tracking
├── pipeline.log         # Execution logs
├── output/              # Processed videos ready for upload
└── temp/                # Temporary processing files
```

## 🔄 Pipeline Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TWITCH SOURCING                              │
│  1. Authenticate with Twitch API                                     │
│  2. Fetch top clips from game/broadcasters                          │
│  3. Filter: last 24 hours, deduplicate via history.json             │
└─────────────────────────────────────────────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      VIDEO PROCESSING                                │
│  1. Download clip with yt-dlp                                        │
│  2. Trim to max duration if needed                                   │
│  3. Transcribe audio with Whisper                                    │
│  4. Smart crop 16:9 → 9:16 with face tracking                        │
│  5. Burn TikTok-style captions                                       │
└─────────────────────────────────────────────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       YOUTUBE UPLOAD                                 │
│  1. Check daily quota (stop if limit reached)                        │
│  2. Upload with metadata: Title + #Shorts                            │
│  3. Add credit, tags, and description                                │
│  4. Update history and upload count                                  │
└─────────────────────────────────────────────────────────────────────┘
```

## 🛠️ Troubleshooting

### Face recognition installation fails
```bash
# macOS
brew install cmake dlib

# Ubuntu
sudo apt-get install cmake libdlib-dev

# Then reinstall
pip install face_recognition
```

### YouTube quota exceeded
The script automatically limits uploads. If you hit quota:
1. Wait until midnight Pacific Time (quota resets)
2. Reduce `DAILY_UPLOAD_LIMIT` in `.env`

### Whisper runs slow
- Use `tiny` or `base` model for faster processing
- For GPU acceleration, ensure CUDA is properly installed

### No clips found
- Verify `TARGET_GAME_ID` is correct
- Check if game has clips created in last 24 hours
- Try popular games like Just Chatting (`509658`)

## 📜 Legal Notice

**Important**: This tool should only be used for:
- Your own Twitch content
- Content you have explicit permission to use
- Content that falls under fair use

Respect content creators' rights and platform Terms of Service.

## 📄 License

MIT License - See LICENSE file for details.
