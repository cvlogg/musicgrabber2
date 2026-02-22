"""
MusicGrabber - Common Utilities

Filename sanitisation, title cleaning, track hashing, duplicate detection.
"""

import hashlib
import os
import re
import secrets
import threading
from pathlib import Path
from typing import Optional

from constants import AUDIO_EXTENSIONS, MAX_FILENAME_LENGTH
from settings import get_singles_dir, get_download_dir


def sanitize_filename(name: str) -> str:
    """Remove/replace characters that are problematic in filenames"""
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name[:MAX_FILENAME_LENGTH]


def is_valid_youtube_id(video_id: str) -> bool:
    """Basic validation for YouTube video/playlist IDs."""
    return bool(re.match(r'^[A-Za-z0-9_-]+$', video_id or ""))


def clean_title(title: str) -> str:
    """Clean up YouTube title by removing common suffixes and annotations"""
    # Remove common bracketed annotations (lyrics, remaster, official, etc.)
    title = re.sub(
        r'\s*[\(\[][^\)\]]*(?:official|lyrics?|lyric|audio|h[dq]|remaster|music\s*video)[^\)\]]*[\)\]]',
        '',
        title,
        flags=re.IGNORECASE
    )
    # Remove standalone "Official (Music) Video" text
    title = re.sub(r'\s*official\s*(music\s*)?video', '', title, flags=re.IGNORECASE)

    # Remove trailing dash-separated suffixes: "- Official Audio", "- Official Music Video", etc.
    title = re.sub(
        r'\s+[-–—]\s+(?:official\s+)?(?:music\s+)?(?:audio|video|lyric\s+video)\s*$',
        '',
        title,
        flags=re.IGNORECASE
    )

    # Strip any trailing dangling separators left after cleanup (e.g. "Title -")
    title = re.sub(r'\s+[-–—]\s*$', '', title)

    return title.strip()


def normalise_track_for_hash(artist: str, title: str) -> str:
    """Normalise artist/title for consistent hashing across playlist checks.

    Examples:
    "Daft Punk feat. Pharrell Williams | Get Lucky (Radio Edit)"
        -> "daft punk | get lucky"
    "SZA - Kill Bill [Official Lyric Video]"
        -> "sza | kill bill"
    """
    text = f"{artist}|{title}".lower()
    # Remove feat./ft./featuring and everything after
    text = re.sub(r'\s*(feat\.?|ft\.?|featuring)\s+.*?\|', '|', text)
    text = re.sub(r'\s*(feat\.?|ft\.?|featuring)\s+.*$', '', text)
    # Remove common suffixes in parens/brackets
    text = re.sub(r'\s*[\(\[].*?[\)\]]', '', text)
    # Remove punctuation except pipe separator
    text = re.sub(r'[^\w\s|]', '', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def hash_track(artist: str, title: str) -> str:
    """Generate hash for track identification in watched playlists"""
    normalised = normalise_track_for_hash(artist, title)
    return hashlib.sha256(normalised.encode()).hexdigest()[:16]


def extract_artist_title(full_title: str, channel: str) -> tuple[str, str]:
    """Try to extract artist and title from YouTube video title"""
    # Guard against None — some YouTube videos have no channel/uploader in their metadata
    full_title = full_title or "Unknown Title"
    channel = channel or "Unknown Artist"

    # Common patterns: "Artist -- Title", "Artist - Title", "Artist — Title", "Artist | Title"
    # Require spaces around hyphens to avoid splitting compound words like "T-4"
    patterns = [
        r'^(.+?)\s+--\s+(.+)$',
        r'^(.+?)\s+[-–—]\s+(.+)$',
        r'^(.+?)\s*\|\s*(.+)$',
    ]

    for pattern in patterns:
        match = re.match(pattern, full_title)
        if match:
            artist, title = match.groups()
            cleaned_title = clean_title(title)
            # If suffix stripping nukes the whole title (e.g. "... - Official Video"),
            # treat this split as invalid and try other patterns/fallback.
            if cleaned_title:
                return artist.strip(), cleaned_title

    # Fallback: use channel as artist, full title as title
    # Remove common channel suffixes like "VEVO", "Official", "- Topic"
    artist = re.sub(r'\s*[-–—]\s*Topic$', '', channel, flags=re.IGNORECASE)
    artist = re.sub(r'\s*(VEVO|Official|Music)$', '', artist, flags=re.IGNORECASE)
    fallback_title = clean_title(full_title)
    if not fallback_title:
        fallback_title = full_title.strip() or "Unknown Title"
    return artist.strip() or "Unknown Artist", fallback_title


def check_duplicate(artist: str, title: str) -> Optional[Path]:
    """Check if a track already exists in the library (any audio format).

    Checks the current download directory (artist subfolder or flat) and also
    peeks at the other layout so switching modes doesn't silently re-download.
    Handles both plain 'Title' stems and 'Artist - Title' stems (flat mode format).
    """
    try:
        sanitized_title = sanitize_filename(title)
        sanitized_artist = sanitize_filename(artist or "")
        artist_title_stem = f"{sanitized_artist} - {sanitized_title}" if sanitized_artist else sanitized_title

        # Check both possible locations so mode switches don't cause re-downloads.
        # Each entry is (directory, stems_to_check).
        checks = [
            (get_download_dir(artist), [sanitized_title, artist_title_stem]),  # current mode
            (get_singles_dir() / sanitize_filename(artist), [sanitized_title, artist_title_stem]),  # artist subfolder
            (get_singles_dir(), [sanitized_title, artist_title_stem]),          # flat
        ]
        seen = set()
        for d, stems in checks:
            d_str = str(d)
            if d_str in seen or not d.exists():
                continue
            seen.add(d_str)

            for stem in stems:
                for ext in AUDIO_EXTENSIONS:
                    expected_file = d / f"{stem}{ext}"
                    if expected_file.exists():
                        return expected_file

            # Case-insensitive fallback
            stem_lowers = {s.lower() for s in stems}
            for ext in AUDIO_EXTENSIONS:
                for file in d.glob(f"*{ext}"):
                    if file.stem.lower() in stem_lowers:
                        return file

        return None
    except Exception:
        return None


def set_file_permissions(file_path: Path):
    """Set file permissions to 666 (rw for all) for NAS/SMB compatibility"""
    try:
        os.chmod(file_path, 0o666)
    except OSError:
        pass  # Silently ignore permission errors (may not have rights)


def subsonic_auth_params(username: str, password: str) -> dict:
    """Build Subsonic API authentication parameters (for Navidrome)."""
    salt = secrets.token_hex(8)
    token = hashlib.md5(f"{password}{salt}".encode()).hexdigest()
    return {
        "u": username,
        "t": token,
        "s": salt,
        "v": "1.16.1",
        "c": "MusicGrabber",
        "f": "json",
    }


def spawn_daemon_thread(target, *args, **kwargs) -> None:
    """Start a daemon thread for background work."""
    thread = threading.Thread(target=target, args=args, kwargs=kwargs, daemon=True)
    thread.start()
