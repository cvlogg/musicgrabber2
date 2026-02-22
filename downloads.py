"""
MusicGrabber - Download Processing

Single track, playlist, and Soulseek download handlers.
Library scan triggers and M3U playlist generation.
"""

import base64
import json
import os
import sqlite3
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from constants import (
    AUDIO_EXTENSIONS,
    COOKIES_FILE, MUSIC_DIR,
    MONOCHROME_API_URL, MONOCHROME_COVER_BASE, TIMEOUT_MONOCHROME_API,
    TIMEOUT_YTDLP_INFO, TIMEOUT_YTDLP_DOWNLOAD, TIMEOUT_YTDLP_PLAYLIST,
    TIMEOUT_FFMPEG_CONVERT, TIMEOUT_HTTP_REQUEST,
    YTDLP_403_MAX_RETRIES, YTDLP_403_RETRY_DELAY,
    SLSKD_MAX_RETRIES, TIMEOUT_SLSKD_SEARCH,
    PLAYLIST_WAIT_MAX, PLAYLIST_WAIT_INTERVAL,
)
from db import db_conn
from metadata import lookup_metadata, fetch_lyrics, save_lyrics_file, apply_metadata_to_file
from notifications import send_notification
from settings import get_setting, get_setting_bool, get_setting_int, get_singles_dir, get_download_dir, get_playlists_dir, get_albums_dir
from slskd import (
    download_from_slskd, extract_track_info_from_path,
    search_slskd, should_retry_slskd_error,
)
from utils import (
    sanitize_filename,
    extract_artist_title,
    check_duplicate,
    is_valid_youtube_id,
    set_file_permissions,
    subsonic_auth_params,
)
from youtube import (
    _ytdlp_base_args, _is_ytdlp_403, _strip_cookies_args,
    _should_retry_without_cookies, _sleep_if_botted, _note_bot_block, _note_cookie_failure,
)


def _default_metadata_source(source: str) -> str:
    """Metadata fallback label when no AcoustID/MusicBrainz match is available."""
    source_name = (source or "youtube").lower()
    if source_name == "soundcloud":
        return "soundcloud_guessed"
    if source_name == "monochrome":
        return "monochrome_guessed"
    if source_name == "soulseek":
        return "soulseek_guessed"
    return "youtube_guessed"


def _safe_sanitized_title(title: str, fallback: str) -> str:
    """Return a filesystem-safe non-empty title for output templates/lookup."""
    cleaned = sanitize_filename(title or "")
    if cleaned:
        return cleaned
    fallback_cleaned = sanitize_filename(fallback or "")
    return fallback_cleaned or "Unknown Title"


def _output_stem(artist: str, title: str, fallback: str) -> str:
    """Return the output filename stem for a track.

    In flat (no-artist-subfolder) mode: 'Artist - Title'
    In organised mode: 'Title'

    The artist prefix in flat mode saves you from a directory full of files
    called 'Track 1.flac' with no idea who they belong to.
    """
    safe_title = _safe_sanitized_title(title, fallback)
    if not get_setting_bool("organise_by_artist", True):
        safe_artist = sanitize_filename(artist or "Unknown Artist")
        return f"{safe_artist} - {safe_title}"
    return safe_title


def _playlist_stem(artist: str, title: str, fallback: str) -> str:
    """Return 'Artist - Title' filename stem for tracks inside a playlist folder.

    Always flat — no organise_by_artist logic needed since the playlist folder
    itself provides the organisational context.
    """
    safe_artist = sanitize_filename(artist or "Unknown Artist")
    safe_title = _safe_sanitized_title(title, fallback)
    return f"{safe_artist} - {safe_title}"


def _find_downloaded_audio_or_raise(artist_dir: Path, sanitized_title: str) -> Path:
    """Find downloaded audio file by expected base name, or raise with useful context."""
    for ext in AUDIO_EXTENSIONS:
        candidate = artist_dir / f"{sanitized_title}{ext}"
        if candidate.exists():
            return candidate

    seen_files = []
    try:
        seen_files = [p.name for p in artist_dir.iterdir() if p.is_file()][:8]
    except OSError:
        pass
    raise Exception(
        f"Download completed but expected '{sanitized_title}' audio file not found in {artist_dir}. "
        f"Found files: {', '.join(seen_files) if seen_files else 'none'}"
    )


def trigger_navidrome_scan():
    """Trigger a Navidrome library scan via API"""
    navidrome_url = get_setting("navidrome_url")
    navidrome_user = get_setting("navidrome_user")
    navidrome_pass = get_setting("navidrome_pass")

    if not (navidrome_url and navidrome_user and navidrome_pass):
        return

    try:
        params = subsonic_auth_params(navidrome_user, navidrome_pass)

        with httpx.Client(timeout=TIMEOUT_HTTP_REQUEST) as client:
            client.get(
                f"{navidrome_url}/rest/startScan",
                params=params
            )
    except Exception:
        pass  # Non-critical, scan will happen on schedule anyway


def trigger_jellyfin_scan():
    """Trigger a Jellyfin library scan via API"""
    jellyfin_url = get_setting("jellyfin_url")
    jellyfin_api_key = get_setting("jellyfin_api_key")

    if not (jellyfin_url and jellyfin_api_key):
        return

    try:
        with httpx.Client(timeout=TIMEOUT_HTTP_REQUEST) as client:
            client.post(
                f"{jellyfin_url}/Library/Refresh",
                headers={"X-Emby-Token": jellyfin_api_key}
            )
    except Exception:
        pass  # Non-critical, scan will happen on schedule anyway


def probe_audio_quality(
    file_path: Path,
    source_info: tuple[str, int] | None = None,
) -> tuple[str | None, int]:
    """Use ffprobe to extract audio quality info.

    Returns (human_readable_string, bitrate_kbps). For lossless formats like
    FLAC the bitrate is reported as 0 (lossless always passes quality gates).

    source_info is an optional (codec_label, bitrate_kbps) tuple describing
    the original format before conversion. When the final file is FLAC but
    the source was lossy, the display string honestly notes the conversion
    (e.g. "FLAC (from MP3 128kbps)") and the returned bitrate is the SOURCE
    bitrate so the quality gate can reject lipstick-on-a-pig transcodes.
    """
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "a:0",
             "-show_entries", "stream=codec_name,bit_rate,sample_rate,bits_per_raw_sample",
             "-of", "json", str(file_path)],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return None, 0
        info = json.loads(result.stdout)
        stream = info.get("streams", [{}])[0]
        codec = (stream.get("codec_name") or "").upper()
        sample_rate = int(stream.get("sample_rate") or 0)
        bit_rate = int(stream.get("bit_rate") or 0)
        bit_depth = int(stream.get("bits_per_raw_sample") or 0)
        bitrate_kbps = bit_rate // 1000

        sample_khz = f"{sample_rate / 1000:.1f}kHz".replace(".0kHz", "kHz") if sample_rate else ""

        if codec == "FLAC":
            # Check if this FLAC was converted from a lossy source
            if source_info:
                src_codec, src_bitrate = source_info
                lossless_codecs = {"FLAC", "ALAC", "WAV", "PCM_S16LE", "PCM_S24LE"}
                if src_codec and src_codec.upper() not in lossless_codecs:
                    src_kbps = f" {src_bitrate}kbps" if src_bitrate else ""
                    label = f"FLAC (from {src_codec}{src_kbps})"
                    return label, src_bitrate  # Source bitrate for quality gate

            # Genuinely lossless
            parts = ["FLAC", sample_khz]
            if bit_depth:
                parts.append(f"{bit_depth}bit")
            return " ".join(p for p in parts if p), 0  # Lossless — always passes
        else:
            kbps = f"{bitrate_kbps}kbps" if bitrate_kbps else ""
            label = " ".join(p for p in [codec, kbps] if p) or None
            return label, bitrate_kbps
    except Exception:
        return None, 0


def _extract_source_format_from_info(info: dict) -> tuple[str, int]:
    """Extract the source audio codec and bitrate from yt-dlp info JSON.

    Returns (codec_label, bitrate_kbps). The top-level 'acodec' and 'abr'
    fields describe what yt-dlp actually selected to download, before any
    post-processing conversion.
    """
    acodec = (info.get("acodec") or "").strip().lower()
    abr = info.get("abr")  # Already in kbps (float or None)

    codec_map = {
        "mp3": "MP3", "aac": "AAC", "opus": "OPUS", "vorbis": "VORBIS",
        "flac": "FLAC", "alac": "ALAC", "pcm_s16le": "WAV", "pcm_s24le": "WAV",
        "mp4a.40.2": "AAC", "mp4a.40.5": "AAC",
    }

    codec_label = codec_map.get(acodec, acodec.upper() if acodec else "")
    bitrate_kbps = int(abr) if abr else 0

    return codec_label, bitrate_kbps


def _build_ytdlp_download_cmd(
    video_id: str,
    output_template: str,
    convert_to_flac: bool,
    source_url: str = None,
    use_cookies: bool = True,
) -> list[str]:
    """Build yt-dlp args for audio extraction, metadata, and thumbnail embedding.

    source_url overrides the default YouTube URL (used for SoundCloud etc.).
    use_cookies=False skips cookie/player-client args (not needed for SoundCloud).
    """
    if convert_to_flac:
        fmt = get_setting("audio_format", "flac")
        format_args = ["--audio-format", fmt if fmt in ("flac", "opus") else "flac"]
    else:
        format_args = []  # Keep original format from source
    base_args = _ytdlp_base_args() if use_cookies else []
    url = source_url or f"https://www.youtube.com/watch?v={video_id}"
    return [
        "yt-dlp",
        *base_args,
        "-f", "bestaudio/best",
        "-x",
        *format_args,
        "--audio-quality", "0",
        "--embed-metadata",
        "--embed-thumbnail",
        "--convert-thumbnails", "jpg",
        "--ppa", "ffmpeg:-c:v mjpeg -vf crop=\"'if(gt(ih,iw),iw,ih)':'if(gt(iw,ih),ih,iw)'\"",
        "--add-metadata",
        "--parse-metadata", "%(artist,channel,uploader)s:%(meta_artist)s",
        "--parse-metadata", "%(track,title)s:%(meta_title)s",
        "-o", output_template,
        "--no-warnings",
        url,
    ]


def _update_job(job_id: str, **fields) -> None:
    """Update job fields in the database."""
    if not fields:
        return
    columns = ", ".join(f"{key} = ?" for key in fields)
    values = list(fields.values())
    with db_conn() as conn:
        conn.execute(f"UPDATE jobs SET {columns} WHERE id = ?", (*values, job_id))
        conn.commit()


def _mark_watched_track_downloaded(job_id: str) -> None:
    with db_conn() as conn:
        conn.execute(
            "UPDATE watched_playlist_tracks SET downloaded_at = datetime('now') WHERE job_id = ?",
            (job_id,)
        )
        conn.commit()


def _cleanup_temp_files(artist_dir: Path, sanitized_title: str) -> int:
    """Remove yt-dlp .temp.* leftover files for a given track. Returns count removed."""
    removed = 0
    for temp_file in artist_dir.glob(f"{sanitized_title}.temp.*"):
        try:
            temp_file.unlink()
            removed += 1
            print(f"Cleaned up temp file: {temp_file.name}")
        except OSError:
            pass
    return removed


def _relocate_for_normalised_artist(audio_file: Path, old_artist: str, new_artist: str) -> Path:
    """Move a downloaded file to the correct artist directory after MusicBrainz normalisation.

    Because MusicBrainz actually knows how to spell, unlike half the uploaders on YouTube.
    Returns the new file path (or the original if no move was needed).
    """
    # Flat directory mode doesn't use artist names — nothing to shuffle
    if not get_setting_bool("organise_by_artist", True):
        return audio_file

    new_dir = get_download_dir(new_artist)
    old_dir = audio_file.parent

    if new_dir == old_dir:
        return audio_file

    new_dir.mkdir(parents=True, exist_ok=True)
    new_path = new_dir / audio_file.name

    # Don't trample an existing file — paranoia beats regret
    if new_path.exists():
        print(f"Artist normalisation: target already exists, skipping move: {new_path}")
        return audio_file

    audio_file.rename(new_path)
    print(f"Artist normalised: {old_dir.name}/{audio_file.name} -> {new_dir.name}/{audio_file.name}")
    set_file_permissions(new_path)

    # Relocate any lyrics file that tagged along
    old_lrc = audio_file.with_suffix(".lrc")
    if old_lrc.exists():
        new_lrc = new_path.with_suffix(".lrc")
        old_lrc.rename(new_lrc)
        set_file_permissions(new_lrc)

    # Tidy up the old directory if it's now gathering dust
    try:
        if old_dir.exists() and not any(old_dir.iterdir()):
            old_dir.rmdir()
            print(f"Removed empty artist directory: {old_dir.name}")
    except OSError:
        pass

    return new_path


def _is_permission_error(stderr: str) -> bool:
    """Check if yt-dlp failed due to a permission denied error on rename."""
    return "Permission denied" in stderr and ".temp." in stderr


def _run_ytdlp_with_retries(
    download_cmd: list[str],
    timeout_secs: int,
    has_cookies: bool
) -> tuple[subprocess.CompletedProcess | None, bool]:
    """Run yt-dlp with retry/backoff and optional cookie fallback."""
    download_result = None
    download_timed_out = False

    for attempt in range(1 + YTDLP_403_MAX_RETRIES):
        _sleep_if_botted()
        try:
            download_result = subprocess.run(
                download_cmd,
                capture_output=True,
                text=True,
                timeout=timeout_secs
            )
        except subprocess.TimeoutExpired:
            download_timed_out = True
            break

        if download_result.returncode == 0:
            break

        if _is_ytdlp_403(download_result.stderr) and attempt < YTDLP_403_MAX_RETRIES:
            print(f"YouTube 403 for {download_cmd[-1]}, retrying (attempt {attempt + 1})")
            time.sleep(YTDLP_403_RETRY_DELAY * (attempt + 1))
        else:
            break

    cookies_may_be_at_fault = has_cookies and download_result and _should_retry_without_cookies(download_result.stderr)

    if download_timed_out or (download_result and _should_retry_without_cookies(download_result.stderr)):
        _note_bot_block()

    if (download_timed_out or (download_result and download_result.returncode != 0)) and has_cookies:
        if download_timed_out or _should_retry_without_cookies(download_result.stderr):
            # Retry without cookies — if this succeeds, it confirms cookies were the problem.
            # If it also fails, the video itself is blocked (geo-lock, ContentID, etc.) and
            # cookies were innocent bystanders — don't penalise them.
            download_cmd_no_cookies = _strip_cookies_args(download_cmd)
            try:
                download_result = subprocess.run(
                    download_cmd_no_cookies,
                    capture_output=True,
                    text=True,
                    timeout=timeout_secs
                )
                download_timed_out = False
                if download_result.returncode == 0 and cookies_may_be_at_fault:
                    # Cookieless worked — so cookies were actively causing the 403.
                    # Disable them for a while so they don't break other downloads too.
                    print("Cookie-related 403 confirmed (cookieless retry succeeded) — disabling cookies temporarily")
                    _note_cookie_failure()
            except subprocess.TimeoutExpired:
                download_timed_out = True

    return download_result, download_timed_out


def create_bulk_playlist(bulk_import_id: str, playlist_name: str, expected_count: int, use_playlists_dir: bool = False):
    """Create an M3U playlist from a bulk import after all downloads complete

    Waits for all jobs with the matching playlist_name to complete, then generates the M3U file.
    When use_playlists_dir is True and playlists_subdir is configured, the M3U and its
    relative track paths are written into the Playlists folder instead of Singles.
    """
    # Wait for all downloads to complete (with timeout)
    max_wait_time = PLAYLIST_WAIT_MAX
    check_interval = PLAYLIST_WAIT_INTERVAL
    waited = 0

    while waited < max_wait_time:
        with db_conn() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) as total, SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed FROM jobs WHERE playlist_name = ?",
                (bulk_import_id,)
            )
            row = cursor.fetchone()

        total, completed = row
        completed = completed or 0

        # All downloads complete
        if completed >= expected_count or total == completed:
            break

        time.sleep(check_interval)
        waited += check_interval

    # Gather all successfully downloaded files
    with db_conn() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT artist, title FROM jobs WHERE playlist_name = ? AND status = 'completed' AND error IS NULL ORDER BY created_at",
            (bulk_import_id,)
        )
        jobs = [dict(row) for row in cursor.fetchall()]

    if not jobs:
        return  # No successful downloads

    # Determine whether to write into the Playlists folder
    playlists_dir = get_playlists_dir() if use_playlists_dir else None
    safe_playlist = sanitize_filename(playlist_name)

    # Build M3U playlist
    playlist_files = []
    for job in jobs:
        artist = job.get("artist", "Unknown")
        title = job.get("title", "Unknown")

        if playlists_dir:
            # Tracks were downloaded into Playlists/PlaylistName/
            track_dir = playlists_dir / safe_playlist
            stem = _playlist_stem(artist, title, title)
            for ext in ['.flac', '.opus', '.m4a', '.mp3', '.ogg', '.webm']:
                candidate = track_dir / f"{stem}{ext}"
                if candidate.exists():
                    # Path in M3U is relative to the M3U file (which sits one level up)
                    playlist_files.append(f"{safe_playlist}/{stem}{ext}")
                    break
        else:
            audio_file = check_duplicate(artist, title)
            if audio_file:
                rel_path = audio_file.relative_to(get_singles_dir())
                playlist_files.append(str(rel_path))

    if playlist_files:
        if playlists_dir:
            m3u_path = playlists_dir / f"{safe_playlist}.m3u"
            playlists_dir.mkdir(parents=True, exist_ok=True)
        else:
            m3u_path = get_singles_dir() / f"{safe_playlist}.m3u"
        with open(m3u_path, 'w', encoding='utf-8') as f:
            f.write("#EXTM3U\n")
            for file_path in playlist_files:
                f.write(f"{file_path}\n")
        set_file_permissions(m3u_path)


def rebuild_watched_playlist_m3u(playlist_id: str, playlist_name: str, use_playlists_dir: bool = False) -> Path | None:
    """Rebuild the M3U file for a watched playlist from all tracks marked as downloaded.

    Walks every downloaded track in the playlist, resolves the file on disk, and
    writes (or overwrites) the M3U. Called after each refresh cycle so the playlist
    file grows in step with the library.

    Returns the M3U path on success, None if no files could be resolved.
    """
    with db_conn() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT artist, title FROM watched_playlist_tracks
               WHERE playlist_id = ? AND downloaded_at IS NOT NULL
               ORDER BY first_seen""",
            (playlist_id,)
        ).fetchall()

    playlists_dir = get_playlists_dir() if use_playlists_dir else None
    safe_playlist = sanitize_filename(playlist_name)

    playlist_files = []
    for row in rows:
        artist, title = row["artist"], row["title"]
        if playlists_dir:
            track_dir = playlists_dir / safe_playlist
            stem = _playlist_stem(artist, title, title)
            for ext in ['.flac', '.opus', '.m4a', '.mp3', '.ogg', '.webm']:
                candidate = track_dir / f"{stem}{ext}"
                if candidate.exists():
                    playlist_files.append(f"{safe_playlist}/{stem}{ext}")
                    break
        else:
            audio_file = check_duplicate(artist, title)
            if audio_file:
                try:
                    rel_path = audio_file.relative_to(get_singles_dir())
                    playlist_files.append(str(rel_path))
                except ValueError:
                    pass  # File outside Singles dir — skip it

    if not playlist_files:
        return None

    if playlists_dir:
        m3u_path = playlists_dir / f"{safe_playlist}.m3u"
        playlists_dir.mkdir(parents=True, exist_ok=True)
    else:
        m3u_path = get_singles_dir() / f"{safe_playlist}.m3u"

    with open(m3u_path, 'w', encoding='utf-8') as f:
        f.write("#EXTM3U\n")
        for file_path in playlist_files:
            f.write(f"{file_path}\n")
    set_file_permissions(m3u_path)
    print(f"Watched playlist M3U updated: {m3u_path.name} ({len(playlist_files)} tracks)")
    return m3u_path


def process_playlist_download(job_id: str, playlist_id: str, playlist_name: str, convert_to_flac: bool = True, use_playlists_dir: bool = True):
    """Process a playlist download job.

    When use_playlists_dir is True and playlists_subdir is configured, tracks are saved to
    Playlists/PlaylistName/ with 'Artist - Title' naming. Otherwise falls back to Singles.
    """
    try:
        _update_job(job_id, status="downloading")

        # Get playlist information and extract all video IDs
        info_cmd = [
            "yt-dlp",
            *_ytdlp_base_args(),
            "--dump-json",
            "--flat-playlist",
            "--no-warnings",
            f"https://www.youtube.com/playlist?list={playlist_id}"
        ]

        info_result = subprocess.run(info_cmd, capture_output=True, text=True, timeout=TIMEOUT_YTDLP_PLAYLIST)
        if info_result.returncode != 0:
            raise Exception("Failed to get playlist info")

        # Parse all videos from playlist
        videos = []
        for line in info_result.stdout.strip().split('\n'):
            if not line:
                continue
            try:
                data = json.loads(line)
                if data.get("id"):
                    videos.append({
                        "id": data["id"],
                        "title": data.get("title", "Unknown"),
                        "channel": data.get("channel", data.get("uploader", "Unknown"))
                    })
            except json.JSONDecodeError:
                continue

        if not videos:
            raise Exception("No videos found in playlist")

        _update_job(job_id, total_tracks=len(videos))

        # Resolve the download directory for this playlist
        playlists_dir = get_playlists_dir() if use_playlists_dir else None
        safe_playlist = sanitize_filename(playlist_name)
        if playlists_dir:
            playlist_track_dir = playlists_dir / safe_playlist
            playlist_track_dir.mkdir(parents=True, exist_ok=True)

        # Download each video in the playlist
        downloaded_files = []
        completed_tracks = 0
        failed_tracks = 0
        skipped_tracks = 0
        has_cookies = COOKIES_FILE.exists() and COOKIES_FILE.stat().st_size > 0

        for video in videos:
            track_label = video.get("title", "Unknown")
            try:
                video_id = video["id"]

                # Get detailed video info
                detail_cmd = [
                    "yt-dlp",
                    *_ytdlp_base_args(),
                    "--dump-json",
                    "--no-warnings",
                    f"https://www.youtube.com/watch?v={video_id}"
                ]

                detail_result = subprocess.run(detail_cmd, capture_output=True, text=True, timeout=TIMEOUT_YTDLP_INFO)
                if detail_result.returncode != 0:
                    failed_tracks += 1
                    continue

                info = json.loads(detail_result.stdout)
                full_title = info.get("title", "Unknown")
                channel = info.get("channel", info.get("uploader", "Unknown"))
                artist, title = extract_artist_title(full_title, channel)

                # Check for duplicates — still add to M3U even if we're not downloading
                existing_file = check_duplicate(artist, title)
                if existing_file:
                    skipped_tracks += 1
                    if playlists_dir:
                        stem = _playlist_stem(artist, title, video_id)
                        downloaded_files.append(f"{safe_playlist}/{stem}{existing_file.suffix}")
                    else:
                        downloaded_files.append(str(existing_file.relative_to(get_singles_dir())))
                    continue

                # Create download directory
                if playlists_dir:
                    artist_dir = playlist_track_dir
                    safe_title = _playlist_stem(artist, title, video_id)
                else:
                    artist_dir = get_download_dir(artist)
                    artist_dir.mkdir(parents=True, exist_ok=True)
                    safe_title = _output_stem(artist, title, video_id)
                output_template = str(artist_dir / f"{safe_title}.%(ext)s")
                download_cmd = _build_ytdlp_download_cmd(video_id, output_template, convert_to_flac)

                download_result, download_timed_out = _run_ytdlp_with_retries(
                    download_cmd,
                    TIMEOUT_YTDLP_DOWNLOAD,
                    has_cookies
                )

                if download_timed_out or not download_result or download_result.returncode != 0:
                    # Permission denied on temp file rename — clean up and retry once
                    stderr = download_result.stderr if download_result else ""
                    if not download_timed_out and download_result and _is_permission_error(stderr):
                        cleaned = _cleanup_temp_files(artist_dir, safe_title)
                        if cleaned:
                            print(f"Retrying playlist track after cleaning {cleaned} temp file(s)")
                            download_result, download_timed_out = _run_ytdlp_with_retries(
                                download_cmd, TIMEOUT_YTDLP_DOWNLOAD, has_cookies
                            )
                    if download_timed_out or not download_result or download_result.returncode != 0:
                        failed_tracks += 1
                        continue

                try:
                    audio_file = _find_downloaded_audio_or_raise(artist_dir, safe_title)
                except Exception as e:
                    print(f"Playlist track output lookup failed: {e}")
                    failed_tracks += 1
                    continue

                # Set permissions for NAS/SMB compatibility
                set_file_permissions(audio_file)

                # Try to enrich metadata with AcoustID fingerprinting, then MusicBrainz
                mb_metadata = lookup_metadata(artist, title, audio_file)
                if mb_metadata:
                    mb_artist = mb_metadata.get("artist", artist)
                    mb_title = mb_metadata.get("title", title)
                    apply_metadata_to_file(
                        audio_file, mb_artist, mb_title,
                        mb_metadata.get("album", ""),
                        mb_metadata.get("year")
                    )
                    # Use canonical artist/title from MusicBrainz
                    if mb_artist != artist:
                        audio_file = _relocate_for_normalised_artist(audio_file, artist, mb_artist)
                        artist = mb_artist
                    if mb_title != title:
                        title = mb_title
                else:
                    apply_metadata_to_file(audio_file, artist, title)

                # Fetch and save lyrics
                lyrics = fetch_lyrics(artist, title)
                if lyrics:
                    save_lyrics_file(audio_file, lyrics)

                if playlists_dir:
                    downloaded_files.append(f"{safe_playlist}/{safe_title}{audio_file.suffix}")
                else:
                    downloaded_files.append(str(audio_file.relative_to(get_singles_dir())))
                completed_tracks += 1

            except Exception as track_error:
                # Track this individual failure and continue
                print(f"Playlist track failed: {track_label} - {track_error}")
                failed_tracks += 1
            finally:
                _update_job(
                    job_id,
                    completed_tracks=completed_tracks,
                    failed_tracks=failed_tracks,
                    skipped_tracks=skipped_tracks
                )

        # Generate M3U playlist file
        if downloaded_files:
            if playlists_dir:
                m3u_path = playlists_dir / f"{safe_playlist}.m3u"
            else:
                m3u_path = get_singles_dir() / f"{safe_playlist}.m3u"
            with open(m3u_path, 'w', encoding='utf-8') as f:
                f.write("#EXTM3U\n")
                for file_path in downloaded_files:
                    f.write(f"{file_path}\n")
            set_file_permissions(m3u_path)

            _update_job(job_id, m3u_path=str(m3u_path.relative_to(MUSIC_DIR)))

        # Trigger library rescans if configured
        trigger_navidrome_scan()
        trigger_jellyfin_scan()

        # Update job status based on results
        final_status = "completed"
        error_message = None

        if failed_tracks:
            final_status = "completed_with_errors"
            error_message = f"{failed_tracks} track(s) failed"
            if skipped_tracks:
                error_message += f", {skipped_tracks} skipped (duplicates)"

        if final_status == "completed":
            error_message = None

        _update_job(
            job_id,
            status=final_status,
            error=error_message,
            completed_at=datetime.now(timezone.utc).isoformat()
        )

        # Send notification for playlist
        send_notification(
            notification_type="playlist",
            title=playlist_name,
            playlist_name=playlist_name,
            source="youtube",
            status=final_status,
            error=error_message,
            track_count=len(videos),
            failed_count=failed_tracks,
            skipped_count=skipped_tracks
        )

    except Exception as e:
        _update_job(job_id, status="failed", error=str(e), completed_at=datetime.now(timezone.utc).isoformat())

        # Send notification for playlist failure
        send_notification(
            notification_type="error",
            title=playlist_name,
            playlist_name=playlist_name,
            source="youtube",
            status="failed",
            error=str(e)
        )



def process_slskd_download(job_id: str, username: str, filename: str, artist: str, title: str, convert_to_flac: bool = True):
    """Process a Soulseek download job via slskd"""
    try:
        _update_job(job_id, status="downloading")

        # If artist/title not provided, extract from filename
        if not artist or not title:
            artist, title = extract_track_info_from_path(filename)

        # Update job with extracted info (store slskd peer as uploader for blacklist)
        _update_job(job_id, title=title, artist=artist, uploader=username)

        # Check for duplicates
        existing_file = check_duplicate(artist, title)
        if existing_file:
            _update_job(
                job_id,
                status="completed",
                completed_at=datetime.now(timezone.utc).isoformat(),
                error=f"Already exists: {existing_file.name}"
            )
            _mark_watched_track_downloaded(job_id)
            return

        # Create download directory (with or without artist subfolder)
        artist_dir = get_download_dir(artist)
        artist_dir.mkdir(parents=True, exist_ok=True)

        # Download from slskd with retries on common queue/abort failures
        downloaded_file = None
        attempts = 0
        tried_candidates = set()
        candidate_queue = [(username, filename)]
        last_error = None

        while candidate_queue:
            cand_username, cand_filename = candidate_queue.pop(0)
            if (cand_username, cand_filename) in tried_candidates:
                continue
            tried_candidates.add((cand_username, cand_filename))

            try:
                downloaded_file = download_from_slskd(cand_username, cand_filename, artist_dir)
                break
            except Exception as e:
                last_error = str(e)
                print(f"slskd download attempt failed: {last_error}")
                if attempts >= SLSKD_MAX_RETRIES or not should_retry_slskd_error(last_error):
                    break
                attempts += 1

                # Refresh candidates from a new search if we don't have any left
                if not candidate_queue:
                    retry_query = f"{artist} {title}".strip()
                    retry_results = search_slskd(retry_query, timeout_secs=TIMEOUT_SLSKD_SEARCH)
                    for r in retry_results:
                        candidate = (r.get("slskd_username", ""), r.get("slskd_filename", ""))
                        if candidate[0] and candidate[1] and candidate not in tried_candidates:
                            candidate_queue.append(candidate)

        if not downloaded_file:
            raise Exception(last_error or "Soulseek download failed")

        if not downloaded_file or not downloaded_file.exists():
            raise Exception("Download completed but file not found")

        # Rename to our standard naming
        sanitized_title = _output_stem(artist, title, Path(filename).stem or job_id)
        source_ext = downloaded_file.suffix.lower()

        # Probe the source file BEFORE conversion so we know the real quality
        source_format_info = None
        if convert_to_flac and source_ext != '.flac':
            src_quality_str, src_bitrate = probe_audio_quality(downloaded_file)
            if src_quality_str:
                src_codec = src_quality_str.split()[0]
                source_format_info = (src_codec, src_bitrate)

        # Determine final filename
        audio_fmt = get_setting("audio_format", "flac") if convert_to_flac else None
        if audio_fmt not in ("flac", "opus"):
            audio_fmt = "flac"

        target_ext = f".{audio_fmt}" if audio_fmt else source_ext
        needs_convert = convert_to_flac and source_ext != target_ext

        if needs_convert:
            # Convert to the target format
            ffmpeg_codec = "flac" if audio_fmt == "flac" else "libopus"
            final_file = artist_dir / f"{sanitized_title}{target_ext}"
            convert_cmd = ["ffmpeg", "-y", "-i", str(downloaded_file), "-c:a", ffmpeg_codec]
            if audio_fmt == "opus":
                convert_cmd += ["-b:a", "320k"]
            convert_cmd.append(str(final_file))
            result = subprocess.run(convert_cmd, capture_output=True, timeout=TIMEOUT_FFMPEG_CONVERT)
            if result.returncode == 0:
                downloaded_file.unlink()
            else:
                # Conversion failed, keep original with new name
                final_file = artist_dir / f"{sanitized_title}{source_ext}"
                downloaded_file.rename(final_file)
        else:
            # Already in target format (or no conversion requested), just rename
            final_file = artist_dir / f"{sanitized_title}{source_ext}"
            if downloaded_file != final_file:
                downloaded_file.rename(final_file)

        # Set permissions for NAS/SMB compatibility
        set_file_permissions(final_file)

        # Probe audio quality (with source info so FLAC-from-lossy is reported honestly)
        audio_quality, bitrate_kbps = probe_audio_quality(final_file, source_info=source_format_info)
        min_bitrate = get_setting_int("min_audio_bitrate", 0)
        if min_bitrate and bitrate_kbps and bitrate_kbps < min_bitrate:
            final_file.unlink(missing_ok=True)
            raise Exception(f"Audio quality too low ({bitrate_kbps}kbps, minimum is {min_bitrate}kbps)")

        # Apply metadata (AcoustID fingerprinting first, then text-based MusicBrainz fallback)
        metadata_source = _default_metadata_source("soulseek")
        mb_metadata = lookup_metadata(artist, title, final_file)
        if mb_metadata:
            metadata_source = mb_metadata.get("metadata_source", metadata_source)
            mb_artist = mb_metadata.get("artist", artist)
            mb_title = mb_metadata.get("title", title)
            apply_metadata_to_file(
                final_file, mb_artist, mb_title,
                mb_metadata.get("album", ""),
                mb_metadata.get("year")
            )
            # Use canonical artist/title from MusicBrainz
            if mb_artist != artist:
                final_file = _relocate_for_normalised_artist(final_file, artist, mb_artist)
                artist = mb_artist
            if mb_title != title:
                title = mb_title
            _update_job(job_id, artist=artist, title=title)
        else:
            apply_metadata_to_file(final_file, artist, title)

        # Fetch and save lyrics
        lyrics = fetch_lyrics(artist, title)
        if lyrics:
            save_lyrics_file(final_file, lyrics)
            print(f"Saved lyrics for {artist} - {title}")
        else:
            print(f"No lyrics found for {artist} - {title}")

        # Trigger library rescans if configured
        trigger_navidrome_scan()
        trigger_jellyfin_scan()

        # Update job status
        _update_job(
            job_id,
            status="completed",
            error=None,
            audio_quality=audio_quality,
            metadata_source=metadata_source,
            completed_at=datetime.now(timezone.utc).isoformat()
        )
        _mark_watched_track_downloaded(job_id)

        print(f"slskd: Successfully downloaded {artist} - {title}")

        # Send notification for Soulseek single
        send_notification(
            notification_type="single",
            title=title,
            artist=artist,
            source="soulseek",
            status="completed"
        )

    except Exception as e:
        print(f"slskd download failed: {e}")
        _update_job(job_id, status="failed", error=str(e), completed_at=datetime.now(timezone.utc).isoformat())

        # Send notification for Soulseek failure
        send_notification(
            notification_type="error",
            title=title,
            artist=artist,
            source="soulseek",
            status="failed",
            error=str(e)
        )



def _monochrome_cover_url(cover_uuid: str) -> str:
    """Turn a Tidal cover UUID into a CDN thumbnail URL."""
    if not cover_uuid:
        return ""
    return f"{MONOCHROME_COVER_BASE}/{cover_uuid.replace('-', '/')}/640x640.jpg"


def _download_monochrome_direct(track_id: str, output_path: Path) -> None:
    """Download a FLAC directly from the Monochrome/Tidal API.

    No yt-dlp, no messing about — just a straight FLAC off the CDN.
    Tries LOSSLESS first; falls back to HIGH on 403 (some tracks are restricted
    at the lossless tier). Raises on any other failure.
    """
    quality_attempts = ["LOSSLESS", "HIGH"]
    resp = None
    with httpx.Client(timeout=TIMEOUT_MONOCHROME_API) as client:
        for quality in quality_attempts:
            resp = client.get(
                f"{MONOCHROME_API_URL}/track/",
                params={"id": track_id, "quality": quality},
            )
            if resp.status_code != 403:
                break
            print(f"Monochrome: {quality} quality returned 403 for track {track_id}, trying next tier...")
    resp.raise_for_status()
    data = resp.json().get("data") or {}
    if not data.get("manifest"):
        raise Exception(f"No stream manifest returned for Monochrome track {track_id}")

    manifest = json.loads(base64.b64decode(data["manifest"]))
    encryption = manifest.get("encryptionType", "NONE")
    if encryption != "NONE":
        raise Exception(f"Monochrome track {track_id} is encrypted ({encryption}) — cannot download")

    urls = manifest.get("urls") or []
    if not urls:
        raise Exception(f"Empty URL list in manifest for Monochrome track {track_id}")

    # Stream the FLAC to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", urls[0], timeout=120) as stream_resp:
        stream_resp.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in stream_resp.iter_bytes(chunk_size=8192):
                f.write(chunk)


def _embed_monochrome_cover(audio_file: Path, cover_uuid: str) -> None:
    """Download cover art from Tidal CDN and embed it in a FLAC file."""
    if not cover_uuid:
        return
    try:
        from mutagen.flac import FLAC, Picture

        cover_url = _monochrome_cover_url(cover_uuid)
        resp = httpx.get(cover_url, timeout=10)
        resp.raise_for_status()

        pic = Picture()
        pic.type = 3  # Cover (front)
        pic.mime = "image/jpeg"
        pic.data = resp.content

        audio = FLAC(str(audio_file))
        audio.clear_pictures()
        audio.add_picture(pic)
        audio.save()
    except Exception as e:
        # Non-critical — the track still plays fine without cover art
        print(f"Monochrome cover embed failed: {e}")


def _get_monochrome_track_info(track_id: str) -> dict | None:
    """Fetch track metadata from the Monochrome API info endpoint."""
    try:
        resp = httpx.get(
            f"{MONOCHROME_API_URL}/info/",
            params={"id": track_id},
            timeout=TIMEOUT_MONOCHROME_API,
        )
        resp.raise_for_status()
        return resp.json().get("data")
    except Exception as e:
        print(f"Monochrome track info lookup failed: {e}")
        return None


def _process_monochrome_download(job_id: str, track_id: str, convert_to_flac: bool = True):
    """Download a track directly from Monochrome/Tidal — no yt-dlp needed.

    The API gives us proper metadata (artist, album, ISRC) so we don't need
    to guess from dodgy YouTube titles. The audio is genuine lossless FLAC
    straight off the Tidal CDN.
    """
    source_label = "monochrome"
    artist = None
    title = track_id

    try:
        _update_job(job_id, status="downloading")

        # Get track metadata from the API — artist, title, album, the lot
        info = _get_monochrome_track_info(track_id)
        if not info:
            raise Exception(f"Failed to get track info for Monochrome track {track_id}")

        title = info.get("title", "Unknown")
        artist_obj = info.get("artist") or {}
        artist = artist_obj.get("name", "Unknown")
        album_obj = info.get("album") or {}
        album_title = album_obj.get("title", "Singles")
        cover_uuid = album_obj.get("cover", "")
        isrc = info.get("isrc", "")

        _update_job(job_id, title=title, artist=artist, uploader=artist)

        # Duplicate check
        existing_file = check_duplicate(artist, title)
        if existing_file:
            _update_job(
                job_id,
                status="completed",
                completed_at=datetime.now(timezone.utc).isoformat(),
                error=f"Already exists: {existing_file.name}"
            )
            _mark_watched_track_downloaded(job_id)
            return

        # Create download directory
        artist_dir = get_download_dir(artist)
        artist_dir.mkdir(parents=True, exist_ok=True)

        safe_title = _output_stem(artist, title, track_id)
        output_path = artist_dir / f"{safe_title}.flac"

        # Download the FLAC
        _download_monochrome_direct(track_id, output_path)

        if not output_path.exists():
            raise Exception("Download completed but FLAC file not found")

        set_file_permissions(output_path)

        # Embed cover art from Tidal CDN
        _embed_monochrome_cover(output_path, cover_uuid)

        # Probe audio quality — this is genuine lossless, no transcode shenanigans
        audio_quality, bitrate_kbps = probe_audio_quality(output_path)
        min_bitrate = get_setting_int("min_audio_bitrate", 0)
        if min_bitrate and bitrate_kbps and bitrate_kbps < min_bitrate:
            output_path.unlink(missing_ok=True)
            raise Exception(f"Audio quality too low ({bitrate_kbps}kbps, minimum is {min_bitrate}kbps)")

        # Metadata enrichment — Tidal already gave us artist/title/album, which is authoritative.
        # We only use MusicBrainz to fill in the year (which Tidal doesn't provide).
        # We deliberately don't let MusicBrainz overwrite artist/title/album here — it has
        # a nasty habit of matching a live recording or remaster and silently making things worse.
        metadata_source = "monochrome_api"
        mb_metadata = lookup_metadata(artist, title, output_path)
        year = mb_metadata.get("year") if mb_metadata else None
        apply_metadata_to_file(output_path, artist, title, album_title, year)

        # Lyrics
        lyrics = fetch_lyrics(artist, title)
        if lyrics:
            save_lyrics_file(output_path, lyrics)
            print(f"Saved lyrics for {artist} - {title}")
        else:
            print(f"No lyrics found for {artist} - {title}")

        # Library scans
        trigger_navidrome_scan()
        trigger_jellyfin_scan()

        # Done!
        _update_job(
            job_id,
            status="completed",
            error=None,
            audio_quality=audio_quality,
            metadata_source=metadata_source,
            completed_at=datetime.now(timezone.utc).isoformat()
        )
        _mark_watched_track_downloaded(job_id)

        print(f"Monochrome: Downloaded {artist} - {title} (lossless FLAC)")

        send_notification(
            notification_type="single",
            title=title,
            artist=artist,
            source=source_label,
            status="completed"
        )

    except Exception as e:
        print(f"Monochrome download failed: {e}")
        _update_job(job_id, status="failed", error=str(e), completed_at=datetime.now(timezone.utc).isoformat())

        send_notification(
            notification_type="error",
            title=title,
            artist=artist,
            source=source_label,
            status="failed",
            error=str(e)
        )


def _fetch_album_info(album_id: str) -> dict | None:
    """Fetch album metadata and track list from the Monochrome API."""
    try:
        resp = httpx.get(
            f"{MONOCHROME_API_URL}/album/",
            params={"id": album_id},
            timeout=TIMEOUT_MONOCHROME_API,
        )
        resp.raise_for_status()
        return resp.json().get("data")
    except Exception as e:
        print(f"Monochrome album info lookup failed: {e}")
        return None


def process_album_download(job_id: str, album_id: str, album_artist: str, album_title: str, convert_to_flac: bool = True):
    """Download all tracks of a Monochrome/Tidal album.

    Saves tracks to Albums/Artist - Album Title/ with track-number prefixed
    filenames (e.g. '01 - Track Name.flac') and generates an M3U playlist.
    Falls back to Singles if no albums directory is configured.
    """
    source_label = "monochrome"

    try:
        _update_job(job_id, status="downloading")

        # Fetch album metadata and track list
        album_data = _fetch_album_info(album_id)
        if not album_data:
            raise Exception(f"Failed to get album info for Monochrome album {album_id}")

        album_title = album_data.get("title", album_title or "Unknown Album")
        artist_obj = album_data.get("artist") or {}
        album_artist = artist_obj.get("name", album_artist or "Unknown Artist")
        cover_uuid = album_data.get("cover", "")

        # Tracks may be under data.tracks.items or data.items depending on API version
        tracks_obj = album_data.get("tracks") or {}
        tracks = tracks_obj.get("items") or album_data.get("items") or []

        if not tracks:
            raise Exception("No tracks found in album")

        _update_job(job_id, total_tracks=len(tracks), title=album_title, artist=album_artist, uploader=album_artist)

        # Determine the album directory
        albums_dir = get_albums_dir()
        safe_artist = sanitize_filename(album_artist)
        safe_album = sanitize_filename(album_title)

        if albums_dir:
            album_dir = albums_dir / f"{safe_artist} - {safe_album}"
        else:
            # No albums dir configured — nest under Singles/Artist/Album
            album_dir = get_singles_dir() / safe_artist / safe_album

        album_dir.mkdir(parents=True, exist_ok=True)

        downloaded_files = []
        completed_tracks = 0
        failed_tracks = 0
        skipped_tracks = 0

        for track in tracks:
            track_id = str(track.get("id", ""))
            track_title = track.get("title", "Unknown")
            track_number = track.get("trackNumber") or track.get("volumeNumber")
            track_label = track_title

            if not track_id:
                failed_tracks += 1
                continue

            try:
                # Check for duplicates
                existing_file = check_duplicate(album_artist, track_title)
                if existing_file:
                    skipped_tracks += 1
                    downloaded_files.append(existing_file.name)
                    continue

                # Build filename: 01 - Track Title.flac
                safe_track_title = sanitize_filename(track_title)
                track_num_str = f"{int(track_number):02d}" if track_number else "00"
                filename_stem = f"{track_num_str} - {safe_track_title}"
                output_path = album_dir / f"{filename_stem}.flac"

                # Download the FLAC from Monochrome
                _download_monochrome_direct(track_id, output_path)

                if not output_path.exists():
                    raise Exception("Download completed but FLAC file not found")

                set_file_permissions(output_path)

                # Embed cover art from Tidal CDN
                _embed_monochrome_cover(output_path, cover_uuid)

                # Probe audio quality
                audio_quality, bitrate_kbps = probe_audio_quality(output_path)
                min_bitrate = get_setting_int("min_audio_bitrate", 0)
                if min_bitrate and bitrate_kbps and bitrate_kbps < min_bitrate:
                    output_path.unlink(missing_ok=True)
                    raise Exception(f"Audio quality too low ({bitrate_kbps}kbps, minimum is {min_bitrate}kbps)")

                # Apply metadata — Tidal data is authoritative; only use MusicBrainz for the year
                year = None
                if get_setting_bool("enable_musicbrainz", True):
                    mb_metadata = lookup_metadata(album_artist, track_title, output_path)
                    if mb_metadata:
                        year = mb_metadata.get("year")
                apply_metadata_to_file(output_path, album_artist, track_title, album_title, year, tracknumber=track_number)

                # Fetch and save lyrics
                lyrics = fetch_lyrics(album_artist, track_title)
                if lyrics:
                    save_lyrics_file(output_path, lyrics)

                downloaded_files.append(output_path.name)
                completed_tracks += 1

            except Exception as track_error:
                print(f"Album track failed: {track_label} - {track_error}")
                failed_tracks += 1
            finally:
                _update_job(
                    job_id,
                    completed_tracks=completed_tracks,
                    failed_tracks=failed_tracks,
                    skipped_tracks=skipped_tracks,
                )

        # Generate M3U playlist for the album
        if downloaded_files:
            m3u_path = album_dir / f"{safe_album}.m3u"
            with open(m3u_path, 'w', encoding='utf-8') as f:
                f.write("#EXTM3U\n")
                for filename in downloaded_files:
                    f.write(f"{filename}\n")
            set_file_permissions(m3u_path)
            _update_job(job_id, m3u_path=str(album_dir.relative_to(MUSIC_DIR) / m3u_path.name))

        # Trigger library rescans if configured
        trigger_navidrome_scan()
        trigger_jellyfin_scan()

        # Final status
        final_status = "completed"
        error_message = None
        if failed_tracks:
            final_status = "completed_with_errors"
            error_message = f"{failed_tracks} track(s) failed"
            if skipped_tracks:
                error_message += f", {skipped_tracks} skipped (duplicates)"

        _update_job(
            job_id,
            status=final_status,
            error=error_message,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )

        print(f"Monochrome album: Downloaded {completed_tracks}/{len(tracks)} tracks for '{album_artist} - {album_title}'")

        send_notification(
            notification_type="playlist",
            title=album_title,
            playlist_name=f"{album_artist} - {album_title}",
            source=source_label,
            status=final_status,
            error=error_message,
            track_count=len(tracks),
            failed_count=failed_tracks,
            skipped_count=skipped_tracks,
        )

    except Exception as e:
        print(f"Monochrome album download failed: {e}")
        _update_job(job_id, status="failed", error=str(e), completed_at=datetime.now(timezone.utc).isoformat())

        send_notification(
            notification_type="error",
            title=album_title,
            artist=album_artist,
            source=source_label,
            status="failed",
            error=str(e),
        )


def process_download(job_id: str, video_id: str, convert_to_flac: bool = True, source_url: str = None,
                     playlist_name: str = None, use_playlists_dir: bool = False):
    """Process a download job.

    source_url overrides the default YouTube URL construction — used for
    SoundCloud and any future yt-dlp-supported source.
    Monochrome tracks bypass yt-dlp entirely and download via the API.
    playlist_name + use_playlists_dir route bulk import tracks into the Playlists folder.
    """
    is_soundcloud = source_url and "soundcloud.com" in source_url
    is_monochrome = source_url and "monochrome.tf" in source_url
    is_url_source = bool(source_url)

    # Monochrome gets its own dedicated download path — no yt-dlp needed
    if is_monochrome:
        _process_monochrome_download(job_id, video_id, convert_to_flac)
        return

    if is_soundcloud:
        source_label = "soundcloud"
    else:
        source_label = "youtube"
    target_url = source_url or f"https://www.youtube.com/watch?v={video_id}"

    try:
        if not is_url_source and not is_valid_youtube_id(video_id):
            raise Exception("Invalid YouTube video ID")

        # Defaults in case extraction fails before artist/title are assigned
        artist = None
        title = video_id
        has_cookies = (not is_url_source) and COOKIES_FILE.exists() and COOKIES_FILE.stat().st_size > 0

        # Update status to downloading
        _update_job(job_id, status="downloading")

        # First, get video info for proper metadata
        base_args = _ytdlp_base_args() if not is_url_source else []
        info_cmd = [
            "yt-dlp",
            *base_args,
            "--dump-json",
            "--no-warnings",
            target_url,
        ]

        info_result = subprocess.run(info_cmd, capture_output=True, text=True, timeout=TIMEOUT_YTDLP_INFO)
        if info_result.returncode != 0:
            if not is_url_source and _is_ytdlp_403(info_result.stderr):
                if has_cookies:
                    _note_cookie_failure()
                hint = "Your cookies may have expired — try re-exporting them in Settings." if has_cookies else "Add browser cookies in Settings to authenticate."
                raise Exception(f"YouTube blocked this request (403). {hint}")
            raise Exception("Failed to get video info")

        info = json.loads(info_result.stdout)

        # Capture source audio format before yt-dlp converts it
        source_format_info = _extract_source_format_from_info(info) if convert_to_flac else None

        # Extract artist and title — SoundCloud uses 'uploader' for artist
        full_title = info.get("title", "Unknown")
        channel = info.get("uploader", info.get("channel", "Unknown")) if is_url_source else info.get("channel", info.get("uploader", "Unknown"))
        artist, title = extract_artist_title(full_title, channel)

        # Update job with extracted info (store raw uploader for blacklist reporting)
        _update_job(job_id, title=title, artist=artist, uploader=channel)

        # Check for duplicates
        existing_file = check_duplicate(artist, title)
        if existing_file:
            _update_job(
                job_id,
                status="completed",
                completed_at=datetime.now(timezone.utc).isoformat(),
                error=f"Already exists: {existing_file.name}"
            )
            _mark_watched_track_downloaded(job_id)
            return

        # Create download directory — either Playlists/Name/ or the standard Singles layout
        playlists_dir = get_playlists_dir() if (use_playlists_dir and playlist_name) else None
        if playlists_dir:
            artist_dir = playlists_dir / sanitize_filename(playlist_name)
            safe_title = _playlist_stem(artist, title, video_id)
        else:
            artist_dir = get_download_dir(artist)
            safe_title = _output_stem(artist, title, video_id)
        artist_dir.mkdir(parents=True, exist_ok=True)

        # Download with best audio quality
        output_template = str(artist_dir / f"{safe_title}.%(ext)s")
        download_cmd = _build_ytdlp_download_cmd(
            video_id, output_template, convert_to_flac,
            source_url=source_url,
            use_cookies=not is_url_source,
        )

        # Retry strategy: back off on suspected bot blocks; if cookies seem to cause 403s,
        # try again without cookies once to distinguish auth problems from general rate limits.
        download_result, download_timed_out = _run_ytdlp_with_retries(
            download_cmd,
            TIMEOUT_YTDLP_DOWNLOAD,
            has_cookies
        )

        if download_timed_out:
            raise Exception("Download timed out (no progress)")

        if not download_result or download_result.returncode != 0:
            stderr = download_result.stderr if download_result else ""

            # Permission denied on temp file rename — clean up and retry once
            if download_result and _is_permission_error(stderr):
                cleaned = _cleanup_temp_files(artist_dir, safe_title)
                if cleaned:
                    print(f"Retrying download after cleaning {cleaned} temp file(s)")
                    download_result, download_timed_out = _run_ytdlp_with_retries(
                        download_cmd, TIMEOUT_YTDLP_DOWNLOAD, has_cookies
                    )
                    if not download_timed_out and download_result and download_result.returncode == 0:
                        stderr = None  # Clear error — retry succeeded

            if stderr:
                error_msg = f"Download failed: {stderr}"
                if not is_url_source and download_result and _is_ytdlp_403(stderr):
                    if has_cookies:
                        error_msg = "YouTube blocked this download (403). Your cookies may have expired — try re-exporting them in Settings."
                    else:
                        error_msg = "YouTube blocked this download (403). Add browser cookies in Settings to authenticate."
                raise Exception(error_msg)

        audio_file = _find_downloaded_audio_or_raise(artist_dir, safe_title)

        # Set permissions for NAS/SMB compatibility
        set_file_permissions(audio_file)

        # Probe audio quality (with source info so FLAC-from-lossy is reported honestly)
        audio_quality, bitrate_kbps = probe_audio_quality(audio_file, source_info=source_format_info)
        min_bitrate = get_setting_int("min_audio_bitrate", 0)
        if min_bitrate and bitrate_kbps and bitrate_kbps < min_bitrate:
            audio_file.unlink(missing_ok=True)
            raise Exception(f"Audio quality too low ({bitrate_kbps}kbps, minimum is {min_bitrate}kbps)")

        # Try to enrich metadata with AcoustID fingerprinting, then MusicBrainz
        metadata_source = _default_metadata_source(source_label)
        mb_metadata = lookup_metadata(artist, title, audio_file)
        if mb_metadata:
            metadata_source = mb_metadata.get("metadata_source", metadata_source)
            mb_artist = mb_metadata.get("artist", artist)
            mb_title = mb_metadata.get("title", title)
            apply_metadata_to_file(
                audio_file, mb_artist, mb_title,
                mb_metadata.get("album", ""),
                mb_metadata.get("year")
            )
            # Use the canonical artist/title from MusicBrainz everywhere
            if mb_artist != artist:
                audio_file = _relocate_for_normalised_artist(audio_file, artist, mb_artist)
                artist = mb_artist
            if mb_title != title:
                title = mb_title
            _update_job(job_id, artist=artist, title=title)
        else:
            apply_metadata_to_file(audio_file, artist, title)

        # Fetch and save lyrics
        lyrics = fetch_lyrics(artist, title)
        if lyrics:
            save_lyrics_file(audio_file, lyrics)
            print(f"Saved lyrics for {artist} - {title}")
        else:
            print(f"No lyrics found for {artist} - {title}")

        # Trigger library rescans if configured
        trigger_navidrome_scan()
        trigger_jellyfin_scan()

        # Update job status
        _update_job(
            job_id,
            status="completed",
            error=None,
            audio_quality=audio_quality,
            metadata_source=metadata_source,
            completed_at=datetime.now(timezone.utc).isoformat()
        )
        _mark_watched_track_downloaded(job_id)

        # Send notification for single track
        send_notification(
            notification_type="single",
            title=title,
            artist=artist,
            source=source_label,
            status="completed"
        )

    except Exception as e:
        _update_job(job_id, status="failed", error=str(e), completed_at=datetime.now(timezone.utc).isoformat())

        # Send notification for failure
        send_notification(
            notification_type="error",
            title=title,
            artist=artist,
            source=source_label,
            status="failed",
            error=str(e)
        )
