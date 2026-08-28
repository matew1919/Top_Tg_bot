"""Safe local preparation of audio/video media for AI analysis.

The module deliberately does not know anything about Telegram or an AI provider.
It only validates supported public video URLs, downloads bounded files and uses
the bundled FFmpeg executable to prepare an audio track and representative
frames. Network and subprocess work is kept off the event loop.
"""

from __future__ import annotations

import asyncio
import html
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


URL_PATTERN = re.compile(r"https?://[^\s<>]+", re.IGNORECASE)


class MediaProcessingError(RuntimeError):
    """An expected, user-facing failure while preparing media."""


@dataclass(frozen=True)
class DownloadedVideo:
    path: Path
    title: str
    duration: float | None
    subtitles: str | None = None


def _normalized_host(url: str) -> str | None:
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError:
        return None
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return None
    if parsed.username or parsed.password or port not in {None, 80, 443}:
        return None
    return parsed.hostname.lower().rstrip(".")


def is_supported_video_url(url: str) -> bool:
    """Only allow known public video hosts; this also prevents generic SSRF."""

    host = _normalized_host(url)
    if not host:
        return False
    parsed = urlsplit(url)
    path = parsed.path.rstrip("/")
    if host == "youtu.be" or host.endswith(".youtu.be"):
        return bool(path and path.count("/") == 1)
    if not any(
        host == allowed or host.endswith("." + allowed)
        for allowed in {"youtube.com", "youtube-nocookie.com"}
    ):
        return False
    if path == "/watch":
        return bool(parse_qs(parsed.query).get("v", [""])[0])
    return bool(re.fullmatch(r"/(?:shorts|embed|live)/[^/]+", path))


def extract_supported_video_urls(text: str, limit: int = 1) -> list[str]:
    """Return distinct supported links in their original order."""

    found: list[str] = []
    for match in URL_PATTERN.finditer(text or ""):
        url = match.group(0).rstrip(".,!?;:)]}'\"")
        if url not in found and is_supported_video_url(url):
            found.append(url)
            if len(found) >= max(1, limit):
                break
    return found


def _plain_subtitles(path: Path, max_chars: int = 50_000) -> str | None:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")[: max_chars * 3]
    except OSError:
        return None
    lines: list[str] = []
    previous = ""
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if (
            not line
            or line == "WEBVTT"
            or line.isdigit()
            or "-->" in line
            or line.startswith(("Kind:", "Language:", "NOTE "))
        ):
            continue
        line = html.unescape(re.sub(r"<[^>]+>", "", line)).strip()
        if line and line != previous:
            lines.append(line)
            previous = line
    text = "\n".join(lines).strip()
    return text[:max_chars].rstrip() or None


def _ffmpeg_executable() -> str:
    try:
        import imageio_ffmpeg
    except ImportError as error:  # pragma: no cover - depends on deployment
        raise MediaProcessingError("FFmpeg-компонент не установлен.") from error
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as error:  # pragma: no cover - platform-specific
        raise MediaProcessingError("Не удалось запустить FFmpeg.") from error


async def _run_ffmpeg(*arguments: str, timeout: float) -> None:
    executable = await asyncio.to_thread(_ffmpeg_executable)
    process = await asyncio.create_subprocess_exec(
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-protocol_whitelist",
        "file,pipe",
        *arguments,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.CancelledError:
        process.kill()
        await process.communicate()
        raise
    except TimeoutError as error:
        process.kill()
        await process.communicate()
        raise MediaProcessingError("Обработка медиа заняла слишком много времени.") from error
    if process.returncode:
        detail = stderr.decode("utf-8", errors="replace").strip()[-500:]
        raise MediaProcessingError("Не удалось прочитать медиафайл.") from RuntimeError(detail)


async def probe_media_duration(source: Path, timeout: float) -> float | None:
    """Read container duration without decoding the complete media file."""

    executable = await asyncio.to_thread(_ffmpeg_executable)
    process = await asyncio.create_subprocess_exec(
        executable,
        "-hide_banner",
        "-nostdin",
        "-protocol_whitelist",
        "file,pipe",
        "-i",
        str(source),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.CancelledError:
        process.kill()
        await process.communicate()
        raise
    except TimeoutError as error:
        process.kill()
        await process.communicate()
        raise MediaProcessingError("Не удалось определить длительность медиа.") from error
    match = re.search(
        rb"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)",
        stderr,
    )
    if not match:
        return None
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


async def extract_audio_track(source: Path, destination: Path, timeout: float) -> Path:
    """Convert any supported media container to compact mono MP3 for STT."""

    await _run_ffmpeg(
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-b:a",
        "64k",
        str(destination),
        timeout=timeout,
    )
    if not destination.exists() or destination.stat().st_size == 0:
        raise MediaProcessingError("В файле не нашлась аудиодорожка.")
    return destination


async def normalize_image(source: Path, destination: Path, timeout: float) -> Path:
    """Strip metadata and bound image dimensions/encoding before an API upload."""

    await _run_ffmpeg(
        "-y",
        "-i",
        str(source),
        "-frames:v",
        "1",
        "-vf",
        "scale=w='min(1280,iw)':h='min(1280,ih)':force_original_aspect_ratio=decrease",
        "-q:v",
        "3",
        str(destination),
        timeout=timeout,
    )
    if not destination.exists() or destination.stat().st_size == 0:
        raise MediaProcessingError("Не удалось подготовить изображение.")
    return destination


async def extract_video_frames(
    source: Path,
    output_directory: Path,
    *,
    duration: float | None,
    max_frames: int,
    timeout: float,
) -> list[Path]:
    """Sample a small, approximately uniform set of frames from a video."""

    frame_count = max(1, max_frames)
    interval = max(0.5, float(duration) / frame_count) if duration else 4.0
    output_pattern = output_directory / "frame_%02d.jpg"
    await _run_ffmpeg(
        "-y",
        "-i",
        str(source),
        "-vf",
        f"fps=1/{interval:.3f},"
        "scale=w='min(1280,iw)':h='min(1280,ih)':force_original_aspect_ratio=decrease",
        "-frames:v",
        str(frame_count),
        "-q:v",
        "3",
        str(output_pattern),
        timeout=timeout,
    )
    return sorted(output_directory.glob("frame_*.jpg"))[:frame_count]


def _download_video_sync(
    url: str,
    output_directory: Path,
    *,
    max_bytes: int,
    max_duration_seconds: int,
    proxy: str | None,
    timeout: float,
    cancel_event: threading.Event,
) -> DownloadedVideo:
    if not is_supported_video_url(url):
        raise MediaProcessingError("Эта видеоссылка не поддерживается.")
    try:
        import yt_dlp
    except ImportError as error:  # pragma: no cover - depends on deployment
        raise MediaProcessingError("Компонент загрузки видеоссылок не установлен.") from error

    deadline = time.monotonic() + timeout

    def enforce_deadline(_status: dict[str, object]) -> None:
        if cancel_event.is_set():
            raise MediaProcessingError("Загрузка видео отменена.")
        if time.monotonic() > deadline:
            raise MediaProcessingError("Загрузка видео заняла слишком много времени.")

    options: dict[str, object] = {
        "format": (
            "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/"
            "bestvideo[height<=720]+bestaudio/best[height<=720]/best"
        ),
        "merge_output_format": "mp4",
        "ffmpeg_location": _ffmpeg_executable(),
        "outtmpl": str(output_directory / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "playlist_items": "1",
        "max_filesize": max_bytes,
        "socket_timeout": min(25.0, max(5.0, timeout / 4)),
        "retries": 2,
        "fragment_retries": 2,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "ignoreconfig": True,
        "cachedir": False,
        "restrictfilenames": True,
        "overwrites": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["ru", "ru-orig", "en", "en-orig"],
        "subtitlesformat": "vtt/best",
        "progress_hooks": [enforce_deadline],
    }
    if proxy:
        options["proxy"] = proxy

    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(url, download=False)
            enforce_deadline({})
            if not isinstance(info, dict):
                raise MediaProcessingError("Сервис не вернул сведения о видео.")
            if info.get("_type") == "playlist":
                entries = [item for item in info.get("entries") or [] if isinstance(item, dict)]
                if not entries:
                    raise MediaProcessingError("В ссылке не найдено видео.")
                info = entries[0]
            duration_value = info.get("duration")
            duration = float(duration_value) if isinstance(duration_value, (int, float)) else None
            if duration and duration > max_duration_seconds:
                raise MediaProcessingError(
                    f"Видео длиннее допустимых {max_duration_seconds // 60} мин."
                )
            if info.get("is_live") or info.get("live_status") == "is_live":
                raise MediaProcessingError("Прямые трансляции пока не анализируются.")
            info = downloader.process_ie_result(info, download=True)
            enforce_deadline({})
    except MediaProcessingError:
        raise
    except Exception as error:
        if time.monotonic() > deadline:
            raise MediaProcessingError("Загрузка видео заняла слишком много времени.") from error
        raise MediaProcessingError("Не удалось скачать видео по ссылке.") from error

    def subtitle_priority(path: Path) -> tuple[int, str]:
        name = path.name.lower()
        if re.search(r"\.ru(?:-orig)?\.(?:vtt|srt)$", name):
            return 0, name
        if re.search(r"\.en(?:-orig)?\.(?:vtt|srt)$", name):
            return 1, name
        return 2, name

    subtitle_paths = sorted(
        (
            path
            for path in output_directory.iterdir()
            if path.suffix.lower() in {".vtt", ".srt"}
        ),
        key=subtitle_priority,
    )
    subtitles = next(
        (text for text in (_plain_subtitles(path) for path in subtitle_paths) if text),
        None,
    )
    candidates = [
        path
        for path in output_directory.iterdir()
        if path.is_file()
        and path.suffix.lower()
        not in {".part", ".ytdl", ".json", ".vtt", ".srt", ".webp", ".jpg", ".png"}
    ]
    if not candidates:
        raise MediaProcessingError("Видео недоступно или превышает лимит размера.")
    path = max(candidates, key=lambda item: item.stat().st_size)
    if path.stat().st_size > max_bytes:
        raise MediaProcessingError("Видео превышает допустимый размер.")
    title = str(info.get("title") or "Видео по ссылке").strip()[:300]
    duration_value = info.get("duration")
    duration = float(duration_value) if isinstance(duration_value, (int, float)) else None
    return DownloadedVideo(path=path, title=title, duration=duration, subtitles=subtitles)


async def download_video(
    url: str,
    output_directory: Path,
    *,
    max_bytes: int,
    max_duration_seconds: int,
    proxy: str | None,
    timeout: float,
) -> DownloadedVideo:
    """Download one validated video with an overall wall-clock timeout."""

    cancel_event = threading.Event()
    worker = asyncio.create_task(
        asyncio.to_thread(
            _download_video_sync,
            url,
            output_directory,
            max_bytes=max_bytes,
            max_duration_seconds=max_duration_seconds,
            proxy=proxy,
            timeout=timeout,
            cancel_event=cancel_event,
        )
    )
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        cancel_event.set()
        try:
            await worker
        except Exception:
            pass
        raise
