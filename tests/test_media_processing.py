import asyncio
import subprocess
import tempfile
import unittest
from pathlib import Path

import imageio_ffmpeg

from media_processing import (
    _plain_subtitles,
    extract_audio_track,
    extract_video_frames,
    normalize_image,
    probe_media_duration,
)


class FfmpegPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_audio_and_frames_are_extracted_from_video(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            source = directory / "source.mp4"
            audio = directory / "audio.mp3"
            frames = directory / "frames"
            frames.mkdir()
            executable = imageio_ffmpeg.get_ffmpeg_exe()
            await asyncio.to_thread(
                subprocess.run,
                [
                    executable,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=blue:s=320x240:d=2",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:duration=2",
                    "-shortest",
                    "-c:v",
                    "libx264",
                    "-c:a",
                    "aac",
                    "-y",
                    str(source),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )

            await extract_audio_track(source, audio, timeout=20)
            duration = await probe_media_duration(source, timeout=20)
            extracted_frames = await extract_video_frames(
                source,
                frames,
                duration=2,
                max_frames=3,
                timeout=20,
            )

            self.assertGreater(audio.stat().st_size, 0)
            self.assertIsNotNone(duration)
            self.assertAlmostEqual(duration, 2.0, delta=0.2)
            self.assertGreaterEqual(len(extracted_frames), 2)
            self.assertLessEqual(len(extracted_frames), 3)
            self.assertTrue(all(path.stat().st_size > 0 for path in extracted_frames))

            normalized = directory / "normalized.jpg"
            await normalize_image(extracted_frames[0], normalized, timeout=20)
            self.assertGreater(normalized.stat().st_size, 0)

            tall = directory / "tall.ppm"
            tall.write_bytes(b"P6\n100 5000\n255\n" + b"\0" * (100 * 5000 * 3))
            bounded = directory / "bounded.jpg"
            await normalize_image(tall, bounded, timeout=20)
            reader = imageio_ffmpeg.read_frames(str(bounded))
            try:
                metadata = next(reader)
            finally:
                reader.close()
            self.assertLessEqual(max(metadata["size"]), 1280)


class SubtitleTests(unittest.TestCase):
    def test_vtt_metadata_and_duplicate_cues_are_removed(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            subtitle = Path(directory_name) / "video.ru.vtt"
            subtitle.write_text(
                "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n<c>Привет</c>\n\n"
                "00:00:01.000 --> 00:00:02.000\nПривет\nмир &amp; все\n",
                encoding="utf-8",
            )

            self.assertEqual(_plain_subtitles(subtitle), "Привет\nмир & все")


if __name__ == "__main__":
    unittest.main()
