import math
import os

import whisper
from pydub import AudioSegment

from .utils import action_logger, log_action, time_logger


@action_logger
@time_logger
def split_audio_20s(input_path):
    """
    Split an audio file into 20-second chunks and save them in a 'segments' folder.
    Returns a list of paths to the generated segment files.
    """
    if not os.path.exists(input_path):
        log_action(f"Audio file not found: {input_path}", level="error")
        return []

    sound = AudioSegment.from_file(input_path)

    segment_length = 20_000
    total_length = len(sound)
    num_segments = math.ceil(total_length / segment_length)

    log_action(f"Splitting '{input_path}' ({total_length/1000:.1f}s) into {num_segments} x 20s segments...")

    output_dir = os.path.join(os.path.dirname(input_path), "segments")
    os.makedirs(output_dir, exist_ok=True)

    segment_paths = []

    for i in range(num_segments):
        start = i * segment_length
        end = min((i + 1) * segment_length, total_length)
        chunk = sound[start:end]

        out_file = os.path.join(output_dir, f"segment_{i+1}.wav")
        chunk.export(out_file, format="wav")
        segment_paths.append(out_file)

        log_action(f"Created segment {i+1}: {out_file} ({(end-start)/1000:.1f}s)")

    log_action(f"✅ Successfully created {len(segment_paths)} segments of 20s each.")
    return segment_paths

@action_logger
@time_logger
def transcribe_segments(segment_paths, model_size="medium", language="fa"):
    """
    Transcribe a list of audio segment files using Whisper and return full text.
    """
    if not segment_paths:
        log_action("No audio segments provided for transcription.", level="warning")
        return ""

    log_action(f"🔄 Loading Whisper model: {model_size}")
    model = whisper.load_model(model_size)

    all_text = []

    for i, seg_path in enumerate(segment_paths, 1):
        log_action(f"🎧 Transcribing segment {i}/{len(segment_paths)}: {seg_path}")
        result = model.transcribe(seg_path, language=language, fp16=False)
        segment_text = result["text"].strip()
        all_text.append(segment_text)
        log_action(f"📝 Segment {i} result: {segment_text[:50]}...")

    full_text = " ".join(all_text).strip()

    log_action(f"✅ Transcription complete. Total length: {len(full_text)} characters.")
    return full_text










