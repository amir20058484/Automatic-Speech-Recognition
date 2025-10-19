import os

from .core import split_audio_20s, transcribe_segments
from .sentiment import analyze_sentiment
from .text_utils import preprocess_text
from .utils import action_logger, log_action, prepare_audio, time_logger


def asr_with_sentiment(audio_path: str):
        """
        Full pipeline:
        1️⃣ Prepare audio
        2️⃣ Split into segments and transcribe
        3️⃣ Analyze sentiment
        """
        try:
            save_path = prepare_audio(audio_path)

            # Split and transcribe
            log_action(f"Starting ASR pipeline for: {save_path}")
            segments = split_audio_20s(save_path)
            if not segments:
                log_action("No segments generated", level="error")
                return "Audio splitting failed!", "❌ No sentiment analysis."

            text = transcribe_segments(segments, model_size="medium", language="fa")
            log_action("ASR process completed successfully")

            if not text:
                return "No speech detected in the audio.", "❌ No sentiment analysis."

            # Sentiment analysis
            sentiment_result = analyze_sentiment(text)
            sentiment_text = f"{sentiment_result['label']} ({sentiment_result['score']})"
            log_action(f"Sentiment analysis completed: {sentiment_text}")

            return text, sentiment_text

        except Exception as e:
            log_action(f"Pipeline failed: {e}", level="error")
            return f"❌ Error during transcription: {e}", "❌ No sentiment analysis."

@action_logger
@time_logger
def asr(input_path, model_size="base", language="fa"):
    if not input_path or not os.path.exists(input_path):
        log_action(f"Invalid or missing audio file: {input_path}", level="error")
        return "Error: Audio file not found."

    log_action(f"🚀 Starting ASR pipeline for file: {input_path}")


    segment_paths = split_audio_20s(input_path)
    if not segment_paths:
        log_action("No segments created. Aborting transcription.", level="warning")
        return "Error: Could not process audio file."


    raw_text = transcribe_segments(segment_paths, model_size=model_size, language=language)

    processed_text = preprocess_text(raw_text)

    log_action(f"✅ ASR + Preprocessing complete. Final text length: {len(processed_text)} chars.")
    return processed_text
