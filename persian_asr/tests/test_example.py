import os

import pytest
from pydub import AudioSegment

from persian_asr.core import split_audio_20s, transcribe_segments
from persian_asr.pipeline import asr
from persian_asr.sentiment import analyze_sentiment


@pytest.fixture
def sample_audio(tmp_path):
    """
    Creates a small dummy WAV file (1-second silent audio) for testing.
    """
    dummy_audio = AudioSegment.silent(duration=1000)
    test_file = tmp_path / "test_audio.wav"
    dummy_audio.export(test_file, format="wav")
    return str(test_file)


@pytest.fixture
def dummy_segments(tmp_path):
    """
    Creates fake segment files to test transcribe_segments().
    """
    seg1 = tmp_path / "segment_1.wav"
    seg2 = tmp_path / "segment_2.wav"
    seg1.write_text("fake audio content")
    seg2.write_text("fake audio content")
    return [str(seg1), str(seg2)]



def test_split_audio_creates_segments(sample_audio):
    """
    Check if split_audio_20s() creates at least one segment file.
    """
    segments = split_audio_20s(sample_audio)
    assert isinstance(segments, list)
    assert len(segments) > 0
    for seg in segments:
        assert os.path.exists(seg)


def test_split_audio_missing_file():
    """
    Ensure split_audio_20s() handles missing file gracefully.
    """
    segments = split_audio_20s("non_existing_file.wav")
    assert segments == []


def test_transcribe_segments_handles_empty_list():
    """
    Ensure transcribe_segments() returns empty string if no segments.
    """
    result = transcribe_segments([])
    assert result == ""


def test_transcribe_segments_calls_whisper(monkeypatch, dummy_segments):
    """
    Mock Whisper model to test without real audio processing.
    """

    class DummyModel:
        def transcribe(self, path, language="fa", * ,fp16=False):
            _ = fp16
            _ = language
            _ = path
            return {"text": f"transcribed from {os.path.basename(path)}"}

    def fake_load_model(_name):
        return DummyModel()

    monkeypatch.setattr("persian_asr.core.whisper.load_model", fake_load_model)

    result = transcribe_segments(dummy_segments, model_size="medium", language="fa")

    assert "transcribed from segment_1.wav" in result
    assert "transcribed from segment_2.wav" in result


def test_asr_pipeline(monkeypatch, tmp_path):
    """
    Test the full ASR pipeline using mocked split and transcribe functions.
    Ensures the system goes through all steps correctly.
    """
    fake_audio = tmp_path / "fake_audio.wav"
    fake_audio.write_text("fake audio data")

    def fake_split_audio_20s(path):
        assert os.path.exists(path)
        return [str(tmp_path / "segment_1.wav"), str(tmp_path / "segment_2.wav")]

    def fake_transcribe_segments(segment_paths, model_size="medium", language="fa"):
        _ = language
        _ = model_size
        _ = segment_paths
        return "segment one text segment two text"

    monkeypatch.setattr("persian_asr.pipeline.split_audio_20s", fake_split_audio_20s)
    monkeypatch.setattr("persian_asr.pipeline.transcribe_segments", fake_transcribe_segments)

    result = asr(str(fake_audio), model_size="medium", language="fa")

    assert isinstance(result, str)
    assert "segment" in result
    assert "text" in result



def test_analyze_sentiment_empty_text():
    """
    Check that analyze_sentiment() returns 'unknown' for empty input.
    """

    result = analyze_sentiment("")
    assert result["label"] == "unknown"
    assert result["score"] == 0.0


def test_analyze_sentiment_positive(monkeypatch):
    """
    Mock SentimentClassifier to simulate a positive sentiment.
    """

    class DummyModel:
        def __call__(self, _text):
            return ("positive", 0.9)

    monkeypatch.setattr("persian_asr.sentiment.SentimentClassifier", lambda: DummyModel())

    result = analyze_sentiment("این فیلم خیلی عالی بود!")
    assert result["label"] == "positive"
    assert 0.8 < result["score"] <= 1.0


def test_analyze_sentiment_negative(monkeypatch):
    """
    Mock SentimentClassifier to simulate a negative sentiment.
    """

    class DummyModel:
        def __call__(self, _text):
            return ("negative", 0.75)

    monkeypatch.setattr("persian_asr.sentiment.SentimentClassifier", lambda **_: DummyModel())

    result = analyze_sentiment("خیلی بد و افتضاح بود")
    assert result["label"] == "negative"
    assert -1.0 <= result["score"] < -0.7



