import functools
import logging
import os
import shutil
import time

from pydub import AudioSegment


def setup_logger():
    """
    Initialize the logging system for the Persian ASR project.
    All logs are written to a single file 'asr.log' inside resources/asr_log.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(base_dir, "resources", "asr_log")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "asr.log")

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="a",  # Append mode, don't overwrite the log file
    )

    logging.info("Logger initialized successfully.")


def log_action(action: str, level: str = "info"):
    """
    Log a general action or event.
    Example:
        log_action("Audio file uploaded")
        log_action("Network error", level="error")
    """
    level = level.lower()
    if level == "debug":
        logging.debug(action)
    elif level == "warning":
        logging.warning(action)
    elif level == "error":
        logging.error(action)
    else:
        logging.info(action)


def log_time(action: str, start_time: float, end_time: float):
    """
    Log the time taken for a specific operation.
    Example:
        start = time.time()
        ... some code ...
        end = time.time()
        log_time("ASR processing", start, end)
    """
    duration = end_time - start_time
    logging.info(f"{action} took {duration:.3f} seconds.")


def time_logger(func):
    """
    Decorator that measures and logs the execution time of a function.
    Example:
        @time_logger
        def asr(audio):
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        logging.info(f"Started timing for '{func.__name__}'")
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start
            logging.info(f"Finished '{func.__name__}' in {duration:.3f} seconds.")
    return wrapper


def action_logger(func):
    """
    Decorator that logs when a function starts and finishes (regardless of timing).
    Example:
        @action_logger
        def preprocess(audio):
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"Action started: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logging.info(f"Action completed: {func.__name__}")
            return result
        except Exception as e:
            logging.error(f"Error in action '{func.__name__}': {e}")
            raise
    return wrapper

def prepare_audio(audio_path: str) -> str:
        """
        Copy the uploaded audio to a short fixed path and convert MP3 to WAV if needed.
        Returns the path to the prepared audio file.
        """
        save_dir = os.path.join("resources")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "temp_audio.wav")

        # Wait briefly to ensure the Temp file is ready
        time.sleep(0.1)

        ext = os.path.splitext(audio_path)[1].lower()
        temp_copy_path = os.path.join(save_dir, "temp_input" + ext)
        shutil.copy(audio_path, temp_copy_path)

        if ext == ".mp3":
            sound = AudioSegment.from_mp3(temp_copy_path)
            sound.export(save_path, format="wav")
        else:
            shutil.copy(temp_copy_path, save_path)

        os.remove(temp_copy_path)
        log_action(f"🎵 Audio prepared successfully at {save_path}")
        return save_path
