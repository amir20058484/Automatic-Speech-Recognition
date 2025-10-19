from shekar import Normalizer

from .utils import action_logger, log_action, time_logger


@action_logger
@time_logger
def preprocess_text(text: str) -> str:
    """
    Clean and normalize Persian text using Shekar Normalizer.
    """
    if not text or text.strip() == "":
        log_action("Empty text provided for preprocessing", level="warning")
        return ""

    normalizer = Normalizer()
    cleaned_text = normalizer(text)

    log_action(f"✅ Text normalized. Length: {len(cleaned_text)} chars. Preview: {cleaned_text[:50]!r}")
    return cleaned_text
