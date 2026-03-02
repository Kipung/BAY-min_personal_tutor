from typing import Any


def _read(obj: Any, key: str, default: Any = None) -> Any:
    """
    Safely read a key from a dict or attribute from an object, returning default if not found.
    """
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def server_content(response: Any) -> Any:
    return _read(response, "server_content")


def is_interrupted(response: Any) -> bool:
    return bool(_read(server_content(response), "interrupted", False))


def iter_model_parts(response: Any):
    """
    Yield each part of the model's response, which may contain text and/or audio.
    """
    sc = server_content(response)
    model_turn = _read(sc, "model_turn")
    if model_turn is None:
        return
    for part in _read(model_turn, "parts", []) or []:
        yield part


def extract_audio_chunks(response: Any) -> list[bytes]:
    """
    Extract audio chunks (as bytes) from the model's response parts.
    """
    chunks: list[bytes] = []
    for part in iter_model_parts(response):
        inline_data = _read(part, "inline_data")
        data = _read(inline_data, "data")
        if isinstance(data, (bytes, bytearray)):
            chunks.append(bytes(data))
    return chunks

def extract_input_transcript(response: Any) -> str | None:
    """
    Extract the user's input transcript from the response, if available.
    """
    tx = _read(server_content(response), "input_transcription")
    text = _read(tx, "text")
    if isinstance(text, str) and text.strip():
        return text
    if isinstance(tx, str) and tx.strip():
        return tx
    return None


def extract_output_transcript(response: Any) -> str | None:
    """
    Extract the assistant's spoken transcript from the response, if available.
    """
    tx = _read(server_content(response), "output_transcription")
    text = _read(tx, "text")
    if isinstance(text, str) and text.strip():
        return text
    if isinstance(tx, str) and tx.strip():
        return tx
    return None
