import base64
import io
import os

from mistralai.client import Mistral
from pypdf import PdfReader

MAX_TTS_CHARS = 950
OCR_MODEL = "mistral-ocr-latest"
TRANSLATE_MODEL = "mistral-small-latest"
TTS_MODEL = "voxtral-tts"
TTS_VOICE = "simone-english"


def _client() -> Mistral:
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is not set")
    return Mistral(api_key=api_key)


def extract_text(filename: str, data: bytes) -> str:
    """Extract text from an uploaded file using Mistral OCR (or plain read for txt)."""
    lower = filename.lower()

    if lower.endswith((".txt", ".md")):
        return data.decode("utf-8", errors="replace")

    client = _client()

    if lower.endswith(".pdf") or data.startswith(b"%PDF"):
        # pypdf cannot rasterize pages, so use Mistral OCR's document URL mode
        # via base64 upload of the whole PDF.
        b64 = base64.b64encode(data).decode()
        resp = client.ocr.process(
            model=OCR_MODEL,
            document={
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{b64}",
            },
        )
        return "\n\n".join(p.markdown for p in resp.pages)

    # Treat as a single image (png/jpg/webp)
    b64 = base64.b64encode(data).decode()
    mime = "image/png" if lower.endswith(".png") else "image/jpeg"
    resp = client.ocr.process(
        model=OCR_MODEL,
        document={"type": "image_url", "image_url": f"data:{mime};base64,{b64}"},
    )
    return "\n\n".join(p.markdown for p in resp.pages)


def translate_to_german(text: str) -> str:
    """Translate French text to German with a Mistral chat model."""
    client = _client()
    resp = client.chat.complete(
        model=TRANSLATE_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional translator. Translate the following French "
                    "document into natural, fluent German. Output only the German "
                    "translation, no commentary."
                ),
            },
            {"role": "user", "content": text},
        ],
    )
    return resp.choices[0].message.content


def synthesize_speech(text: str) -> bytes:
    """Generate German audio with Voxtral TTS, chunking long text."""
    client = _client()
    audio = b""
    for chunk in _chunk(text):
        resp = client.audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=chunk,
        )
        audio += resp.audio
    return audio


def _chunk(text: str, size: int = MAX_TTS_CHARS) -> list[str]:
    text = text.strip()
    if len(text) <= size:
        return [text]
    parts = []
    current = ""
    for sentence in text.replace("\n", " ").split(". "):
        candidate = f"{current} {sentence}".strip() + "."
        if len(candidate) > size and current:
            parts.append(current)
            current = sentence + "."
        else:
            current = candidate
    if current.strip():
        parts.append(current)
    return parts
