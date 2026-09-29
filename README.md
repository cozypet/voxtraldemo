# Voxtral Demo — Read French documents aloud in German

A demo showing Mistral models working together:

1. **Upload** a French document (PDF, image, or plain text)
2. **OCR**: Mistral OCR extracts the text
3. **Translate**: a Mistral chat model translates French → German
4. **Speak**: [Voxtral TTS](https://docs.mistral.ai/api/#operation/audioSpeech) reads the German text aloud

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export MISTRAL_API_KEY=...
```

## Run

```bash
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000, upload a French document, and listen to the German audio.

## Notes

- PDFs are converted page-by-page to images and sent to Mistral OCR.
- Text extracted from images/PDFs goes through `mistral-small` for translation.
- German audio is generated with `voxtral-tts` and playable in the browser.
