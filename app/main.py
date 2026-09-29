import base64
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.templating import Jinja2Templates

from . import mistral_flow

app = FastAPI(title="Voxtral Demo")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/process")
async def process(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    try:
        french = mistral_flow.extract_text(file.filename, data)
        if not french.strip():
            raise HTTPException(400, "No text could be extracted from the document")
        german = mistral_flow.translate_to_german(french)
        audio = mistral_flow.synthesize_speech(german)
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(400, str(e))

    return {
        "french": french,
        "german": german,
        "audio": base64.b64encode(audio).decode(),
    }
