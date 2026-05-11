from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.services.file_parser import FileParserError, inspect_upload, parse_upload_comments
from backend.services.predictor import predict_many, predict_text
from backend.services.triage import build_review_item, build_summary


app = FastAPI(title="Comment Safety Monitor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
        "https://comment-safety-monitor.vercel.app",
    ],
    allow_origin_regex=r"https://comment-safety-monitor.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeTextRequest(BaseModel):
    text: str | None = None


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "Comment Safety Monitor"}


@app.get("/")
def root() -> dict[str, str]:
    return health()


@app.post("/api/analyze-text")
def analyze_text(payload: AnalyzeTextRequest) -> dict[str, object]:
    if payload.text is None or not payload.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Please enter a comment before running the check.",
        )

    try:
        prediction = predict_text(payload.text)
        return build_review_item(prediction, original_text=payload.text)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="The comment could not be analyzed. Please try again.",
        ) from exc


@app.post("/api/inspect-file")
async def inspect_file(file: UploadFile = File(...)) -> dict[str, object]:
    content = await file.read()
    try:
        return inspect_upload(file.filename, content)
    except FileParserError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/analyze-file")
async def analyze_file(
    file: UploadFile = File(...),
    text_column: str | None = Form(default=None),
) -> dict[str, object]:
    content = await file.read()
    try:
        _, comments = parse_upload_comments(file.filename, content, text_column)
        predictions = predict_many(comment.text for comment in comments)
        items = [
            build_review_item(
                prediction,
                row_id=comment.row_id,
                original_text=comment.text,
            )
            for comment, prediction in zip(comments, predictions, strict=True)
        ]
    except FileParserError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="The uploaded comments could not be analyzed. Please try again.",
        ) from exc

    items.sort(key=lambda item: float(item.get("risk_score", 0.0)), reverse=True)
    return {"summary": build_summary(items), "items": items}
