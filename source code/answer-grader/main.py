"""
main.py — AI Answer Sheet Evaluator v3.1

Endpoints:
  GET  /                      health check
  POST /ocr/extract           OCR preview with quality score
  POST /grade/single          student image + TYPED model answer
  POST /grade/single/image    student image + model answer IMAGE (validates OCR quality first)
  POST /grade/multi           student image + JSON model answers
  POST /grade/multi/image     student image + model answer IMAGES per question

Key improvements in v3.1:
  - Model answer image OCR now validated before grading (prevents 0% scores from bad images)
  - Clear error message when model answer image has dark ruled lines / unreadable
  - Student OCR quality included in every response
  - Auto page splitting for combined multi-page screenshots
  - Adaptive upscaling for narrow images
"""

import os, shutil, uuid, json, logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from grader import (
    grade_answer_sheet, grade_single_answer,
    extract_text_from_image, extract_model_answer_from_image,
    get_model, extract_keywords,
)
from utils import clean_text, ocr_quality_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading NLP model...")
    get_model()
    log.info("Server ready.")
    yield

app = FastAPI(
    title="AI Answer Sheet Evaluator",
    description="""
Grade handwritten answer sheets automatically.

## Endpoints

| Endpoint | Student Answer | Model Answer |
|---|---|---|
| `/grade/single` | Image | Typed text ← **Most reliable** |
| `/grade/single/image` | Image | Handwritten image |
| `/grade/multi` | Image | Typed JSON |
| `/grade/multi/image` | Image | Multiple images |
| `/ocr/extract` | Image | — (preview only) |

### ⚠️ Model Answer Image Requirements
For image-based model answers, the teacher copy must have:
- **Blue or black ink on plain white paper** (no dark printed ruled lines)
- Good lighting, flat paper, clear handwriting

If OCR quality is too low the API returns a **422 error** with instructions to type instead.
""",
    version="3.1.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def save_upload(file: UploadFile) -> str:
    ext = os.path.splitext(file.filename or "upload")[-1].lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Use: JPG, PNG, BMP, TIFF, WEBP.")
    path = os.path.join(TEMP_DIR, f"{uuid.uuid4().hex}{ext}")
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return path


def cleanup(*paths: str):
    for p in paths:
        try:
            if p and os.path.exists(p): os.remove(p)
        except Exception: pass


# ── HEALTH ───────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def health():
    return {
        "status": "running", "version": "3.1.0",
        "endpoints": {
            "ocr_preview":       "POST /ocr/extract",
            "grade_typed":       "POST /grade/single        ← recommended",
            "grade_image_model": "POST /grade/single/image",
            "grade_multi_typed": "POST /grade/multi",
            "grade_multi_image": "POST /grade/multi/image",
        }
    }


# ── OCR PREVIEW ──────────────────────────────────────────────────────────────

@app.post("/ocr/extract", tags=["OCR"],
          summary="Preview OCR from any handwritten image")
async def ocr_extract(
    file: UploadFile = File(..., description="Any handwritten image"),
):
    """Run OCR and return extracted text + quality score. Use to check image quality before grading."""
    path = save_upload(file)
    try:
        raw   = extract_text_from_image(path)
        clean = clean_text(raw)
        quality = ocr_quality_score(raw)
        quality_label = (
            "good"       if quality >= 0.55 else
            "acceptable" if quality >= 0.35 else
            "poor"       if quality >= 0.20 else
            "failed"
        )
        return {
            "raw_text":      raw,
            "cleaned_text":  clean,
            "word_count":    len(clean.split()),
            "ocr_quality":   round(quality, 2),
            "quality_label": quality_label,
            "auto_keywords": extract_keywords(clean, top_n=15),
        }
    except Exception as e:
        log.error(f"OCR error: {e}")
        raise HTTPException(500, f"OCR failed: {e}")
    finally:
        cleanup(path)


# ── GRADE SINGLE — TYPED MODEL ANSWER ────────────────────────────────────────

@app.post("/grade/single", tags=["Grade — Typed Model Answer"],
          summary="Student image + typed model answer (most reliable)")
async def grade_single_text(
    file:          UploadFile = File(..., description="Student's handwritten answer sheet image"),
    model_answer:  str        = Form(..., description="Type the correct/expected answer"),
    max_marks:     float      = Form(..., description="Maximum marks"),
    question_text: str        = Form("",  description="(Optional) The question"),
    keywords:      str        = Form("",  description="(Optional) Comma-separated key terms"),
):
    """Grade one handwritten answer against a **typed** model answer. Most accurate mode."""
    if not model_answer.strip():
        raise HTTPException(422, "model_answer cannot be empty.")
    if max_marks <= 0:
        raise HTTPException(422, "max_marks must be positive.")

    path = save_upload(file)
    try:
        raw    = extract_text_from_image(path, model_answer=model_answer)
        quality = ocr_quality_score(raw)
        kw     = [k.strip() for k in keywords.split(",") if k.strip()] or None
        result = grade_single_answer(raw, model_answer, max_marks, kw, ocr_quality=quality)
        result["question_text"]       = question_text
        result["extracted_text"]      = clean_text(raw, model_answer)
        result["model_answer_source"] = "typed_text"
        log.info(f"[single/text] {result['marks_awarded']}/{max_marks} ({result['percentage']}%)")
        return result
    except Exception as e:
        log.error(f"Error: {e}")
        raise HTTPException(500, f"Grading failed: {e}")
    finally:
        cleanup(path)


# ── GRADE SINGLE — IMAGE MODEL ANSWER ────────────────────────────────────────

@app.post("/grade/single/image", tags=["Grade — Image Model Answer"],
          summary="Student image + teacher's handwritten model answer image")
async def grade_single_image(
    student_file:       UploadFile = File(..., description="Student's handwritten answer sheet"),
    model_answer_file:  UploadFile = File(..., description="Teacher's model answer image — MUST be blue/black ink on plain white paper (no dark ruled lines)"),
    max_marks:          float      = Form(..., description="Maximum marks"),
    question_text:      str        = Form("",  description="(Optional) The question"),
    keywords:           str        = Form("",  description="(Optional) Comma-separated key terms"),
):
    """
    Grade against a **handwritten teacher model answer image**.

    ⚠️ **Requirements for model answer image:**
    - Blue or black ink on **plain white paper** (NOT lined/ruled paper with dark printed lines)
    - Well-lit, flat, in focus
    - If the image has dark ruled lines, OCR will fail and a 422 error is returned

    If you get a 422 error, use `/grade/single` and type the model answer instead.
    """
    if max_marks <= 0:
        raise HTTPException(422, "max_marks must be positive.")

    student_path = save_upload(student_file)
    model_path   = save_upload(model_answer_file)

    try:
        # Step 1: OCR and VALIDATE the model answer image
        log.info("OCR-ing model answer image...")
        model_result = extract_model_answer_from_image(model_path)
        model_text   = model_result["text"]
        model_quality = model_result["quality"]
        model_warning = model_result.get("warning")

        log.info(f"Model OCR quality: {model_quality:.0%} ({model_result['quality_label']}) | '{model_text[:60]}...'")

        # Step 2: OCR student answer guided by model vocabulary
        log.info("OCR-ing student answer image...")
        student_raw = extract_text_from_image(student_path, model_answer=model_text)
        student_quality = ocr_quality_score(student_raw)

        # Step 3: Grade
        kw = [k.strip() for k in keywords.split(",") if k.strip()] or None
        result = grade_single_answer(student_raw, model_text, max_marks, kw, ocr_quality=student_quality)
        result["question_text"]           = question_text
        result["extracted_text"]          = clean_text(student_raw, model_text)
        result["model_answer_extracted"]  = model_text
        result["model_ocr_quality"]       = model_quality
        result["model_ocr_quality_label"] = model_result["quality_label"]
        result["model_answer_source"]     = "image_ocr"
        if model_warning:
            result["model_ocr_warning"] = model_warning

        log.info(f"[single/image] {result['marks_awarded']}/{max_marks} ({result['percentage']}%)")
        return result

    except ValueError as e:
        # extract_model_answer_from_image raises ValueError for catastrophic OCR failure
        raise HTTPException(422, str(e))
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error: {e}")
        raise HTTPException(500, f"Grading failed: {e}")
    finally:
        cleanup(student_path, model_path)


# ── GRADE MULTI — TYPED MODEL ANSWERS ────────────────────────────────────────

@app.post("/grade/multi", tags=["Grade — Typed Model Answer"],
          summary="Full student sheet + typed model answers as JSON")
async def grade_multi_text(
    file:      UploadFile = File(..., description="Student's full answer sheet image"),
    questions: str        = Form(..., description=
        'JSON list. Example:\n[{"question_no":1,"question_text":"What is a stub?",'
        '"model_answer":"A stub is a dummy module...","max_marks":5}]'),
):
    """Grade multiple questions against **typed** model answers."""
    try:
        raw_q = json.loads(questions)
        if not isinstance(raw_q, list) or not raw_q:
            raise ValueError("Must be a non-empty list.")
        for i, q in enumerate(raw_q):
            if not q.get("model_answer", "").strip():
                raise ValueError(f"Question {i+1} missing model_answer.")
            if not q.get("max_marks", 0):
                raise ValueError(f"Question {i+1} missing max_marks.")
    except json.JSONDecodeError as e:
        raise HTTPException(422, f"Invalid JSON: {e}")
    except ValueError as e:
        raise HTTPException(422, str(e))

    path = save_upload(file)
    try:
        raw_q.sort(key=lambda x: x.get("question_no", 0))
        result = grade_answer_sheet(path, raw_q)
        log.info(f"[multi/text] {result['total_marks_awarded']}/{result['total_max_marks']} ({result['overall_percentage']}%)")
        return result
    except Exception as e:
        log.error(f"Error: {e}")
        raise HTTPException(500, f"Grading failed: {e}")
    finally:
        cleanup(path)


# ── GRADE MULTI — IMAGE MODEL ANSWERS ────────────────────────────────────────

@app.post("/grade/multi/image", tags=["Grade — Image Model Answer"],
          summary="Full student sheet + teacher model answer images (one per question)")
async def grade_multi_image(
    student_file:       UploadFile       = File(..., description="Student's full answer sheet"),
    model_answer_files: list[UploadFile] = File(..., description="Teacher model answer images — one per question, in order"),
    max_marks_per_q:    str              = Form(..., description="Comma-separated marks e.g. '5,5,10'"),
    question_texts:     str              = Form("",  description="(Optional) Comma-separated question texts"),
):
    """
    Grade a full answer sheet against **teacher model answer images**.

    Upload one model answer image per question. Each image must be blue/black ink
    on plain white paper. If any image fails OCR quality check, a 422 is returned.
    """
    try:
        marks_list = [float(m.strip()) for m in max_marks_per_q.split(",") if m.strip()]
    except ValueError:
        raise HTTPException(422, "max_marks_per_q must be comma-separated numbers e.g. '5,5,10'")

    if len(marks_list) != len(model_answer_files):
        raise HTTPException(422,
            f"Number of marks ({len(marks_list)}) must equal number of model images ({len(model_answer_files)}).")

    q_texts = [t.strip() for t in question_texts.split(",")] if question_texts.strip() else []

    student_path = save_upload(student_file)
    model_paths  = [save_upload(f) for f in model_answer_files]

    try:
        # OCR + validate all model answer images
        log.info(f"OCR-ing {len(model_paths)} model answer image(s)...")
        model_results = []
        warnings = []
        for i, mp in enumerate(model_paths):
            try:
                mr = extract_model_answer_from_image(mp)
            except ValueError as e:
                raise HTTPException(422, f"Model answer image {i+1}: {e}")
            model_results.append(mr)
            if mr.get("warning"):
                warnings.append(f"Q{i+1}: {mr['warning']}")
            log.info(f"  Q{i+1} model quality: {mr['quality']:.0%} | '{mr['text'][:60]}...'")

        questions = [
            {
                "question_no":   i + 1,
                "question_text": q_texts[i] if i < len(q_texts) else f"Question {i+1}",
                "model_answer":  model_results[i]["text"],
                "max_marks":     marks_list[i],
                "keywords":      [],
            }
            for i in range(len(model_results))
        ]

        log.info("Grading student answer sheet...")
        result = grade_answer_sheet(student_path, questions)

        for i, qr in enumerate(result.get("question_results", [])):
            qr["model_answer_extracted"]  = model_results[i]["text"]
            qr["model_ocr_quality"]       = model_results[i]["quality"]
            qr["model_ocr_quality_label"] = model_results[i]["quality_label"]
            if model_results[i].get("warning"):
                qr["model_ocr_warning"] = model_results[i]["warning"]

        result["model_answer_source"] = "image_ocr"
        if warnings:
            result["model_ocr_warnings"] = warnings

        log.info(f"[multi/image] {result['total_marks_awarded']}/{result['total_max_marks']} ({result['overall_percentage']}%)")
        return result

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error: {e}")
        raise HTTPException(500, f"Grading failed: {e}")
    finally:
        cleanup(student_path, *model_paths)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)