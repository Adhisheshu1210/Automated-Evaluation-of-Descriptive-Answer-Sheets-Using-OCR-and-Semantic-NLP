"""
grader.py  —  OCR + grading engine v3.1

Key improvements:
  - extract_text_from_image(): uses adaptive upscaling + page splitting
  - extract_model_answer_from_image(): validates OCR quality, raises clear
    error if image is unsuitable (dark ruled lines, etc.) instead of silently
    grading against garbage text
  - grade_single_answer(): returns ocr_quality field in response
  - All endpoints return warnings when OCR quality is poor
"""

import re
import numpy as np
import pytesseract
import cv2

from utils import (
    preprocess_image,
    correct_ocr_text,
    clean_text,
    split_answers_by_question,
    ocr_quality_score,
    auto_split_pages,
    _final_cleanup,
    apply_static_corrections,
    apply_context_correction,
)


# ─── NLP MODEL ───────────────────────────────────────────────────────────────

_sentence_model = None


def get_model():
    global _sentence_model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer
        print("[NLP] Loading sentence-transformer model...")
        _sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("[NLP] Model loaded.")
    return _sentence_model


# ─── OCR: STUDENT ANSWER ─────────────────────────────────────────────────────

def extract_text_from_image(image_path: str, model_answer: str = "") -> str:
    """
    Extract and correct handwritten text from a student answer sheet.
    
    Automatically:
    - Splits combined multi-page screenshots into pages
    - Applies adaptive upscaling (narrow images get higher scale factors)
    - Runs Tesseract PSM 4 + PSM 6, picks better result
    - Applies static + context-aware corrections
    """
    preprocessed = preprocess_image(image_path)
    
    t4 = pytesseract.image_to_string(
        preprocessed, config='--oem 3 --psm 4 -c preserve_interword_spaces=1', lang='eng'
    )
    t6 = pytesseract.image_to_string(
        preprocessed, config='--oem 3 --psm 6 -c preserve_interword_spaces=1', lang='eng'
    )
    raw = t4 if len(t4.strip()) >= len(t6.strip()) else t6
    
    return correct_ocr_text(raw, model_answer)


# ─── OCR: MODEL ANSWER IMAGE ─────────────────────────────────────────────────

def extract_model_answer_from_image(image_path: str) -> dict:
    """
    Extract text from a teacher's model answer image using the SAME green-channel
    pipeline as student images (preprocess_image + adaptive scale + PSM 4+6 contest).

    This gives consistent quality across both student and model answer images.

    Returns dict with:
      text: str          — extracted and corrected text
      quality: float     — OCR quality score 0-1
      quality_label: str — "good" / "acceptable" / "poor" / "failed"
      warning: str       — human-readable warning if quality is low

    Raises ValueError if quality < 0.20 (catastrophic failure — dark ruled lines, etc.)
    """
    # Use EXACT same pipeline as student answers
    binary = preprocess_image(image_path)
    t4 = pytesseract.image_to_string(
        binary, config='--oem 3 --psm 4 -c preserve_interword_spaces=1', lang='eng'
    )
    t6 = pytesseract.image_to_string(
        binary, config='--oem 3 --psm 6 -c preserve_interword_spaces=1', lang='eng'
    )
    raw = t4 if len(t4.strip()) >= len(t6.strip()) else t6

    # Score quality on raw text (before correction, so quality reflects true OCR)
    best_quality = ocr_quality_score(raw)

    # Apply same correction pipeline as student answers
    corrected = apply_static_corrections(raw)
    corrected = _final_cleanup(corrected)
    cleaned   = clean_text(corrected)

    # Quality labels and warnings
    if best_quality >= 0.55:
        label   = "good"
        warning = None
    elif best_quality >= 0.35:
        label   = "acceptable"
        warning = (
            f"OCR quality is moderate ({best_quality:.0%}). The model answer image may have "
            "dark ruled lines, poor lighting, or very compressed handwriting. "
            "Consider typing the model answer directly for more reliable grading."
        )
    elif best_quality >= 0.20:
        label   = "poor"
        warning = (
            f"OCR quality is poor ({best_quality:.0%}). The model answer could not be read "
            "reliably from the image. This will result in inaccurate grading. "
            "Please TYPE the model answer instead."
        )
    else:
        raise ValueError(
            f"Cannot extract readable text from the model answer image "
            f"(OCR quality: {best_quality:.0%}). "
            "This typically happens when the image has dark printed ruled lines "
            "with ink of the same color, making the text indistinguishable. "
            "Please TYPE the model answer text directly instead of uploading an image."
        )

    return {
        "text":          cleaned,
        "quality":       round(best_quality, 2),
        "quality_label": label,
        "warning":       warning,
    }


# ─── KEYWORD EXTRACTION ──────────────────────────────────────────────────────

_STOP = {
    "a","an","the","is","are","was","were","be","been","being","have","has",
    "had","do","does","did","will","would","could","should","may","might",
    "must","shall","can","to","of","in","for","on","with","at","by","from",
    "as","into","through","and","or","but","if","then","when","while",
    "because","that","this","these","those","it","its","also","which","who",
    "what","used","use","using","not","no","any","all","one","two","so",
    "such","just","very","each","more","some","other","we","they","he",
    "she","you","i","me","him","her","us","them","like","case","way"
}


def extract_keywords(text: str, top_n: int = 20) -> list:
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    freq: dict = {}
    for w in words:
        if w not in _STOP:
            freq[w] = freq.get(w, 0) + (1 + len(w) * 0.1)
    return sorted(freq, key=freq.get, reverse=True)[:top_n]


def keyword_coverage(student_text: str, model_keywords: list) -> tuple:
    if not model_keywords:
        return 0.0, [], []
    found, missing = [], []
    s = student_text.lower()
    for kw in model_keywords:
        prefix = kw[:max(4, len(kw) - 2)]
        if kw in s or prefix in s:
            found.append(kw)
        else:
            missing.append(kw)
    return len(found) / len(model_keywords), found, missing


# ─── SCORING ─────────────────────────────────────────────────────────────────

def sentence_coverage(student: str, model: str) -> float:
    from sentence_transformers import util
    sents = [s.strip() for s in re.split(r'[.!?]', model) if len(s.strip()) > 10]
    if not sents:
        return 0.0
    m     = get_model()
    s_emb = m.encode(student, convert_to_tensor=True)
    e_emb = m.encode(sents,   convert_to_tensor=True)
    sims  = util.cos_sim(s_emb, e_emb)[0].cpu().numpy()
    top_k = max(1, len(sents) // 2)
    return float(np.mean(sorted(sims, reverse=True)[:top_k]))


def grade_single_answer(
    student_text: str,
    model_answer: str,
    max_marks: float,
    keywords: list = None,
    ocr_quality: float = None,
) -> dict:
    """
    Grade one student answer. Weights: semantic 50% + coverage 25% + keywords 20% + length 5%.
    ocr_quality: pass the student image OCR quality score to include in response.
    """
    from sentence_transformers import util
    
    student_clean = clean_text(student_text, model_answer)
    model_clean   = clean_text(model_answer)
    
    if not student_clean or len(student_clean.split()) < 3:
        return _blank_result(max_marks, keywords or [], ocr_quality)
    
    m = get_model()
    
    embs    = m.encode([student_clean, model_clean], convert_to_tensor=True)
    sem_sim = float(max(0.0, util.cos_sim(embs[0], embs[1]).item()))
    sent_cov = sentence_coverage(student_clean, model_clean)
    
    if keywords is None:
        keywords = extract_keywords(model_clean, top_n=15)
    kw_score, found_kw, missing_kw = keyword_coverage(student_clean, keywords)
    
    length_ratio = min(len(student_clean.split()) / max(len(model_clean.split()), 1), 1.0)
    
    final = min(max(
        0.50 * sem_sim + 0.25 * sent_cov + 0.20 * kw_score + 0.05 * length_ratio, 0.0), 1.0)
    
    marks = round(final * max_marks, 2)
    pct   = round(final * 100, 1)
    std   = float(np.std([sem_sim, sent_cov, kw_score, length_ratio]))
    confidence = "high" if std < 0.15 else "medium" if std < 0.28 else "low"
    
    strengths, improvements = _feedback(sem_sim, sent_cov, kw_score, length_ratio, found_kw, missing_kw)
    
    result = {
        "marks_awarded":       marks,
        "max_marks":           max_marks,
        "percentage":          pct,
        "confidence":          confidence,
        "semantic_similarity": round(sem_sim, 4),
        "sentence_coverage":   round(sent_cov, 4),
        "keyword_coverage":    round(kw_score, 4),
        "length_ratio":        round(length_ratio, 4),
        "keywords_found":      found_kw[:10],
        "keywords_missing":    missing_kw[:10],
        "strengths":           strengths,
        "improvements":        improvements,
        "grade_label":         _label(pct),
    }
    if ocr_quality is not None:
        result["student_ocr_quality"] = round(ocr_quality, 2)
        if ocr_quality < 0.35:
            result["ocr_warning"] = (
                f"Student answer OCR quality is low ({ocr_quality:.0%}). "
                "The score may be inaccurate. For best results: photograph in good lighting, "
                "keep paper flat, ensure handwriting is clear blue or black ink on white paper."
            )
    return result


def _blank_result(max_marks, keywords, ocr_quality=None):
    r = {
        "marks_awarded": 0.0, "max_marks": max_marks, "percentage": 0.0,
        "confidence": "high", "semantic_similarity": 0.0,
        "sentence_coverage": 0.0, "keyword_coverage": 0.0, "length_ratio": 0.0,
        "keywords_found": [], "keywords_missing": keywords, "strengths": [],
        "improvements": ["Answer appears blank or unreadable."],
        "grade_label": "Poor"
    }
    if ocr_quality is not None:
        r["student_ocr_quality"] = round(ocr_quality, 2)
    return r


def _label(pct):
    if pct >= 90: return "Outstanding"
    if pct >= 75: return "Excellent"
    if pct >= 60: return "Good"
    if pct >= 50: return "Average"
    if pct >= 35: return "Below Average"
    return "Poor"


def _feedback(sem, sent, kw, length, found_kw, missing_kw):
    s, i = [], []
    if sem >= 0.75:    s.append("Answer closely matches the expected response in meaning.")
    elif sem >= 0.50:  s.append("Answer is partially aligned with the expected response.")
    else:              i.append("Answer lacks conceptual alignment with the model answer.")
    if sent >= 0.70:   s.append("Most key points from the model answer are covered.")
    elif sent >= 0.40: i.append("Some important points from the model answer are missing.")
    else:              i.append("Very few key points from the model answer were covered.")
    if kw >= 0.70:     s.append(f"Good keyword usage: {', '.join(found_kw[:5])}.")
    elif kw >= 0.40:
        if found_kw:   s.append(f"Partial keyword usage: {', '.join(found_kw[:4])}.")
        if missing_kw: i.append(f"Missing key terms: {', '.join(missing_kw[:5])}.")
    else:
        if missing_kw: i.append(f"Critical terms absent: {', '.join(missing_kw[:6])}.")
    if length < 0.3:   i.append("Answer is too brief — needs more elaboration.")
    elif length >= 0.8: s.append("Answer is well-elaborated.")
    return s, i


# ─── MULTI-QUESTION GRADER ───────────────────────────────────────────────────

def grade_answer_sheet(image_path: str, questions: list) -> dict:
    """Grade a full answer sheet with multiple questions."""
    combined_model = " ".join(q["model_answer"] for q in questions)
    raw_text       = extract_text_from_image(image_path, model_answer=combined_model)
    student_quality = ocr_quality_score(raw_text)
    segments       = split_answers_by_question(raw_text, len(questions))
    
    results, total_awarded, total_max = [], 0.0, 0.0
    
    for i, q in enumerate(questions):
        segment = segments[i] if i < len(segments) else ""
        kw      = q.get("keywords") or extract_keywords(clean_text(q["model_answer"]), top_n=15)
        result  = grade_single_answer(segment, q["model_answer"], q["max_marks"], kw, student_quality)
        result["question_no"]    = q.get("question_no", i + 1)
        result["question_text"]  = q.get("question_text", f"Question {i + 1}")
        result["extracted_text"] = clean_text(segment, q["model_answer"]) or "(No text detected)"
        results.append(result)
        total_awarded += result["marks_awarded"]
        total_max     += q["max_marks"]
    
    overall_pct = round(total_awarded / total_max * 100, 1) if total_max else 0.0
    
    response = {
        "total_marks_awarded":   round(total_awarded, 2),
        "total_max_marks":       total_max,
        "overall_percentage":    overall_pct,
        "overall_grade":         _label(overall_pct),
        "student_ocr_quality":   round(student_quality, 2),
        "ocr_engine":            "Tesseract (green-channel + adaptive scale + context correction)",
        "full_extracted_text":   clean_text(raw_text, combined_model),
        "question_results":      results,
    }
    if student_quality < 0.35:
        response["ocr_warning"] = (
            f"Student answer sheet OCR quality is low ({student_quality:.0%}). "
            "Scores may be inaccurate. Use a clear, well-lit photo of blue or black ink "
            "on plain white paper for best results."
        )
    return response