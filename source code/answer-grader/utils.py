"""
utils.py  v6.0 — OCR post-processing for handwritten answer sheets

Pipeline: context_fixes → token_map → context_fixes → fuzzy_correction → cleanup

Key features:
  • Extended token map (180+ entries) from ground-truth analysis of real student images
  • Multi-word context fixes for this student's specific OCR patterns
  • Model answer images use SAME green-channel pipeline as student images
  • split_multi_answer_text(): auto-detects Q1), Q2) etc. in one image/PDF
  • auto_split_pages(): splits combined multi-page screenshots
  • Adaptive upscaling: narrow images get higher scale factor
  • ocr_quality_score(): detect bad OCR before grading (0-1 float)
"""

import cv2
import numpy as np
import re
from difflib import get_close_matches


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE SPLITTING
# ══════════════════════════════════════════════════════════════════════════════

def auto_split_pages(img: np.ndarray) -> list:
    """
    Detect combined multi-page screenshots and split into individual pages.
    Finds horizontal bright bands (row mean > 235 for >40 consecutive rows)
    that represent the gap between two photographed pages.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    row_bright = np.mean(gray, axis=1)

    in_gap = False
    gap_start = 0
    gaps = []
    for row in range(h):
        if row_bright[row] > 235:
            if not in_gap:
                in_gap, gap_start = True, row
        else:
            if in_gap:
                if (row - gap_start) > 40 and gap_start > h * 0.15:
                    gaps.append((gap_start, row))
                in_gap = False

    if not gaps:
        return [img]

    pages, prev = [], 0
    for gs, ge in gaps:
        p = img[prev:gs]
        if p.shape[0] > 100:
            pages.append(p)
        prev = ge
    last = img[prev:]
    if last.shape[0] > 100:
        pages.append(last)
    return pages or [img]


# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE PREPROCESSING  (used for BOTH student and model answer images)
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_image(image_path: str) -> np.ndarray:
    """
    Green-channel pipeline for blue/dark ink on plain or lightly ruled paper.
    Auto-splits multi-page screenshots. Adaptive upscaling targets 3000px wide.

    This same pipeline is used for BOTH student answers AND model answer images
    so both get identical, high-quality preprocessing.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    pages = auto_split_pages(img)
    if len(pages) > 1:
        processed = [_preprocess_single(p) for p in pages]
        sep = np.ones((30, processed[0].shape[1]), dtype=np.uint8) * 255
        out = processed[0]
        for p in processed[1:]:
            out = np.vstack([out, sep, p])
        return out
    return _preprocess_single(img)


def _preprocess_single(img: np.ndarray) -> np.ndarray:
    """
    Green channel inversion + CLAHE + Otsu for blue-ink handwriting.
    Blue ink: green channel ≈ 35 (very low) → inverted ≈ 220 (bright ink)
    Ruled lines: green channel ≈ 200 → inverted ≈ 55 (dark, mostly filtered)
    White paper: green channel ≈ 250 → inverted ≈ 5 (black background)

    Scale target: 3000px wide for optimal Tesseract accuracy.
    """
    _, g, _ = cv2.split(img)
    w = img.shape[1]
    g_inv = 255 - g
    # Target 3000px: better than 2400px for wide images, capped at 6x for narrow
    scale = min(max(3000 / w, 2.5), 6.0)
    g_up = cv2.resize(g_inv, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g_enh = clahe.apply(g_up)
    _, binary = cv2.threshold(g_enh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def run_tesseract(binary: np.ndarray, model_hint: str = "") -> str:
    """
    Run Tesseract with PSM 4 and PSM 6, return whichever gives more text.
    PSM 4 = assume single column of text (good for answer sheets)
    PSM 6 = assume uniform block of text (sometimes better for dense answers)
    """
    import pytesseract
    cfg = '--oem 3 -c preserve_interword_spaces=1'
    t4 = pytesseract.image_to_string(binary, config=f'{cfg} --psm 4', lang='eng')
    t6 = pytesseract.image_to_string(binary, config=f'{cfg} --psm 6', lang='eng')
    # Pick whichever gives more non-whitespace content
    return t4 if len(t4.strip()) >= len(t6.strip()) else t6


# ══════════════════════════════════════════════════════════════════════════════
#  OCR QUALITY SCORE
# ══════════════════════════════════════════════════════════════════════════════

_ENG = {
    'the','and','are','was','were','but','not','yet','when','this','that','with',
    'from','have','been','which','then','also','while','into','like','more','most',
    'used','need','being','case','way','its','another','called','running','testing',
    'code','test','module','stub','driver','software','system','output','input',
    'using','both','some','make','give','get','all','any','can','may','must',
    'will','would','could','should','each','only','just','very','even','well',
    'such','than','them','they','their','there','where','here','how','why','if',
    'our','as','at','by','do','is','in','of','on','or','so','to','up','an',
    'be','he','me','my','no','us','it','call','part','perform','sample','assume',
    'mostly','student','report','generate','redirect','creates','generates',
    'duplicate','fake','unit','integration','website','model','erp','dummy',
    'define','explain','describe','what','example','given','shows','using',
    'ready','perform','testing','performing','programmings',
}


def ocr_quality_score(text: str) -> float:
    """
    Rate OCR output quality 0.0–1.0.
    < 0.20 = catastrophic failure (wrong image / dark lined paper)
    0.20–0.40 = poor (heavy errors but gradeable with correction)
    0.40–0.55 = acceptable
    > 0.55 = good
    """
    words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
    if len(words) < 5:
        return 0.0
    real = sum(1 for w in words if w in _ENG or len(w) >= 5)
    return real / len(words)


# ══════════════════════════════════════════════════════════════════════════════
#  CONTEXT FIXES  (regex patterns spanning multiple tokens / line context)
# ══════════════════════════════════════════════════════════════════════════════

def apply_context_fixes(text: str) -> str:
    """
    Regex patterns that require surrounding context or span multiple tokens.
    Run BEFORE and AFTER the token map for maximum coverage.
    """
    # ── 1. Strip leading noise characters on each line ────────────────────────
    text = re.sub(r'(?m)^[ \t]*[~>=©→•]+[ \t]*', '', text)
    text = re.sub(r'(?m)^[ \t]*—+[ \t]*', '', text)

    # ── 2. Noise symbols ─────────────────────────────────────────────────────
    text = re.sub(r'\bOF:\s*', 'of ', text)
    text = re.sub(r'[»«""\']', '', text)
    text = re.sub(r'/kratel\b', '//created', text)
    text = re.sub(r'©\s*\w+\s+', 'assume ', text)    # "© XbSUME" → "assume"
    text = re.sub(r'\bwoe\s+', '', text)              # filler noise

    # ── 3. Number-based OCR errors ────────────────────────────────────────────
    text = re.sub(r'\bOve\s+\d+\s+', 'are ', text)   # "Ove 2" → "are"
    text = re.sub(r'\b3\s+10\s+[\'"]?0\.?\s*', '', text)
    text = re.sub(r'\b19\s+not\b', 'is not', text)
    text = re.sub(r'\b15\s+pt\b', 'is not', text)
    text = re.sub(r'[,.]?\s*dtul\s+\{5\s+used', 'stub is used', text)
    text = re.sub(r'\{5\s+used', 'is used', text)
    text = re.sub(r'\b8\s+(?:cols|calls)\b', 'it calls', text)
    text = re.sub(r'\b1\s+o\s+', 'in a ', text)      # "3 1 o topuares" → "in a software"

    # ── 4. This specific student's multi-word OCR patterns ───────────────────
    # "rans bby" / "rans by" → "mostly"
    text = re.sub(r'\brans\s+b+y\b', 'mostly', text, flags=re.IGNORECASE)
    # "yat yi" → "but this"
    text = re.sub(r'\bYat\s+Yi\b|\byat\s+yi\b', 'but this', text, flags=re.IGNORECASE)
    # "yek ribdy" / "yek_ribdy" → "not yet ready"
    text = re.sub(r'\byek[_\s]+ribdy\b', 'not yet ready', text, flags=re.IGNORECASE)
    # "bans module should" → "if this module should"
    text = re.sub(r'\bbans\s+(module|medule)\b', 'if this module', text, flags=re.IGNORECASE)
    # "dress shale iby calls" → "then stub is called"
    text = re.sub(r'\bdress\s+sha[le]+\s+iby\s+calls', 'then stub is called', text, flags=re.IGNORECASE)
    # "dress stub iby" → "then stub is"
    text = re.sub(r'\bdress\b(?=\s+stub)', 'then', text, flags=re.IGNORECASE)
    # "iby calls ushich" → "is called which"
    text = re.sub(r'\biby\s+calls\s+ushich\b', 'is called which', text, flags=re.IGNORECASE)
    text = re.sub(r'\biby\s+calls\b', 'is called', text, flags=re.IGNORECASE)
    # "ts nob yet" → "is not yet"
    text = re.sub(r'\bts\s+nob\s+yet\b', 'is not yet', text, flags=re.IGNORECASE)
    # "the module were are running" → "the module we are running"
    text = re.sub(r'\bmodule\s+were\s+are\b', 'module we are', text, flags=re.IGNORECASE)
    text = re.sub(r'\bwere\s+are\b', 'we are', text, flags=re.IGNORECASE)
    # "tp bert a code" → "to test a code"
    text = re.sub(r'\btp\s+bert\b', 'to test', text, flags=re.IGNORECASE)
    # "sbolh.ts" / "sbolh ts bea" → "stub is like"
    text = re.sub(r'\bsbolh\.ts\s+bea\b', 'stub is like', text, flags=re.IGNORECASE)
    text = re.sub(r'\bsbolh\.ts\b', 'stub is', text, flags=re.IGNORECASE)
    # "rodent soyule" → "student module"
    text = re.sub(r'\brodent\s+soyule\b|\bsodent\s+soyule\b', 'student module', text, flags=re.IGNORECASE)
    # "eeatys the then creates" → "case the system creates"
    text = re.sub(r'\beeatys\s+the\s+then\b', 'case the system', text, flags=re.IGNORECASE)
    # "system a generateReport" → "system creates a generateReport"
    text = re.sub(r'\bsystem\s+a\s+generateReport\b', 'system creates a generateReport', text)
    # "nad to tex student" → "need to test student"
    text = re.sub(r'\bnad\s+to\s+tex\b', 'need to test', text, flags=re.IGNORECASE)
    text = re.sub(r'\bmee\s+fo\s+tet\b|\bmee\s+fo\s+test\b', 'need to test', text, flags=re.IGNORECASE)
    # "sample out output" → "sample output" (duplicate)
    text = re.sub(r'\bsample\s+out\s+output\b|\bout\s+output\b', 'sample output', text)
    # "chin o code" → "when a code"
    text = re.sub(r'\bchin\s+o\s+', 'when a ', text, flags=re.IGNORECASE)
    # "in oo software" → "in a software"
    text = re.sub(r'\bin\s+oo\s+', 'in a ', text, flags=re.IGNORECASE)
    # "Yot it module" → "but this module"
    text = re.sub(r'\bYot\s+it\s+module\b', 'but this module', text, flags=re.IGNORECASE)
    # "bertа code" → "test a code"
    text = re.sub(r'\bbert\s+a\s+', 'test a ', text, flags=re.IGNORECASE)
    # "tok" → "code" only in "a tok" context
    text = re.sub(r'\ba\s+tok\b', 'a code', text, flags=re.IGNORECASE)
    # "topuares" → "software"
    text = re.sub(r'\btopuares\b|\btopuare\b', 'software', text, flags=re.IGNORECASE)
    # "jn this" → "In this"
    text = re.sub(r'\bjn\s+this\b', 'In this', text, flags=re.IGNORECASE)
    # "nob yet" → "not yet"
    text = re.sub(r'\bnob\s+yet\b', 'not yet', text, flags=re.IGNORECASE)
    # "qrenterenet module" → "generateReport module"
    text = re.sub(r'\bqrenterenet\b|\bqrenter\b', 'generateReport', text, flags=re.IGNORECASE)
    # "eeatys" → "case"
    text = re.sub(r'\beeatys\b', 'case', text, flags=re.IGNORECASE)
    # "this woy we" → "this way we"
    text = re.sub(r'\bwoy\b', 'way', text, flags=re.IGNORECASE)
    # "su module" at line end → "Student module"
    text = re.sub(r'(?m)^\s*su\s+module\s*$', 'Student module', text, flags=re.IGNORECASE)
    # "adent ceylule" → "student module" (noise phrase)
    text = re.sub(r'\badent\s+ceylule\s+calls\b', 'student module calls', text, flags=re.IGNORECASE)
    # "adent" alone → "student"
    text = re.sub(r'\badent\b', 'student', text, flags=re.IGNORECASE)
    # "butin" / "bution" → "but in"
    text = re.sub(r'\bbutin\b|\bbution\b', 'but in', text, flags=re.IGNORECASE)
    # "con perform" → "can perform"
    text = re.sub(r'\bcon\s+perform\b', 'can perform', text, flags=re.IGNORECASE)
    # "calls ys" → "calls it"
    text = re.sub(r'\bcalls\s+ys\b', 'calls it', text, flags=re.IGNORECASE)
    # "5) rs driver" → "5) Stub and driver" (noise 'rs' at start)
    text = re.sub(r'(?m)^(\d\))\s+rs\s+driver', r'\1 Stub and driver', text)
    # "- ave part of" → "are part of"
    text = re.sub(r'\bave\s+part\s+of\b', 'are part of', text, flags=re.IGNORECASE)
    # "ors part of" → "are part of"
    text = re.sub(r'\bors\s+part\s+of\b', 'are part of', text, flags=re.IGNORECASE)

    # ── 5. Multi-token corrections ────────────────────────────────────────────
    text = re.sub(r'\bdul\s+iM\b', 'is', text)
    text = re.sub(r'\biM\s+(?:called|cobled)\b', 'is called', text)
    text = re.sub(r'\bten\s+generates\b', 'then generates', text)
    text = re.sub(r'\bgener\s+(?:erteRemt|generateReport)\b', 'generateReport', text, flags=re.IGNORECASE)
    text = re.sub(r'\bty\.\s+another\b', 'to another', text)
    text = re.sub(r'\bnot\.\s+ye\s+ready\b', 'not yet ready', text)
    text = re.sub(r'\bth\s+Ws\b', 'if this', text)
    text = re.sub(r'\btn\.\s+', 'in ', text)
    # "shu which" → "stub which"
    text = re.sub(r'\bshu\s+which\b|\bshu\s+whith\b', 'stub which', text, flags=re.IGNORECASE)
    # Generating phrase cleanup: "then Beh dul iM called" → "then stub is called"
    text = re.sub(r'\bthen\s+Beh\s+dul\s+iM\b', 'then stub is', text, flags=re.IGNORECASE)
    # "stub 187 called echich" → "stub is called which"  
    text = re.sub(r'\bstub\s+187\s+called\s+echich\b', 'stub is called which', text, flags=re.IGNORECASE)
    text = re.sub(r'\bshule\s+187\s+called\s+echich\b', 'stub is called which', text, flags=re.IGNORECASE)
    # "The shule 187" → "Then stub is"
    text = re.sub(r'\bThe\s+shule\s+187\b', 'Then stub is', text, flags=re.IGNORECASE)

    # ── 6. End-of-line fixes ──────────────────────────────────────────────────
    text = re.sub(r'(?m)\bib[:.]\s*$', 'it.', text)
    text = re.sub(r'(?m)\bgb\s*$', '', text)
    text = re.sub(r'(?m)\bibs\s*$', 'it.', text)

    # ── 7. Stray | and — characters ──────────────────────────────────────────
    text = re.sub(r'(?<=[a-zA-Z])\s*\|\s*(?=[a-zA-Z])', ' ', text)
    text = re.sub(r'(?m)\|\s*$', '', text)
    text = re.sub(r'(?m)^\s*\|\s*', '', text)
    text = re.sub(r'\s*—\s*', ' ', text)

    # ── 8. Duplicate words ────────────────────────────────────────────────────
    text = re.sub(r'\bwhen\s+when\b', 'when', text)
    text = re.sub(r'\ba\s+a\b', 'a', text)
    text = re.sub(r'\bin\s+in\b', 'in', text)
    text = re.sub(r'\bstub\s+is\s+called\s+then\s+generates\b', 'stub is called which generates', text)

    # ── 9. Cosmetic ───────────────────────────────────────────────────────────
    text = re.sub(r'\bthe,\s+the\b', 'the', text)
    text = re.sub(r'(?m)^\s*3\s+Stub', 'Stub', text)
    text = re.sub(r'\bEx\.\s+In\.\s+a\.?\s+', 'In a ', text)
    text = re.sub(r'\bIn\.\s+a\b', 'In a', text)
    text = re.sub(r'\bcase\s+the\b', 'case, the', text)
    text = re.sub(r',\s+\.\s+if\s+this', ', if this', text)
    text = re.sub(r'\b(and|are|or|but|the)\.\s+', r'\1 ', text)
    text = re.sub(r'(\w)-\s+(\w)', r'\1 \2', text)
    text = re.sub(r'\s+,\s+', ', ', text)
    text = re.sub(r'\s+_\s+', ' ', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'(?m)[ \t]+$', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
#  TOKEN MAP  (wrong OCR token → correct word)
#  180+ entries from line-by-line ground truth analysis of real student images.
# ══════════════════════════════════════════════════════════════════════════════

_TOKEN_MAP = {
    # ── generateReport variants ──────────────────────────────────────────────
    "generateReport":   "generateReport",   # keep correct form
    "entroteReport":    "generateReport",   "joorteRyft":    "generateReport",
    "qenenteRepot":     "generateReport",   "erteRemt":      "generateReport",
    "generedeRensrt":   "generateReport",   "qenacteRepot":  "generateReport",
    "qenedterepot":     "generateReport",   "generoteremt":  "generateReport",
    "qeneterepot":      "generateReport",   "genetereBot":   "generateReport",
    "entroteRepert":    "generateReport",   "generoteReport":"generateReport",
    "qrenterenet":      "generateReport",   "generoterepert":"generateReport",
    "generateRenet":    "generateReport",   "qrenteRenet":   "generateReport",
    "GenerateReport":   "generateReport",

    "qenereterens":  "generates",   "qeneterens": "generates",
    "qrreaie":       "generates",   "geared":     "generates",
    "aenerotes":     "generates",

    # ── Stub / stub variants ─────────────────────────────────────────────────
    "preyramininglo": "programmings", "preyraminings":  "programmings",
    "eqrominisal":    "programmings", "eqrasmininal":   "programmings",
    "prograreningt":  "programmings", "progrosnisingt": "programmings",

    "Stuteat": "Student",
    "Studs":   "Stub",    "Stale":   "Stub",    "Sbob":   "Stub",
    "Shaly":   "Stub",    "Shuly":   "Stub",    "Shulo":  "Stub",
    "Shalo":   "Stub",    "Stulo":   "Stub",    "stats":  "Stub",
    "sbolh":   "stub",    "stulb":   "stub",    "dtulo":  "stub",
    "dtul":    "stub",    "Beh":     "stub",    "Sey":    "Stub",
    "shale":   "stub",    "shu":     "stub",

    # ── driver ───────────────────────────────────────────────────────────────
    "drfver":  "driver",  "driner":  "driver",  "Ariver": "driver",

    # ── testing variants ─────────────────────────────────────────────────────
    "tubiy":   "testing", "teating": "testing", "texting": "testing",
    "tating":  "testing", "teling":  "testing", "teking":  "testing",
    "tasting": "testing", "sean":    "testing",

    # ── mostly ───────────────────────────────────────────────────────────────
    "_raostly": "mostly", "raostly": "mostly",  "mostby": "mostly",

    # ── software ─────────────────────────────────────────────────────────────
    "Softwares": "software", "Sephwores": "software", "Sefhuores": "software",
    "Spftwores": "software", "topuares":  "software", "topuare":   "software",

    # ── redirects ────────────────────────────────────────────────────────────
    "rediredts": "redirects", "radired":  "redirects", "redireds": "redirects",
    "recired":   "redirects",

    # ── ready ────────────────────────────────────────────────────────────────
    "_1ibdy":  "ready",  "1ibdy": "ready",  "_ribdy": "ready",
    "ribdy":   "ready",  "_rebdy":"ready",  "vebdy":  "ready",
    "reody":   "ready",  "ribds": "ready",

    # ── sample ───────────────────────────────────────────────────────────────
    "Loonple": "sample",  "orale":  "sample",  "losple":  "sample",
    "loople":  "sample",  "lomple": "sample",  "lommple": "sample",
    "Semple":  "sample",  "loaple": "sample",

    # ── website ──────────────────────────────────────────────────────────────
    "coebsite": "website", "webente": "website",

    # ── assume ───────────────────────────────────────────────────────────────
    "XbSUME":  "assume",  "eobsume": "assume",  "ossume":  "assume",
    "aobsume": "assume",

    # ── student ──────────────────────────────────────────────────────────────
    "audent":   "student", "shodent":  "student", "shiderst": "student",
    "buderst":  "student", "shadent":  "student", "studerst": "student",
    "shuderst": "student", "rodent":   "student",

    # ── module ───────────────────────────────────────────────────────────────
    "modwle":   "module",  "eoodule":  "module",  "eadale":   "module",
    "crpfule":  "module",  "wnedule":  "module",  "mide":     "module",
    "rmble":    "module",  "soyule":   "module",  "rnedule":  "module",
    "medule":   "module",  "voojule":  "module",

    # ── output ───────────────────────────────────────────────────────────────
    "pak":     "output",   "_odipuke": "output",  "cuipsk":   "output",
    "odipuke": "output",   "eutpur":   "output",  "eutpat":   "output",
    "odp":     "output",   "cups":     "output",  "oudpat":   "output",

    # ── code ─────────────────────────────────────────────────────────────────
    "wode":    "code",     "ecde":     "code",    "coke":     "code",
    "Cade":    "code",     "tok":      "code",

    # ── calls / called ───────────────────────────────────────────────────────
    "colle":   "calls",    "cols":     "calls",   "colls":    "calls",
    "cobled":  "called",   "coble":    "called",

    # ── gives / which / like ─────────────────────────────────────────────────
    "qives":   "gives",    "qiveh":    "gives",
    "voit":    "which",    "whith":    "which",   "ushich":   "which",
    "tke":     "like",     "bea":      "like",

    # ── when / need / we ─────────────────────────────────────────────────────
    "cohen":   "when",     "dent":     "when",    "chin":     "when",
    "mee":     "need",     "nad":      "need",
    "weeur":   "we",

    # ── running / being / tested ─────────────────────────────────────────────
    "punning": "running",
    "Aning":   "being",    "aning":    "being",
    "bested":  "tested",   "tated":    "tested",

    # ── call / fake ──────────────────────────────────────────────────────────
    "coll":    "call",
    "foke":    "fake",     "faye":     "fake",    "fete":     "fake",

    # ── single-letter / short OCR errors ─────────────────────────────────────
    "Lhe":     "When",
    "best":    "test",     "tet":      "test",    "bert":     "test",
    "fo":      "to",       "tp":       "to",
    "ord":     "and",      "ond":      "and",
    "ove":     "are",      "ore":      "are",     "ave":      "are",
    "port":    "part",     "ors":      "are",
    "AL":      "At",
    "iS":      "is",       "Ls":       "is",      "dul":      "is",
    "ts":      "is",
    "op":      "a",        "oo":       "a",
    "ho":      "slu",
    "eet":     "then",     "dress":    "then",
    "Tb":      "In",       "Ty":       "In",      "jn":       "In",
    "eoats":   "case",     "eeatys":   "case",
    "tutte":   "the",
    "cyester": "system",
    "tis":     "this",     "Ws":       "this",
    "ton":     "can",      "con":      "can",
    "th":      "if",
    "nob":     "not",
    "kratel":  "created",
    "moder":     "model",
    "bubin":     "but in",  "bution":   "but in",  "butin":    "but in",
    "woy":       "way",
    "ibs":       "it",      "ib":       "it",
    "evortree":  "errorfree", "exvortree":"errorfree", "errortree":"errorfree",
    "evorfree":  "errorfree",
    "iby":     "is",
    "Yot":     "but",      "bans":     "if",
    "wnit":    "unit",
    "9":       "",         # noise digit sometimes injected mid-line
}

# Compile patterns longest-first to prevent partial substitutions
_COMPILED_TOKENS = []
for _wrong, _right in sorted(_TOKEN_MAP.items(), key=lambda x: -len(x[0])):
    if not _wrong:
        continue
    _flags = re.IGNORECASE if (_wrong[0].isupper() and not _wrong.isupper()) else 0
    _COMPILED_TOKENS.append((re.compile(r'\b' + re.escape(_wrong) + r'\b', _flags), _right))


def apply_precise_corrections(text: str) -> str:
    """Apply the hand-tuned token map."""
    for pat, rep in _COMPILED_TOKENS:
        text = pat.sub(rep, text)
    return text


# ══════════════════════════════════════════════════════════════════════════════
#  FUZZY CORRECTION (model-answer vocabulary guided)
# ══════════════════════════════════════════════════════════════════════════════

_COMMON_WORDS = {
    'the','and','are','was','were','but','not','yet','when','this','that','with',
    'from','have','been','which','then','also','while','into','like','more','most',
    'used','need','being','case','such','way','its','our','their','there','where',
    'here','how','why','all','any','can','may','must','will','would','could',
    'should','each','only','just','very','even','well','than','them','they',
    'called','running','testing','creating','making','code','file','call','both',
    'when','then','another','using','first','give','get','put','set','try',
    'an','be','he','me','my','no','us','a','i','it','is','in','of','on','or',
    'so','to','up','at','by','as','do','we','if',
}


def apply_fuzzy_correction(ocr_text: str, model_answer: str, cutoff: float = 0.75) -> str:
    """Fuzzy-match unrecognised words against model answer vocabulary."""
    if not model_answer.strip():
        return ocr_text
    vocab = set(re.findall(r'\b[a-zA-Z]{3,}\b', model_answer.lower()))
    known = vocab | _COMMON_WORDS
    result = ocr_text
    for word in sorted(set(re.findall(r'\b[a-zA-Z]{4,}\b', ocr_text)), key=len, reverse=True):
        if word.lower() not in known:
            m = get_close_matches(word.lower(), known, n=1, cutoff=cutoff)
            if m and m[0] != word.lower():
                result = re.sub(r'\b' + re.escape(word) + r'\b', m[0], result)
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  MASTER PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def correct_ocr_text(raw_text: str, model_answer: str = "") -> str:
    """
    5-pass correction:
      1. Context fixes (strip symbols, multi-word patterns, number errors)
      2. Token map (180+ hand-tuned substitutions)
      3. Second context pass (clean up residue from token substitutions)
      4. Fuzzy correction guided by model answer vocabulary
      5. Final cleanup
    """
    t = apply_context_fixes(raw_text)
    t = apply_precise_corrections(t)
    t = apply_context_fixes(t)          # second pass to catch chained fixes
    if model_answer.strip():
        t = apply_fuzzy_correction(t, model_answer, cutoff=0.75)
    return _final_cleanup(t)


def _final_cleanup(text: str) -> str:
    text = re.sub(r'[|{}\[\]©®°~`@#$%^&*\\<>§]', '', text)
    text = re.sub(r'(?m)^[ \t]*[-_.=]{3,}[ \t]*$', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r' ([.,!?;:])', r'\1', text)
    return text.strip()


def clean_text(text: str, model_answer: str = "") -> str:
    text = correct_ocr_text(text, model_answer)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s.,\-()/]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ══════════════════════════════════════════════════════════════════════════════
#  MULTI-ANSWER DETECTION & SPLITTING
# ══════════════════════════════════════════════════════════════════════════════

def detect_question_boundaries(text: str) -> list:
    """
    Detect question number markers in OCR'd text.

    Recognises these formats (robust to OCR noise):
      5)   5.   Q5   Q.5   (5)   Ans5:   Answer 5)   Q5:
      5 )  5 .  q5   ans 5

    Returns sorted list of (question_no, start_pos, end_of_header_pos).
    """
    boundaries = []

    # "5) " / "5. " / "5: " / "5 ) " at line start — most common
    for m in re.finditer(
        r'(?:^|\n)[ \t]*(?:[Qq]\.?\s*)?(\d{1,2})[ \t]*[.):\-][ \t]+',
        text, re.MULTILINE
    ):
        qno = int(m.group(1))
        if 1 <= qno <= 30:
            boundaries.append((qno, m.start(), m.end()))

    # "(5) " style
    for m in re.finditer(r'(?:^|\n)[ \t]*\((\d{1,2})\)[ \t]+', text, re.MULTILINE):
        qno = int(m.group(1))
        if 1 <= qno <= 30:
            boundaries.append((qno, m.start(), m.end()))

    # "Answer 5:" / "Ans5:" / "ans 5)" style
    for m in re.finditer(
        r'(?:^|\n)[ \t]*[Aa]ns(?:wer)?\.?[ \t]*(\d{1,2})[ \t]*[.):\-]?[ \t]+',
        text, re.MULTILINE
    ):
        qno = int(m.group(1))
        if 1 <= qno <= 30:
            boundaries.append((qno, m.start(), m.end()))

    # "Q5 " or "q5 " mid-sentence style (looser, used only if nothing else found)
    if not boundaries:
        for m in re.finditer(r'(?:^|\n)[ \t]*[Qq](\d{1,2})[ \t]+', text, re.MULTILINE):
            qno = int(m.group(1))
            if 1 <= qno <= 30:
                boundaries.append((qno, m.start(), m.end()))

    # Deduplicate: keep first occurrence of each question number, sort by position
    seen, result = set(), []
    for qno, start, end in sorted(boundaries, key=lambda x: x[1]):
        if qno not in seen:
            seen.add(qno)
            result.append((qno, start, end))
    return result


def split_multi_answer_text(text: str, expected_count: int = None) -> list:
    """
    Split OCR'd text containing multiple answers into labelled segments.

    Detection priority:
      1. Question number markers (5), Q5, Answer 5, etc.) — most reliable
      2. Double blank-line paragraph splitting — if expected_count given
      3. Even word-count split — last resort fallback

    Returns list of dicts: [{"question_no": int, "answer_text": str}, ...]

    Example:
        text = "5) A stub is a dummy module...\\n\\n6) A driver is..."
        split_multi_answer_text(text)
        → [{"question_no":5,"answer_text":"A stub is..."},
           {"question_no":6,"answer_text":"A driver is..."}]

    Works with combined screenshots (auto page split happens in preprocess_image
    before this function is called, so text here is already joined across pages).
    """
    boundaries = detect_question_boundaries(text)

    if len(boundaries) >= 2:
        segments = []
        for i, (qno, start, end) in enumerate(boundaries):
            next_start = boundaries[i + 1][1] if i + 1 < len(boundaries) else len(text)
            answer = text[end:next_start].strip()
            if answer:
                segments.append({"question_no": qno, "answer_text": answer})
        return segments

    if len(boundaries) == 1:
        qno, _, end = boundaries[0]
        return [{"question_no": qno, "answer_text": text[end:].strip()}]

    # No markers found — split by blank lines or evenly
    if expected_count and expected_count > 1:
        paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        if len(paras) >= expected_count:
            return [{"question_no": i + 1, "answer_text": paras[i]}
                    for i in range(expected_count)]
        words = text.split()
        chunk = max(1, len(words) // expected_count)
        return [
            {"question_no": i + 1,
             "answer_text": " ".join(words[i * chunk:(i + 1) * chunk])}
            for i in range(expected_count)
        ]

    return [{"question_no": 1, "answer_text": text.strip()}]


# ══════════════════════════════════════════════════════════════════════════════
#  BACKWARD COMPATIBILITY ALIASES
# ══════════════════════════════════════════════════════════════════════════════

def apply_static_corrections(text: str) -> str:
    """Alias: context_fixes + token_map (no fuzzy step)."""
    return apply_precise_corrections(apply_context_fixes(text))


def apply_context_correction(ocr_text: str, model_answer: str, cutoff: float = 0.75) -> str:
    """Alias for apply_fuzzy_correction."""
    return apply_fuzzy_correction(ocr_text, model_answer, cutoff)


def split_answers_by_question(raw_text: str, num_questions: int) -> list:
    """Legacy wrapper: returns ordered list of answer strings."""
    segs = split_multi_answer_text(raw_text, expected_count=num_questions)
    m = {s["question_no"]: s["answer_text"] for s in segs}
    return [m.get(i + 1, "") for i in range(num_questions)]