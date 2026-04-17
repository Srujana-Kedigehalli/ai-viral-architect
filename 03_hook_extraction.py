"""
Hook Extraction + Merge Pipeline
==================================
Strategy (no models, no downloads, runs fully local):
  Step 1 — Filler sentence filtering (rule-based regex)
  Step 2 — Score early sentences using TF-IDF similarity to the video title
  Step 3 — Pick the highest scoring sentence from the first 20% of transcript
  Step 4 — Fallback to first non-filler sentence if scoring fails
  Step 5 — Merge with top500_popular.csv to produce final 23-column dataset

Requirements (all pre-installed or standard):
  pip install pandas scikit-learn

Usage:
  - Place this script in the same folder as:
      transcripts_merged.csv
      top500_popular.csv
  - Run: python hook_extraction_pipeline.py
  - Output: final_dataset.csv
"""

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────

TRANSCRIPTS_MERGED    = "transcripts_merged.csv"
TOP500_CSV            = "top500_popular.csv"
OUTPUT_CSV            = "transcripts_with_hooks.csv"

# First 20% of sentences are treated as the "early window" for hook search
EARLY_WINDOW_FRACTION = 0.20

# Need at least this many candidates to bother with TF-IDF scoring
MIN_SENTENCES_FOR_TFIDF = 3

# A valid hook candidate must have at least this many words
MIN_HOOK_WORDS = 6


# ──────────────────────────────────────────────────────────────
# FILLER PATTERNS
# Common YouTube intro phrases that are NOT hooks.
# ──────────────────────────────────────────────────────────────

FILLER_PATTERNS = [
    r"welcome\s+back",
    r"hey\s+(guys?|everyone|y'?all|folks|there|what)",
    r"what'?s\s+up\s+(guys?|everyone|y'?all|people)?",
    r"(don'?t\s+forget\s+to\s+)?(like|subscribe|hit\s+the\s+bell)",
    r"smash\s+that\s+(like|subscribe)",
    r"make\s+sure\s+to\s+(like|subscribe|follow|hit)",
    r"in\s+today'?s?\s+video",
    r"before\s+we\s+(get|start|begin|dive|jump)",
    r"thank\s+you\s+(so\s+much\s+)?for\s+watching",
    r"thank\s+you\s+for\s+tuning\s+in",
    r"if\s+you\s+(haven'?t\s+already|like\s+this\s+video)",
    r"drop\s+(a\s+)?(like|comment)\s+below",
    r"let'?s\s+(get\s+)?(started|into\s+it|go|begin|jump\s+right\s+in)",
    r"so\s+without\s+further\s+ado",
    r"(see|catch)\s+you\s+in\s+the\s+next",
    r"my\s+name\s+is\s+\w+\s+and",
    r"^(hi+|hey+|hello+|yo+|okay|ok|alright|so|well|now|um+|uh+)[,\.\!\s]*$",
    r"(this\s+is|i'?m)\s+\w+\s+(and\s+)?(today|welcome|you'?re\s+watching)",
    r"you'?re\s+watching",
    r"(today|this\s+video)\s+(i'?m\s+going\s+to|we'?re\s+going\s+to|we\s+will|i\s+will)\s+(show|talk|discuss|cover|look)",
]

FILLER_RE = re.compile("|".join(FILLER_PATTERNS), re.IGNORECASE)


# ──────────────────────────────────────────────────────────────
# SENTENCE SPLITTING
# ──────────────────────────────────────────────────────────────

def split_sentences(text: str) -> list:
    """
    Split transcript into sentences.
    Handles newlines, ellipses, and standard punctuation boundaries.
    """
    text = re.sub(r"\n+", " ", text.strip())
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Za-z])', text)
    return [s.strip() for s in sentences if s.strip()]


def is_filler(sentence: str) -> bool:
    """Return True if the sentence is a known YouTube filler/intro phrase."""
    return bool(FILLER_RE.search(sentence.strip()))


def is_valid_candidate(sentence: str) -> bool:
    """Valid hook candidate: not filler and has enough words."""
    return len(sentence.split()) >= MIN_HOOK_WORDS and not is_filler(sentence)


# ──────────────────────────────────────────────────────────────
# HOOK EXTRACTION
# ──────────────────────────────────────────────────────────────

def extract_hook(transcript: str, title: str) -> str:
    """
    Extract hook from transcript using TF-IDF similarity to the video title.

    Logic:
      1. Split transcript into sentences
      2. Filter out filler sentences
      3. Look at the first 20% of sentences as the "early window"
      4. Score each early-window sentence by cosine similarity to the title
      5. Return the highest-scoring sentence (original text, no paraphrasing)
      6. Fallback: first valid sentence if TF-IDF can't run
    """
    if not transcript or not transcript.strip():
        return ""

    sentences = split_sentences(transcript)

    # Collect valid (non-filler, long enough) sentences with their positions
    valid_sentences = [(i, s) for i, s in enumerate(sentences)
                       if is_valid_candidate(s)]

    if not valid_sentences:
        # Nothing passes the filter — return raw first sentence as last resort
        return sentences[0] if sentences else ""

    # Early window: first 20% of all sentences
    early_cutoff = max(1, int(len(sentences) * EARLY_WINDOW_FRACTION))
    early_candidates = [(i, s) for i, s in valid_sentences if i <= early_cutoff]

    # If early window is empty, widen to first 5 valid sentences anywhere
    if not early_candidates:
        early_candidates = valid_sentences[:5]

    # Only 1 candidate — no need to score, just return it
    if len(early_candidates) < MIN_SENTENCES_FOR_TFIDF:
        return early_candidates[0][1]

    # ── TF-IDF Scoring ──────────────────────────────────────────
    # Corpus = [title] + [each early candidate sentence]
    # We measure which candidate sentence shares the most
    # important terms with the title — that sentence is the hook.
    candidate_texts = [s for _, s in early_candidates]
    corpus = [title] + candidate_texts

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),   # unigrams + bigrams
            min_df=1,
            sublinear_tf=True,    # log-dampens frequent terms
        )
        tfidf_matrix  = vectorizer.fit_transform(corpus)
        title_vec     = tfidf_matrix[0]
        candidate_vecs = tfidf_matrix[1:]
        scores        = cosine_similarity(title_vec, candidate_vecs).flatten()
        best_idx      = scores.argmax()
        return candidate_texts[best_idx]

    except Exception:
        # Vectorizer fails on very short/unusual text — safe fallback
        return early_candidates[0][1]


def extract_body(transcript: str, hook: str) -> str:
    """
    Body = everything after the hook in the original transcript.
    Since the hook is always a literal substring, we just slice after it.
    """
    if not hook or not transcript:
        return transcript or ""

    idx = transcript.find(hook)
    if idx == -1:
        # Shouldn't happen, but if hook isn't found, drop first sentence
        sentences = split_sentences(transcript)
        return " ".join(sentences[1:]) if len(sentences) > 1 else transcript

    return transcript[idx + len(hook):].strip()


# ──────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Hook Extraction + Merge Pipeline")
    print("=" * 55)

    # ── Load ────────────────────────────────────────────────────
    print("\n[1/4] Loading data...")
    df_transcripts = pd.read_csv(TRANSCRIPTS_MERGED)
    df_top500      = pd.read_csv(TOP500_CSV)
    print(f"      transcripts_merged : {df_transcripts.shape[0]} rows")
    print(f"      top500_popular      : {df_top500.shape[0]} rows")

    df_transcripts["transcription"] = df_transcripts["transcription"].fillna("")

    # ── Merge first so titles are available for TF-IDF ──────────
    print("\n[2/4] Merging transcripts with top500 metadata...")
    df_top500_deduped = df_top500.drop_duplicates(subset=["video_id"], keep="first")
    df = pd.merge(df_transcripts, df_top500_deduped, on="video_id", how="inner")
    print(f"      Rows after inner merge : {len(df)}")

    df["_wc"]  = df["transcription"].apply(lambda x: len(str(x).split()))
    df_valid   = df[df["_wc"] >= MIN_HOOK_WORDS].copy()
    df_empty   = df[df["_wc"] <  MIN_HOOK_WORDS].copy()
    print(f"      Usable transcripts  : {len(df_valid)}")
    print(f"      Empty/too short     : {len(df_empty)}")

    # ── Hook Extraction ─────────────────────────────────────────
    print("\n[3/4] Extracting hooks (TF-IDF + filler filter)...")

    hooks, bodies = [], []
    total = len(df_valid)

    for i, (_, row) in enumerate(df_valid.iterrows()):
        transcript = str(row["transcription"])
        title      = str(row.get("title", ""))

        hook = extract_hook(transcript, title)
        body = extract_body(transcript, hook)

        hooks.append(hook)
        bodies.append(body)

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"      [{i+1}/{total}] done")

    df_valid = df_valid.copy()
    df_valid["hook"]            = hooks
    df_valid["body"]            = bodies
    df_valid["full_transcript"] = df_valid["transcription"]
    df_valid["word_count"]      = df_valid["_wc"]

    # Fill empty-transcript rows with blank hook/body
    if len(df_empty) > 0:
        df_empty = df_empty.copy()
        df_empty["hook"]            = ""
        df_empty["body"]            = ""
        df_empty["full_transcript"] = ""
        df_empty["word_count"]      = 0

    df_final = pd.concat([df_valid, df_empty], ignore_index=True)

    # ── Column Order ─────────────────────────────────────────────
    print("\n[4/4] Saving final dataset...")
    desired_cols = [
        "video_id", "title", "publishedAt", "channelId", "channelTitle",
        "categoryId", "trending_date", "tags", "view_count", "likes",
        "dislikes", "comment_count", "thumbnail_link", "comments_disabled",
        "ratings_disabled", "description", "category", "country",
        "engagement_ratio", "hook", "body", "full_transcript", "word_count",
    ]
    available_cols = [c for c in desired_cols if c in df_final.columns]
    df_final = df_final[available_cols]
    df_final.to_csv(OUTPUT_CSV, index=False)

    # ── Summary ──────────────────────────────────────────────────
    hooks_filled = (df_final["hook"].str.strip() != "").sum()
    hooks_empty  = (df_final["hook"].str.strip() == "").sum()

    print(f"\n{'='*55}")
    print(f"  Done!")
    print(f"  Rows   : {df_final.shape[0]}  |  Cols : {df_final.shape[1]}")
    print(f"  Hooks extracted : {hooks_filled}")
    print(f"  Hooks empty     : {hooks_empty}")
    print(f"  Output          : {OUTPUT_CSV}")
    print(f"{'='*55}")

    # ── Preview ──────────────────────────────────────────────────
    print("\n--- Sample extracted hooks ---")
    preview = df_final[df_final["hook"].str.strip() != ""].head(8)
    for _, row in preview.iterrows():
        print(f"\n  video   : {row['video_id']}")
        print(f"  title   : {str(row.get('title',''))[:65]}")
        print(f"  category: {row.get('category', '?')}")
        print(f"  hook    : {str(row['hook'])[:120]}")

    return df_final


if __name__ == "__main__":
    df = main()