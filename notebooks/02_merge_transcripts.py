"""
merge_transcriptions.py
-----------------------
Combines transcriptions_1.csv, transcriptions_2.csv, transcriptions_3.csv
into a single transcriptions_merged.csv.

Duplicate handling logic:
  - If a video_id appears more than once, prefer the row that has a
    non-null / non-empty transcription.
  - If multiple rows all have transcriptions, keep the first occurrence
    (files are processed in order: 1 → 2 → 3).
  - If no row has a transcription, keep the first occurrence anyway
    (so the video_id is still represented).

Usage:
    python merge_transcriptions.py

    # Or point to a different input/output location:
    python merge_transcriptions.py \
        --input transcriptions_1.csv transcriptions_2.csv transcriptions_3.csv \
        --output transcriptions_merged.csv
"""

import argparse
import pandas as pd


def has_transcript(value) -> bool:
    """Return True if value is a non-empty, non-null string."""
    if pd.isna(value):
        return False
    return str(value).strip() != ""


def merge_transcriptions(input_files: list[str], output_file: str) -> None:
    # ── 1. Load and concatenate in order ────────────────────────────────────
    frames = []
    for path in input_files:
        df = pd.read_csv(path)
        print(f"  Loaded {path!r:40s}  →  {len(df):>4} rows")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n  Total rows after concat : {len(combined)}")

    # ── 2. Deduplicate ───────────────────────────────────────────────────────
    # Sort so that rows WITH a transcription come first for each video_id.
    # "has_transcript" is False (0) for nulls/empty and True (1) for real text;
    # sorting descending puts the "has transcript" rows at the top.
    combined["_has_transcript"] = combined["transcription"].apply(has_transcript)
    combined_sorted = combined.sort_values(
        by=["video_id", "_has_transcript"],
        ascending=[True, False],   # keep video_id order; transcript rows first
        kind="stable",
    )

    # Drop duplicates, keeping the first row per video_id (which now has a
    # transcript whenever any duplicate did).
    deduped = combined_sorted.drop_duplicates(subset=["video_id"], keep="first")
    deduped = deduped.drop(columns=["_has_transcript"])

    # Restore original file order (first appearance across the three files).
    # Re-index by order of first occurrence in the concatenated frame.
    first_occurrence_order = (
        combined.drop_duplicates(subset=["video_id"], keep="first")["video_id"]
        .reset_index(drop=True)
    )
    deduped = (
        deduped
        .set_index("video_id")
        .loc[first_occurrence_order]
        .reset_index()
    )

    duplicates_removed = len(combined) - len(deduped)
    print(f"  Duplicate rows removed  : {duplicates_removed}")
    print(f"  Final unique video IDs  : {len(deduped)}")

    # ── 3. Transcription coverage summary ───────────────────────────────────
    with_transcript = deduped["transcription"].apply(has_transcript).sum()
    without_transcript = len(deduped) - with_transcript
    print(f"  Rows with transcription : {with_transcript}")
    print(f"  Rows without transcript : {without_transcript}")

    # ── 4. Save ──────────────────────────────────────────────────────────────
    deduped.to_csv(output_file, index=False)
    print(f"\n  ✓ Saved merged file to {output_file!r}")


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge and deduplicate transcript CSVs.")
    parser.add_argument(
        "--input", nargs="+",
        default=[
            "transcriptions_1.csv",
            "transcriptions_2.csv",
            "transcriptions_3.csv",
        ],
        help="Input CSV files in order (default: transcriptions_1/2/3.csv)",
    )
    parser.add_argument(
        "--output", default="transcriptions_merged.csv",
        help="Output file path (default: transcriptions_merged.csv)",
    )
    args = parser.parse_args()

    print("Merging transcription files…\n")
    merge_transcriptions(args.input, args.output)