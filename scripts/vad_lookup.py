#!/usr/bin/env python3
# vad_lookup.py -- NRC VAD word-level emotion scoring (Chinese + English)
#
# Usage: python vad_lookup.py "dream text"
#        python vad_lookup.py -i dream.txt
#        python vad_lookup.py -i dream.txt --json
#        python vad_lookup.py -i dream.txt --lang en   # force English

import sys
import os
import re
import json
import argparse
from collections import defaultdict

# ---- NRC VAD 6-emotion reference coordinates (0-1) ----
EMOTION_VAD = {
    "fear":     (0.083, 0.482, 0.278),   # fearful
    "sadness":  (0.225, 0.333, 0.149),   # sad
    "anger":    (0.122, 0.830, 0.604),   # angry
    "disgust":  (0.051, 0.773, 0.274),   # disgusted
    "joy":      (1.000, 0.735, 0.772),   # happy
    "awe":      (0.650, 0.710, 0.470),   # awe (estimated)
}

# Chinese emotion labels (for display)
EMOTION_ZH = {
    "fear": "恐惧", "sadness": "悲伤", "anger": "愤怒",
    "disgust": "厌恶", "joy": "喜悦", "awe": "敬畏",
}


def detect_language(text):
    """Detect whether text is Chinese-dominant or English-dominant.
    Returns 'zh' or 'en'.
    """
    cjk = 0
    latin = 0
    for ch in text:
        if '\u4e00' <= ch <= '\u9fff' or '\u3400' <= ch <= '\u4dbf':
            cjk += 1
        elif ch.isalpha() and ch.isascii():
            latin += 1
    return "zh" if cjk >= latin else "en"


def load_lexicon_zh(path):
    """Load NRC VAD Chinese lexicon (5-col TSV: en-word V A D zh-word).
    Returns: {zh_word: (V, A, D)} -- many-to-one averaged.
    """
    temp = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            zh = parts[4]
            if not zh or zh.isspace():
                continue
            try:
                temp[zh].append((float(parts[1]), float(parts[2]), float(parts[3])))
            except ValueError:
                continue

    lexicon = {}
    for zh, scores in temp.items():
        n = len(scores)
        lexicon[zh] = (
            sum(s[0] for s in scores) / n,
            sum(s[1] for s in scores) / n,
            sum(s[2] for s in scores) / n,
        )
    return lexicon


def load_lexicon_en(path):
    """Load NRC VAD English lexicon (4-col TSV: word V A D, no header).
    Returns: {word: (V, A, D)}.
    """
    lexicon = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            word = parts[0].lower()
            try:
                lexicon[word] = (float(parts[1]), float(parts[2]), float(parts[3]))
            except ValueError:
                continue
    return lexicon


def segment_zh(text):
    """jieba Chinese word segmentation."""
    import jieba
    return [w.strip() for w in jieba.cut(text) if w.strip()]


def segment_en(text):
    """Simple English word tokenizer (lowercase, strip punctuation)."""
    tokens = re.findall(r"[a-z]+", text.lower())
    return tokens


def score_text(words, lexicon):
    """Score tokenized word list against VAD lexicon."""
    scores = {"V": [], "A": [], "D": []}
    matched = []
    unmatched = []

    for w in words:
        if w in lexicon:
            v, a, d = lexicon[w]
            scores["V"].append(v)
            scores["A"].append(a)
            scores["D"].append(d)
            matched.append({"word": w, "V": round(v, 3), "A": round(a, 3), "D": round(d, 3)})
        else:
            unmatched.append(w)

    if not scores["V"]:
        return None, matched, unmatched

    n = len(scores["V"])
    v_arr = scores["V"]
    a_arr = scores["A"]
    d_arr = scores["D"]

    v_avg = sum(v_arr) / n
    a_avg = sum(a_arr) / n
    d_avg = sum(d_arr) / n

    return {
        "V_avg": v_avg,
        "A_avg": a_avg,
        "D_avg": d_avg,
        "V_std": (sum((x - v_avg) ** 2 for x in v_arr) / n) ** 0.5,
        "A_std": (sum((x - a_avg) ** 2 for x in a_arr) / n) ** 0.5,
        "D_std": (sum((x - d_avg) ** 2 for x in d_arr) / n) ** 0.5,
        "match_rate": n / len(words) if words else 0,
        "matched_count": n,
        "unmatched_count": len(unmatched),
    }, matched, unmatched


def load_emotion_words(path):
    """Load curated emotion-descriptive word list for nearest-neighbor lookup.
    Format: word \\t V \\t A \\t D
    """
    words = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            try:
                words[parts[0]] = (float(parts[1]), float(parts[2]), float(parts[3]))
            except ValueError:
                continue
    return words


def nearest_emotion_words(profile, emotion_lexicon, top_n=10):
    """Find the closest-matching emotion-descriptive words in VAD space.
    Uses a curated emotion-word lexicon (not the full VAD lexicon) for precision.
    """
    if not profile:
        return []
    v, a, d = profile["V_avg"], profile["A_avg"], profile["D_avg"]
    results = []
    for word, (wv, wa, wd) in emotion_lexicon.items():
        dist = ((v - wv) ** 2 + (a - wa) ** 2 + (d - wd) ** 2) ** 0.5
        results.append((word, wv, wa, wd, dist))
    results.sort(key=lambda x: x[4])
    return results[:top_n]


def map_to_emotion(profile):
    """Map VAD profile to nearest 6-category emotion (distance-weighted soft assignment)."""
    if not profile:
        return None, {}
    v, a, d = profile["V_avg"], profile["A_avg"], profile["D_avg"]
    dists = {}
    for emo, (ev, ea, ed) in EMOTION_VAD.items():
        dists[emo] = ((v - ev) ** 2 + (a - ea) ** 2 + (d - ed) ** 2) ** 0.5
    closest = min(dists, key=dists.get)
    total_inv = sum(1.0 / max(d, 1e-6) for d in dists.values())
    weights = {emo: round((1.0 / max(d, 1e-6)) / total_inv, 3) for emo, d in dists.items()}
    return closest, weights


def compute_d_consistency(profile, dominant_emotion):
    """Check if text-aggregated D aligns with dominant-emotion reference D."""
    if not profile:
        return None
    d_text = profile["D_avg"]
    if dominant_emotion not in EMOTION_VAD:
        return None
    d_emo = EMOTION_VAD[dominant_emotion][2]
    text_dir = "低" if d_text < 0.5 else "高"
    emo_dir = "低" if d_emo < 0.5 else "高"
    return {
        "text_D": round(d_text, 3),
        "emotion_D": round(d_emo, 3),
        "text_D_dir": text_dir,
        "emotion_D_dir": emo_dir,
        "consistent": text_dir == emo_dir,
    }


def format_output(profile, matched, unmatched, dominant_emotion, emotion_weights, d_consistency, nearest_words, lang):
    labels = EMOTION_ZH if lang == "zh" else None
    lines = []
    lines.append("=" * 60)
    lines.append("NRC VAD Word-Level Emotion Scoring")
    lines.append(f"Language: {'Chinese' if lang == 'zh' else 'English'}")
    lines.append("=" * 60)

    if not profile:
        lines.append("\n(!) No matched words found in the NRC VAD lexicon.")
        return "\n".join(lines)

    lines.append(f"\nVAD Aggregate Profile (0-1):")
    lines.append(f"  Valence  (V): {profile['V_avg']:.3f}  \u00b1 {profile['V_std']:.3f}")
    lines.append(f"  Arousal  (A): {profile['A_avg']:.3f}  \u00b1 {profile['A_std']:.3f}")
    lines.append(f"  Dominance(D): {profile['D_avg']:.3f}  \u00b1 {profile['D_std']:.3f}")
    lines.append(f"  Match rate: {profile['match_rate']:.1%} ({profile['matched_count']}/{profile['matched_count']+profile['unmatched_count']} words)")

    dom_label = labels[dominant_emotion] if labels else dominant_emotion
    lines.append(f"\nDominant Emotion (nearest 6-anchor): {dom_label}")
    lines.append(f"\nEmotion Distribution (6-anchor soft weights):")
    for emo in sorted(emotion_weights, key=emotion_weights.get, reverse=True):
        label = labels[emo] if labels else emo
        bar = chr(0x2588) * max(1, int(emotion_weights[emo] * 40))
        lines.append(f"  {label:6s} {bar} {emotion_weights[emo]:.3f}")

    if nearest_words:
        lines.append(f"\nNearest Emotion Words (VAD space direct lookup):")
        for word, wv, wa, wd, dist in nearest_words:
            lines.append(f"  {word:16s}  V={wv:.3f} A={wa:.3f} D={wd:.3f}  dist={dist:.3f}")

    if d_consistency:
        status = "OK" if d_consistency["consistent"] else "DIVERGENT"
        lines.append(f"\nD Consistency: {status}")
        lines.append(f"  Text D:    {d_consistency['text_D']:.3f} ({d_consistency['text_D_dir']} dominance)")
        lines.append(f"  Emotion D: {d_consistency['emotion_D']:.3f} ({d_consistency['emotion_D_dir']} dominance)")

    lines.append(f"\nTop 20 Matched Words:")
    matched_sorted = sorted(matched, key=lambda x: abs(x["V"] - 0.5), reverse=True)
    for m in matched_sorted[:20]:
        v_sign = "+" if m["V"] > 0.6 else ("-" if m["V"] < 0.4 else "~")
        lines.append(f"  {m['word']:16s}  V={m['V']:.3f}({v_sign}) A={m['A']:.3f} D={m['D']:.3f}")

    if unmatched:
        lines.append(f"\nUnmatched ({len(unmatched)}): {' '.join(unmatched[:30])}")
        if len(unmatched) > 30:
            lines.append(f"  ... {len(unmatched)} total unmatched words")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="NRC VAD word-level emotion scoring (Chinese + English)")
    parser.add_argument("text", nargs="?", help="Dream text (inline)")
    parser.add_argument("-i", "--input", help="Read dream text from file")
    parser.add_argument("--lang", choices=["zh", "en"], help="Force language (auto-detect by default)")
    parser.add_argument("--lexicon", help="Path to NRC VAD lexicon (overrides auto-selection)")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    # Read text
    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        text = sys.stdin.read()

    if not text.strip():
        print("Error: no text provided", file=sys.stderr)
        sys.exit(1)

    # Determine language
    lang = args.lang or detect_language(text)

    # Determine lexicon path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)  # go up from scripts/ to project root
    if args.lexicon:
        lexicon_path = args.lexicon
    elif lang == "zh":
        lexicon_path = os.path.join(project_dir, "data", "NRC-VAD-Chinese-Simplified.txt")
    else:
        lexicon_path = os.path.join(project_dir, "data", "NRC-VAD-English.txt")

    if not os.path.exists(lexicon_path):
        # fallback relative to cwd
        fallback = os.path.basename(lexicon_path)
        if os.path.exists(fallback):
            lexicon_path = fallback
        else:
            print(f"Error: lexicon not found: {lexicon_path}", file=sys.stderr)
            sys.exit(1)

    # Load lexicon + emotion word list + segment
    emo_words_path = os.path.join(project_dir, "data", "emotion_words_en.txt")
    emotion_lexicon = load_emotion_words(emo_words_path) if os.path.exists(emo_words_path) else {}

    if lang == "zh":
        lexicon = load_lexicon_zh(lexicon_path)
        words = segment_zh(text)
    else:
        lexicon = load_lexicon_en(lexicon_path)
        words = segment_en(text)

    # Score + classify
    profile, matched, unmatched = score_text(words, lexicon)
    dominant_emotion, emotion_weights = map_to_emotion(profile)
    d_consistency = compute_d_consistency(profile, dominant_emotion)
    nearest_words = nearest_emotion_words(profile, emotion_lexicon, 10)

    if args.json:
        output = {
            "language": lang,
            "word_count": len(words),
            "vad_profile": profile,
            "matched_words": matched,
            "unmatched_words": unmatched,
            "dominant_emotion": dominant_emotion,
            "dominant_emotion_zh": EMOTION_ZH.get(dominant_emotion),
            "emotion_weights": emotion_weights,
            "nearest_words": [{"word": w, "V": wv, "A": wa, "D": wd, "distance": round(dist, 4)} for w, wv, wa, wd, dist in nearest_words],
            "d_consistency": d_consistency,
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(format_output(profile, matched, unmatched, dominant_emotion, emotion_weights, d_consistency, nearest_words, lang))


if __name__ == "__main__":
    main()
