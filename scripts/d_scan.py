#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# d_scan.py -- LPP Step 1 D 维度扫描（可计算维度）
#
# 用法：
#   python d_scan.py -i 文本.txt                  # 单文本扫描（D 维度 JSON）
#   python d_scan.py -i 文本.txt --json
#   python d_scan.py --multi f1.txt f2.txt f3.txt  # 多文本 → 面具厚度指数（跨语境变异度）
#
# 设计：把 LPP Step 1 中"可计算"的 D 维度从 LLM 目测改为脚本测量——
#       LLM 拿到的是一组数值而非印象。不可计算的维度（D5/D7/D13/D16/D17/D19/D20/
#       D23/D26-29/D31-40 等）仍由 LLM 观察，脚本输出 note 标注。
# 依赖：jieba；复用 lsm_compare.py 的 FUNCTION_WORDS（同目录）；S9/VAD 类复用 vad_lookup。
# 状态：△ 探索性 —— 阈值与权重为 v1.0 起点，待真实语料校准。

import sys
import os
import re
import json
import argparse
import math
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import lsm_compare  # FUNCTION_WORDS / SENT_SPLIT / cosine
import vad_lookup   # segment_zh / score_text / load_lexicon_zh

FW = lsm_compare.FUNCTION_WORDS
SENT_SPLIT = lsm_compare.SENT_SPLIT

# ---- 各维度专用词表（△ 探索性）----
NEGATIONS = ["不", "没", "无", "别", "未", "勿", "非", "没有", "不是"]
CAUSAL = ["因为", "所以", "因此", "由于", "导致", "因而", "故", "从而"]
TIME_ANCHOR = re.compile(r"\d+[年月日点时分]|昨天|今天|明天|后天|前天|上周|下周|这周|周[一二三四五六日天]|早上|晚上|下午|中午|凌晨")
QUESTION_WORDS = ["什么", "怎么", "为什么", "哪", "谁", "吗", "呢"]
MODAL = {
    "epistemic": ["可能", "应该", "大概", "也许", "似乎", "估计"],
    "deontic": ["必须", "要", "应该", "得", "务必"],
    "dynamic": ["能", "会", "可以", "能够", "敢"],
}
ACTION_VERBS = ["做", "去", "改", "写", "走", "买", "吃", "看", "拿", "建", "做", "完成", "推进"]
STATE_VERBS = ["是", "有", "觉得", "感到", "认为", "想", "知道", "希望", "喜欢", "需要"]
FULLWIDTH_PUNCT = "，。！？；：、""''（）"
HALFWIDTH_PUNCT = ",.!?;:"

# 面具厚度核心维度（跨语境稳定性的代理；对齐 SKILL.md 面具厚度定义 D15/D8/D6/D25/D1）
MASK_DIMS = ["D1", "D6", "D8", "D15", "D25"]


def scan_text(text):
    """扫描单文本，返回 D 维度字典。"""
    n = max(len(text), 1)
    words = vad_lookup.segment_zh(text)
    wc = max(len(words), 1)
    sents = [s for s in SENT_SPLIT.split(text) if s.strip()]
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    n_sents = max(len(sents), 1)

    def per100(cnt):
        return round(cnt * 100.0 / n, 3)

    def count_words(lst):
        return sum(w in lst for w in words)

    def count_substr(lst):
        return sum(text.count(w) for w in lst)

    # 词频统计
    particle_hits = sum(w in FW["particle"] for w in words)
    filler_hits = sum(w in FW["filler"] for w in words)
    abs_hits = sum(w in FW["degree_abs"] for w in words)
    soft_hits = sum(w in FW["degree_soft"] for w in words)
    literary_hits = sum(w in FW["literary"] for w in words)
    prep_hits = sum(w in FW["prep"] for w in words)

    # 人称
    p_wo = sum(w == "我" for w in words)
    p_ni = sum(w == "你" for w in words)
    p_women = sum(w in ("我们", "咱们") for w in words)
    p_total = p_wo + p_ni + p_women or 1

    # 句长
    lens = [len(s) for s in sents]
    mean_len = sum(lens) / n_sents
    sd_len = math.sqrt(sum((l - mean_len) ** 2 for l in lens) / n_sents)
    cv_len = sd_len / max(mean_len, 1e-6)
    commas_per_sent = text.count("，") / n_sents + text.count(",") / n_sents

    # 标点
    half_punct = sum(text.count(c) for c in HALFWIDTH_PUNCT)
    full_punct = sum(text.count(c) for c in FULLWIDTH_PUNCT)
    punct_total = half_punct + full_punct or 1
    ellipsis = text.count("…") + text.count("。。")

    # 段落
    n_paras = max(len(paras), 1)
    single_sent_paras = sum(1 for p in paras if len([s for s in SENT_SPLIT.split(p) if s.strip()]) <= 1)

    # 情绪（VAD）
    vad_profile, _, _ = vad_lookup.score_text(words, vad_lookup.load_lexicon_zh(_lexicon_path()))
    if vad_profile:
        v_avg = vad_profile["V_avg"]
        a_avg = vad_profile["A_avg"]
        match_rate = vad_profile["match_rate"]
    else:
        v_avg = a_avg = 0.5
        match_rate = 0.0

    # 情态三分
    modal_counts = {k: count_words(v) for k, v in MODAL.items()}

    dims = {
        "D1": {"value": round(half_punct / punct_total, 3), "note": "半角标点占比（高=规范性弱）"},
        "D2": {"value": round(per100(len(sents)), 3), "note": "句号/句频次每百字（断句密度）"},
        "D3": {"value": round(single_sent_paras / n_paras, 3), "note": "单句成段率"},
        "D4": {"value": round(mean_len, 2), "note": "平均句长（字）；sd=" + str(round(sd_len, 2)) + " cv=" + str(round(cv_len, 2))},
        "D6": {"value": per100(particle_hits), "note": "语气词每百字"},
        "D8": {"value": round(p_wo / p_total, 3), "note": "我:你:我们 = %d:%d:%d" % (p_wo, p_ni, p_women)},
        "D9": {"value": round(v_avg, 3), "note": "VAD V 均值；A=" + str(round(a_avg, 3)) + " 命中率=" + str(round(match_rate, 3))},
        "D10": {"value": per100(abs_hits), "note": "绝对化词每百字"},
        "D11": {"value": per100(count_substr(NEGATIONS)), "note": "否定词每百字"},
        "D12": {"value": per100(count_substr(CAUSAL)), "note": "因果连接词每百字"},
        "D15": {"value": per100(literary_hits + prep_hits), "note": "书面标记+介词每百字（正式度代理）"},
        "D18": {"value": round(len(TIME_ANCHOR.findall(text)) * 1000.0 / n, 2), "note": "时间锚点每千字"},
        "D21": {"value": round(commas_per_sent, 2), "note": "平均逗号数/句（节奏代理）"},
        "D22": {"value": per100(filler_hits), "note": "填充词每百字（信息密度反向代理）"},
        "D24": {"value": per100(soft_hits), "note": "不确定性词每百字"},
        "D25": {"value": per100(filler_hits), "note": "话语标记每百字（同 D22 源）"},
        "D30": {"value": modal_counts, "note": "情态三分：认识/道义/动力"},
        "D33": {"value": per100(count_substr(QUESTION_WORDS)) + text.count("？") * 100.0 / n, "note": "疑问倾向（疑问词+问号每百字）"},
        "D37": {"value": round(count_words(ACTION_VERBS) / max(count_words(ACTION_VERBS) + count_words(STATE_VERBS), 1), 3), "note": "动作动词占比（动作/状态偏向）"},
        "D_ellipsis": {"value": ellipsis, "note": "省略号/。计数（D1 辅助）"},
    }
    return dims


def _lexicon_path():
    """VAD 词表路径：本包优先，fallback dream-iea。"""
    here = os.path.dirname(os.path.abspath(__file__))
    pkg = os.path.join(here, "..", "references", "lexicons", "zh-CN-NRC-VAD-Lexicon.txt")
    if os.path.exists(pkg):
        return pkg
    return os.path.join(here, "..", "..", "dream-iea", "data", "NRC-VAD-Chinese-Simplified.txt")


def mask_thickness(dim_rows):
    """面具厚度指数：多份文本（同一人不同语境）的跨语境变异度。
    取 MASK_DIMS 各维度跨文本变异系数（CV），综合映射到 0-1。
    变异大 = 语境切换时风格切换明显 = 面具厚（social masking 操作化代理）。
    注意：厚 ≠ 坏——可能是职场/社交熟练度；需结合 ND4（masking 标记）解读。"""
    cvs = {}
    for d in MASK_DIMS:
        vals = [float(r[d]["value"]) for r in dim_rows if d in r]
        if len(vals) < 2:
            continue
        mean = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
        cvs[d] = sd / max(abs(mean), 1e-6)
    if not cvs:
        return None, {}
    m = sum(cvs.values()) / len(cvs)
    thickness = m / (1.0 + m)  # 映射到 0-1
    return round(thickness, 3), {k: round(v, 3) for k, v in cvs.items()}


def main():
    ap = argparse.ArgumentParser(description="LPP D 维度扫描 · △探索性")
    ap.add_argument("-i", "--input", help="单文本文件路径")
    ap.add_argument("--multi", nargs="+", help="多文本（同一人不同语境）→ 面具厚度指数")
    ap.add_argument("--json", action="store_true", help="JSON 输出")
    args = ap.parse_args()

    if args.multi:
        rows = []
        for f in args.multi:
            txt = open(f, encoding="utf-8").read()
            rows.append({"file": os.path.basename(f), **scan_text(txt)})
        thickness, cvs = mask_thickness(rows)
        out = {"mode": "mask", "n_texts": len(rows), "mask_thickness": thickness, "cv_by_dim": cvs,
               "note": "面具厚度=跨语境风格变异度（MASK_DIMS: " + "/".join(MASK_DIMS) + "）；厚≠坏，结合 ND4 解读"}
        if args.json:
            print(json.dumps(out, ensure_ascii=False, indent=2))
        else:
            print("面具厚度指数: %s  (CV: %s)" % (thickness, cvs))
        return

    if not args.input:
        ap.error("需提供 -i 或 --multi")
    text = open(args.input, encoding="utf-8").read()
    dims = scan_text(text)
    out = {"mode": "dscan", "len": len(text), "dims": dims,
           "note": "可计算 D 维度；不可计算维度（D5/D7/D13/D16/D17/D19/D20/D23/D26-29/D31-40）由 LLM 观察"}
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        for k in sorted(dims.keys()):
            v = dims[k]
            print("%-10s %-8s %s" % (k, v["value"], v["note"]))


if __name__ == "__main__":
    main()
