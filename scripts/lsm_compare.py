#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# lsm_compare.py -- 语言风格匹配度（LSM）计算（中文）
#
# 用法：
#   python lsm_compare.py -a 文本A.txt -b 文本B.txt            # 双方独立样本
#   python lsm_compare.py -d 对话.txt                          # 对话文本（A:/B: 前缀或空行块轮流归人）
#   python lsm_compare.py -a a.txt -b b.txt --json             # JSON 输出（供 LLM 消费）
#   python lsm_compare.py -d 对话.txt --segments 5             # 轨迹模式：按轮次分 5 段分别算 LSM
#
# 依赖：jieba；S5 复用 vad_lookup（优先本包 vendored 副本，fallback 到 vault dream-iea）
# 状态：△ 探索性 —— 词表与权重为 v1.2 起点，待真实语料校准（见 references/lsm-zh.md）
# v1.2（2026-08-28）：S9 消息节奏（聊天粒度）+ 表情并入 S2 + 对话模式权重（S3 10% / S9 10%）+ 时间戳间隔

import sys
import os
import re
import json
import argparse
import math
from collections import Counter, defaultdict

# ---- 复用 vad_lookup（S5 情绪同频）：优先本包 vendored 副本，fallback 到 vault dream-iea ----
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_FALLBACK = os.path.join(_HERE, "..", "..", "dream-iea", "scripts")
if not os.path.exists(os.path.join(_HERE, "vad_lookup.py")) and _FALLBACK not in sys.path:
    sys.path.insert(0, _FALLBACK)
import vad_lookup  # segment_zh / score_text / load_lexicon_zh / detect_language

# ---- 功能词词表（与 references/lsm-zh.md 同步，2026-08-27 建立；△ 探索性）----
FUNCTION_WORDS = {
    "pronoun": ["我", "你", "他", "她", "它", "我们", "你们", "他们", "她们", "咱们", "这", "那",
                "这个", "那个", "这些", "那些", "什么", "怎么", "为什么", "哪", "谁"],
    "aux": ["的", "地", "得", "了", "着", "过", "正在"],
    "particle": ["吧", "吗", "呢", "啊", "呀", "哦", "嘛", "哈", "嗯"],
    "conj": ["和", "跟", "还有", "以及", "但是", "可是", "不过", "然而", "其实", "因为", "所以",
             "因此", "既然", "而且", "并且", "甚至", "如果", "假如", "只要", "除非"],
    "filler": ["就", "然后", "就是说", "怎么说呢", "那个", "呃", "对对对", "没错", "确实", "反正", "讲真", "说实话"],
    "degree_abs": ["总是", "从不", "必须", "完全", "根本"],
    "degree_soft": ["可能", "大概", "也许", "应该", "似乎"],
    "prep": ["在", "从", "向", "对", "为", "于", "以", "被", "将", "把", "关于", "对于"],
    "literary": ["其", "之", "该", "及", "亦", "尚", "已", "欲", "宜", "若", "诸", "此", "彼", "皆", "遂", "颇"],
}
# 语气词/填充词用于 S2/S4（S2 用 particle + 表情；S4 用 filler）
# 独立样本权重对齐 lpp-lsm SKILL.md：S1 30% · S3 20% · S5 20% · S6 15% · S2 10%（S7/S8 由 LLM 观察）
WEIGHTS = {"S1": 0.30, "S3": 0.20, "S5": 0.20, "S6": 0.15, "S2": 0.10}
# 对话模式权重（v1.2）：S9 消息节奏 10% 从 S3 拆出（聊天场景节奏=句+消息双粒度）
DIALOG_WEIGHTS = {"S1": 0.30, "S3": 0.10, "S9": 0.10, "S5": 0.20, "S6": 0.15, "S2": 0.10}
# 计算维度的实际权重和（S7 5% 不参与脚本计算，由 LLM 在报告中补）——两种模式均为 0.95
_WEIGHT_SUM = sum(WEIGHTS.values())
_DIALOG_WEIGHT_SUM = sum(DIALOG_WEIGHTS.values())

SENT_SPLIT = re.compile(r"[。！？!?；;\n]+")
EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U0001F600-\U0001F64F\U0001F900-\U0001F9FF\u2600-\u27BF\uFE0F\u2764]"
)
TEXT_EMOJI = [":)", ":-(", "^_^", "T_T", "QAQ", "233", "hhh", "wwww"]


def split_speakers_from_dialog(text):
    """对话文本 → [(speaker, turn_text, timestamp_or_None)]。
    优先解析 '[HH:MM] A: xxx' 或 'HH:MM A: xxx' 前缀（支持 A/B/甲/乙/我/你/ta）；无前缀时按空行块轮流归人。
    """
    prefix = re.compile(
        r"^\s*(?:\[?(\d{1,2}:\d{2}(?::\d{2})?)\]?[\s-]*)?([A-G]|甲|乙|丙|丁|我|你|ta|TA|对方)\s*[：:]\s*(.*)$"
    )
    turns = []
    cur_speaker = None
    cur_text = []
    cur_ts = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        m = prefix.match(line)
        if m:
            if cur_speaker is not None and cur_text:
                turns.append((cur_speaker, " ".join(cur_text), cur_ts))
            cur_speaker = m.group(2)[0].upper()
            cur_ts = m.group(1)
            cur_text = [m.group(3)]
        else:
            if cur_speaker is None:
                continue
            cur_text.append(line)
    if cur_speaker is not None and cur_text:
        turns.append((cur_speaker, " ".join(cur_text), cur_ts))
    if not turns:
        # 无前缀：空行块轮流
        blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
        for i, b in enumerate(blocks):
            turns.append(("A" if i % 2 == 0 else "B", b, None))
    return turns


def aggregate_by_speaker(turns):
    """[(speaker, text, ts)] → {speaker: concat_text}，取字数最多的两个 speaker 为 A/B。"""
    agg = defaultdict(list)
    for s, t, _ts in turns:
        agg[s].append(t)
    joined = {s: " ".join(ts) for s, ts in agg.items()}
    ranked = sorted(joined.items(), key=lambda kv: len(kv[1]), reverse=True)
    if len(ranked) >= 2:
        return {ranked[0][0]: ranked[0][1], ranked[1][0]: ranked[1][1]}
    return joined


def word_features(text):
    """分词 + 词类命中计数。返回 (counter, total_tokens, 各类命中次数)。"""
    words = vad_lookup.segment_zh(text) if text else []
    total = max(len(words), 1)
    cat_hits = Counter()
    for w in words:
        for cat, lst in FUNCTION_WORDS.items():
            if w in lst:
                cat_hits[cat] += 1
                break
    return words, total, cat_hits


def cosine(a, b):
    """两个 Counter 的余弦相似度。"""
    if not a or not b:
        return 0.0
    inter = set(a) & set(b)
    num = sum(a[k] * b[k] for k in inter)
    den = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return num / den if den else 0.0


def s1_function_words(txt_a, txt_b):
    """S1 功能词分布相似度：各词类相对频率向量（对数缩放）余弦 × 总密度相似度。
    密度调节：书面语功能词密度显著高于口语——密度差异大时压低相似度。"""
    def hits(txt):
        h = Counter()
        for w in vad_lookup.segment_zh(txt):
            for cat, lst in FUNCTION_WORDS.items():
                if w in lst:
                    h[cat] += 1
                    break
        return h
    fa = hits(txt_a)
    fb = hits(txt_b)
    na = max(sum(fa.values()), 1)
    nb = max(sum(fb.values()), 1)
    va = Counter({k: math.log1p(v) / math.log1p(na) for k, v in fa.items()})
    vb = Counter({k: math.log1p(v) / math.log1p(nb) for k, v in fb.items()})
    shape = cosine(va, vb)
    # 密度：每百字功能词数（近似密度代理用 词类命中总数/文本长度）
    da = na * 100.0 / max(len(txt_a), 1)
    db = nb * 100.0 / max(len(txt_b), 1)
    dens = 1.0 - abs(da - db) / max(da + db, 1e-6)
    return shape * dens


def s2_particles(txt_a, txt_b):
    """S2 语气词谱 + 表情符号（v1.2）：语气词/emoji/颜文字合并的"语气标记密度"相似度。"""
    def density(txt):
        n = max(len(txt), 1)
        cnt = sum(txt.count(w) for w in FUNCTION_WORDS["particle"])
        cnt += len(EMOJI_RE.findall(txt))
        cnt += sum(txt.count(w) for w in TEXT_EMOJI)
        return cnt * 100.0 / n
    da, db = density(txt_a), density(txt_b)
    return 1.0 - abs(da - db) / max(da + db, 1e-6)


def s3_rhythm(txt_a, txt_b):
    """S3 句长节奏：平均句长 + 变异系数的综合相似度。"""
    def rhythm(txt):
        sents = [s for s in SENT_SPLIT.split(txt) if s.strip()]
        if not sents:
            return 0.0, 0.0
        lens = [len(s) for s in sents]
        mean = sum(lens) / len(lens)
        var = math.sqrt(sum((l - mean) ** 2 for l in lens) / len(lens))
        return mean, var / max(mean, 1e-6)
    ma, va = rhythm(txt_a)
    mb, vb = rhythm(txt_b)
    sim_mean = 1.0 - abs(ma - mb) / max(ma + mb, 1e-6)
    sim_var = 1.0 - abs(va - vb) / max(va + vb, 1e-6)
    return 0.7 * sim_mean + 0.3 * sim_var


def s5_vad_distance(txt_a, txt_b, lexicon_zh):
    """S5 情绪同频：双方 VAD 坐标欧氏距离 → 相似度。
    调节：情绪证据不足（命中率低）时回退中性分 0.5——短文本无情绪词不等于"情绪同频"。"""
    def profile(txt):
        words = vad_lookup.segment_zh(txt)
        p, _, _ = vad_lookup.score_text(words, lexicon_zh)
        if p is None:
            return (0.5, 0.5, 0.5), 0.0
        return (p["V_avg"], p["A_avg"], p["D_avg"]), p["match_rate"]
    va, ma = profile(txt_a)
    vb, mb = profile(txt_b)
    dist = math.sqrt(sum((x - y) ** 2 for x, y in zip(va, vb))) / math.sqrt(3.0)
    raw = 1.0 - dist
    eff = (ma + mb) / 2.0  # 平均情绪词命中率
    sim = 0.5 + (raw - 0.5) * eff  # eff→0 时回退中性 0.5；eff→1 时取 raw
    return sim, va, vb, {"match_rate_a": ma, "match_rate_b": mb}


def s6_content_echo(txt_a, txt_b):
    """S6 内容呼应（近似）：非功能词 top 词的交集率（词面近似，非 LSS 语义）。
    说明：真正语义相似需向量模型；此处为可运行代理指标。"""
    def content_words(txt):
        cw = []
        for w in vad_lookup.segment_zh(txt):
            if len(w) < 2:
                continue
            if any(w in lst for lst in FUNCTION_WORDS.values()):
                continue
            cw.append(w)
        return set(cw)
    sa, sb = content_words(txt_a), content_words(txt_b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    return inter / math.sqrt(len(sa) * len(sb))


def s9_message_rhythm(turns, key_a, key_b):
    """S9 消息节奏（聊天特有，v1.2）：消息长度分布 + 消息内句数 + 时间戳间隔模式。
    仅 -d 对话模式激活——书面样本无消息粒度。返回 (sim, meta)。"""
    def stats_for(key):
        lens, sents, ts = [], [], []
        for sp, tx, t in turns:
            if sp != key:
                continue
            lens.append(len(tx))
            sents.append(len([s for s in SENT_SPLIT.split(tx) if s.strip()]))
            if t:
                parts = t.split(":")
                ts.append(int(parts[0]) * 60 + int(parts[1]))
        if not lens:
            return None
        mean_l = sum(lens) / len(lens)
        cv_l = math.sqrt(sum((l - mean_l) ** 2 for l in lens) / len(lens)) / max(mean_l, 1e-6)
        mean_s = sum(sents) / len(sents)
        intervals = None
        if len(ts) >= 2:
            ts.sort()
            ints = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
            intervals = sum(ints) / len(ints)
        return mean_l, cv_l, mean_s, intervals
    sa, sb = stats_for(key_a), stats_for(key_b)
    if not sa or not sb:
        return 0.5, {"note": "消息不足"}
    sim_len = 1.0 - abs(sa[0] - sb[0]) / max(sa[0] + sb[0], 1e-6)
    sim_cv = 1.0 - abs(sa[1] - sb[1]) / max(sa[1] + sb[1], 1e-6)
    sim_sent = 1.0 - abs(sa[2] - sb[2]) / max(sa[2] + sb[2], 1e-6)
    has_ts = sa[3] is not None and sb[3] is not None
    if has_ts:
        sim_int = 1.0 - abs(sa[3] - sb[3]) / max(sa[3] + sb[3], 1e-6)
        sim = 0.35 * sim_len + 0.25 * sim_cv + 0.2 * sim_sent + 0.2 * sim_int
    else:
        sim = 0.4 * sim_len + 0.3 * sim_cv + 0.3 * sim_sent
    n_a = len([1 for sp, _, _ in turns if sp == key_a])
    n_b = len([1 for sp, _, _ in turns if sp == key_b])
    meta = {"msg_a": n_a, "msg_b": n_b, "has_ts": has_ts}
    return sim, meta


def lsm_score(txt_a, txt_b, lexicon_zh, turns=None, key_a=None, key_b=None, verbose=False):
    """计算各维相似度 + 加权 LSM 分数。对话模式（turns 传入）启用 S9 消息节奏。"""
    s1 = s1_function_words(txt_a, txt_b)
    s2 = s2_particles(txt_a, txt_b)
    s3 = s3_rhythm(txt_a, txt_b)
    s5, va, vb, _ = s5_vad_distance(txt_a, txt_b, lexicon_zh)
    s6 = s6_content_echo(txt_a, txt_b)
    dims = {"S1": s1, "S2": s2, "S3": s3, "S5": s5, "S6": s6}
    if turns is not None and key_a and key_b:
        s9, s9_meta = s9_message_rhythm(turns, key_a, key_b)
        dims["S9"] = s9
        weights = DIALOG_WEIGHTS
        score = sum(weights[k] * v for k, v in dims.items() if k in weights) / _DIALOG_WEIGHT_SUM
    else:
        weights = WEIGHTS
        score = sum(weights[k] * v for k, v in dims.items() if k in weights) / _WEIGHT_SUM
    if verbose:
        return score, dims, va, vb, weights
    return score, dims, va, vb


def verdict(score):
    if score >= 0.75:
        return "高同频"
    if score >= 0.55:
        return "中同频"
    return "低同频"


def main():
    ap = argparse.ArgumentParser(description="语言风格匹配度（LSM）计算 · △探索性")
    ap.add_argument("-a", help="文本 A 文件路径")
    ap.add_argument("-b", help="文本 B 文件路径")
    ap.add_argument("-d", "--dialog", help="对话文本文件路径（自动分离发言人）")
    ap.add_argument("--segments", type=int, default=0, help="轨迹模式：把对话按轮次分为 N 段分别计算 LSM")
    ap.add_argument("--json", action="store_true", help="JSON 输出")
    ap.add_argument("--lexicon", help="NRC VAD 词表路径（覆盖自动选择）")
    args = ap.parse_args()

    # 载入 VAD 词表：优先本包 references/lexicons/zh-CN-NRC-VAD-Lexicon.txt，fallback dream-iea/data
    _HERE2 = os.path.dirname(os.path.abspath(__file__))
    _pkg_lex = os.path.join(_HERE2, "..", "references", "lexicons", "zh-CN-NRC-VAD-Lexicon.txt")
    _ext_lex = os.path.join(_HERE2, "..", "..", "dream-iea", "data", "NRC-VAD-Chinese-Simplified.txt")
    if args.lexicon:
        lexicon_path = args.lexicon
    elif os.path.exists(_pkg_lex):
        lexicon_path = _pkg_lex
    else:
        lexicon_path = _ext_lex
    lexicon_zh = vad_lookup.load_lexicon_zh(lexicon_path)

    if args.dialog:
        raw = open(args.dialog, encoding="utf-8").read()
        turns = split_speakers_from_dialog(raw)
        if len(turns) < 2:
            print(json.dumps({"error": "对话轮次不足（需 ≥2 轮）"}, ensure_ascii=False))
            sys.exit(1)
        if args.segments > 1:
            seg_size = max(1, len(turns) // args.segments)
            seg_scores = []
            for i in range(0, len(turns), seg_size):
                seg = turns[i:i + seg_size]
                agg = aggregate_by_speaker(seg)
                if len(agg) < 2:
                    continue
                keys = list(agg.keys())
                s, dims, _, _ = lsm_score(agg[keys[0]], agg[keys[1]], lexicon_zh,
                                          turns=seg, key_a=keys[0], key_b=keys[1])
                seg_scores.append({"segment": i // seg_size + 1, "lsm": round(s, 3), "verdict": verdict(s)})
            trend = "上升" if len(seg_scores) >= 2 and seg_scores[-1]["lsm"] > seg_scores[0]["lsm"] + 0.05 else (
                "下降" if len(seg_scores) >= 2 and seg_scores[-1]["lsm"] < seg_scores[0]["lsm"] - 0.05 else "持平")
            out = {"mode": "trajectory", "segments": seg_scores, "trend": trend}
            print(json.dumps(out, ensure_ascii=False, indent=2))
            return
        agg = aggregate_by_speaker(turns)
        keys = list(agg.keys())
        txt_a, txt_b = agg[keys[0]], agg[keys[1]]
        dlg_keys = (keys[0], keys[1])
    elif args.a and args.b:
        txt_a = open(args.a, encoding="utf-8").read()
        txt_b = open(args.b, encoding="utf-8").read()
        dlg_keys = None
    else:
        ap.error("需提供 -a/-b 或 -d")

    score, dims, va, vb = lsm_score(txt_a, txt_b, lexicon_zh,
                                    turns=(turns if dlg_keys else None),
                                    key_a=(dlg_keys[0] if dlg_keys else None),
                                    key_b=(dlg_keys[1] if dlg_keys else None))
    result = {
        "lsm": round(score, 3),
        "verdict": verdict(score),
        "dims": {k: round(v, 3) for k, v in dims.items()},
        "vad_a": {"V": round(va[0], 3), "A": round(va[1], 3), "D": round(va[2], 3)},
        "vad_b": {"V": round(vb[0], 3), "A": round(vb[1], 3), "D": round(vb[2], 3)},
        "len_a": len(txt_a), "len_b": len(txt_b),
        "note": "△探索性：词表/权重 v1.2 待真实语料校准；相关非因果；S6 为词面近似非语义相似；S9 仅对话模式",
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("LSM 分数: %.3f  【%s】" % (score, verdict(score)))
        print("维度: " + ", ".join("%s=%.3f" % (k, v) for k, v in sorted(dims.items())))
        print("VAD A: V=%.2f A=%.2f D=%.2f | VAD B: V=%.2f A=%.2f D=%.2f" % (va[0], va[1], va[2], vb[0], vb[1], vb[2]))
        print("文本长度: A=%d 字, B=%d 字" % (len(txt_a), len(txt_b)))


if __name__ == "__main__":
    main()
