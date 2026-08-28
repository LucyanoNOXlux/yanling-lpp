---
name: lpp-vad
description: VAD 情绪剖面分析——从文本中提取效价（Valence）、唤醒（Arousal）、支配（Dominance）三维情绪坐标，输出最近邻情绪词精确标签和 D 一致性检查。可用作 LPP 全流程的 D9 增强层，也可独立使用。触发词：VAD、情绪剖面、情绪坐标、情绪量化、VAD分析、效价唤醒支配。
version: 1.0
---

# lpp-vad — VAD 情绪剖面

> LPP 言灵包子技能 · D9 增强层

## 概述

对文本做词级 VAD（Valence-Arousal-Dominance）三维情绪量化，基于 NRC VAD Lexicon v1.0。中英文自动检测。

## 工具调用

```bash
python _Skills/dream-iea/scripts/vad_lookup.py -i input.txt --json
```

JSON 输出包含：`vad_profile`（V_μ/A_μ/D_μ）、`nearest_words`（最近邻情绪词 Top10）、`d_consistency`（D 一致性检查）。

## 信号解读

| VAD 信号 | 解读 |
|----------|------|
| V_μ < 0.4 | 整体负面情绪基调 |
| V_μ > 0.6 | 整体正面情绪基调 |
| A_μ > 0.6 | 高唤醒——激动/活跃 |
| A_μ < 0.35 | 低唤醒——平静/抑制 |
| V_σ 高 | 情绪波动大，情境敏感 |
| V_σ 低 | 情绪稳定，基调一致 |

**D 一致性检查**：对比 VAD-D（词级支配度）和语言模式-D（D8/D15/D17/D21）——一致说明人格-情绪统一，背离说明可能的自我呈现与内在体验不一致。

## 最近邻情绪词

不使用 6 类情绪分桶，直接在 ~110 个情绪描述词的 VAD 空间中找最近邻，输出精确情绪标签（如 `ambivalent` 矛盾、`determined` 坚定）。

## 段落级 VAD 轨迹（≥600字）

按段落切分 V_μ/A_μ/D_μ，观察时间序列：

| 轨迹模式 | 信号 |
|---------|------|
| V↑ D↓ | 表面好转但掌控感流失 |
| V↓ A↑ | 效价下降+高唤醒（愤怒/焦虑） |
| V平 A↓ D↓ | 情绪"熄灭"——可能解离 |
| V波动 D稳定 | 情绪起伏但掌控不受影响 |

## 置信度

≥600字 + 命中率 ≥5% → ● 中；命中率 3-5% → ▲ 弱；<3% → △ 探索；<600字 → 不激活。

## 依赖

- `_Skills/dream-iea/scripts/vad_lookup.py`（需 jieba）
- `_Skills/dream-iea/data/NRC-VAD-Chinese-Simplified.txt`
- `_Skills/dream-iea/data/emotion_words_en.txt`

完整理论见 `../references/vad-d9plus.md`。
