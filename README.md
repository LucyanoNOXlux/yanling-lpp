# 言灵 LPP · Linguistic Psychological Pattern

**语言中寄宿着力量，能映照灵魂的痕迹。痕迹不是灵魂本身，但痕迹不会说谎。**

言灵 LPP 是一个从文本语言模式推断心理状态的语言心理分析框架——扫描措辞、节奏、标点、句式等模式，映射到人格分类系统，输出带置信度的心理画像。中文名「言灵」源自日本言语信仰：语言中寄宿着力量。

> ⚠️ **边界**：LPP 是概率性的语言模式分析，不是心理诊断，不能替代正式人格测评（NEO-PI-R / BFI / MMPI 等）。每个结论标注信度，描述倾向，不贴标签。

## 技能包结构

```
yanling-lpp/
├── SKILL.md              # 主技能（v2.6，10 子技能路由 + 脚本化测量）
├── CREATOR.md            # 创作者与设计说明
├── LICENSE               # CC BY-NC-SA 4.0
├── README.md
├── references/           # 理论框架 / D1-D40 观察指南 / 场景光谱 / 报告模板 / VAD 词表
│   ├── lsm-zh.md         # lpp-lsm 中文功能词表（△探索性）
│   └── nd-calibration.md # ND confound 调节量依据（证据等级/默认值）
└── scripts/
    ├── d_scan.py         # D 维度扫描（21 个可计算维度 + 面具厚度指数）
    ├── lsm_compare.py    # lpp-lsm 计算脚本（语言风格匹配度）
    └── vad_lookup.py     # NRC VAD 词级情绪打分（vendored，S5 依赖）
```

## 子技能（10 个）

| 子技能 | 作用 | 类型 |
|--------|------|------|
| `lpp-register` | 语域校准（Field/Tenor/Mode） | 预分析层 |
| `lpp-vad` | VAD 情绪剖面（效价/唤醒/支配） | D9 增强层 |
| `lpp-lcs` / `lpp-glm` / `lpp-geo` | 语言复杂度 / 世代 / 地域 | 探索层 |
| `lpp-gle` / `lpp-pcs` / `lpp-dark` | 性别化语言 / 职业风格 / 暗黑人格 | 探索层 |
| `lpp-cogstyle` | 认知风格方向（理科生/ASD/混合态） | 探索层 |
| **`lpp-lsm`** | **语言风格匹配度（双文本·人际维度）** | 脚本驱动 |

## lpp-lsm：语言风格匹配度（脚本驱动）

LPP 唯一的**双文本**子技能：从两个人文本的风格匹配度推断关系质量。通过 `lsm_compare.py` 计算（jieba 分词 + 中文功能词表 + 复用 NRC VAD 做情绪同频）。

```bash
python scripts/lsm_compare.py -a 文本A.txt -b 文本B.txt        # 双方独立样本
python scripts/lsm_compare.py -d 对话.txt                       # 对话（A:/B: 前缀自动分离）
python scripts/lsm_compare.py -d 对话.txt --segments 5          # 轨迹模式：看 LSM 趋势
```

识别维度：S1 功能词分布 · S2 语气标记（语气词+表情）· S3 句节奏 · S9 消息节奏（聊天粒度+时间戳）· S5 情绪同频（VAD）· S6 内容呼应。分数 0-1：≥0.75 高同频 / 0.55-0.75 中 / <0.55 低。

**研究底本**：Ireland et al. (2011, *Psych Sci*) · Niederhoffer & Pennebaker (2002, *JLSP*) · Stephens, Silbert & Hasson (2010, *PNAS*) · Khaleghzadegan et al. (2023) · Chartrand & Bargh (1999)。⚠️ LSM 是相关指标非因果诊断。

## D 维度脚本化扫描（v2.6）

Step 1 的可计算 D 维度由脚本测量（LLM 只补不可计算维度）——扫描层从"目测"升级为"测量 + 观察"双层：

```bash
python scripts/d_scan.py -i 文本.txt --json    # 21 个 D 维度测量值
python scripts/d_scan.py --multi f1 f2 f3       # 面具厚度指数（跨语境变异度，0-1）
```

覆盖维度：D1 标点 / D2 断句 / D3 段落 / D4 句长 / D6 语气词 / D8 人称 / D9 VAD 情绪 / D10 绝对化 / D11 否定 / D12 因果 / D15 正式度 / D18 时间锚 / D21 节奏 / D22 填充词 / D24 不确定性 / D25 话语标记 / D30 情态三分 / D33 疑问 / D37 动作-状态动词。

## 依赖

- Python 3.12 + jieba（`pip install jieba`）
- 本仓库已 vendored：`vad_lookup.py`（S5 依赖）与 NRC VAD 中文词表（`references/lexicons/`）——开箱即用，无外部服务

## 许可

CC BY-NC-SA 4.0 · Copyright (c) 2026 Luci逻辑喵 (EDENOFLUX)

**作者：Luci逻辑喵** · 小红书/B站/视频号：Luci逻辑喵 · ☕ 支持创作：爱发电 https://ifdian.net/a/EDENOFLUX
