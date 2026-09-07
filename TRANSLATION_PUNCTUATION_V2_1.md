# Translation Punctuation Normalization v2.1

Policy ID: `v2.1_translation_typography_eval_20260905`

[中文](#中文) · [English](#english)

## 中文

### 用途

该规范用于比较面向阅读与翻译的日文漫画 OCR。它只忽略不会改变正文语义、阅读
或翻译结果的排印差异。归一化以完全相同的方式应用于参考文本和模型输出；模型的
原始输出不会被覆盖。

本项目公开报告的 EM 和 CER 均采用本规范，除非表格明确标记为 raw。比较方式为：

```text
reference = normalize_v2_translation_eval(reference_raw)
prediction = normalize_v2_translation_eval(prediction_raw)
EM  = mean(reference == prediction)
CER = sum(Levenshtein(reference, prediction)) / sum(len(reference))
```

`CER` 使用全语料字符加权统计。参考文本为空时应单独报告，不能把分母替换成 1。

### 基础排印归一化

V2.1 首先应用 `manga_semantic_visual_v2`：

- 删除普通空格、全角空格、TAB、CR 和 LF；
- 使用 NFC，折叠全角 ASCII、半角片假名和半角日文标点；
- 把批准的同形或排印等价字符归入共同形式；
- `…`、`⋯` 转为三个中点，`‥` 转为两个中点；
- 日文钩括号 `『』` 归入 `「」`；
- 长音和波浪线保持两个不同字符族；一个字符保持一个，连续两个及以上统一为两个。

完整字符映射由
[`normalization_presets_semantic_visual_v2.yaml`](evaluation/normalization_presets_semantic_visual_v2.yaml)
定义。

### 翻译标点规则

V2.1 在基础规则之后增加三组计数归一化。

#### 感叹号与问号

连续两个及以上的 `!/?` 构成一个 run：

- 同时出现 `!` 和 `?`：分别保留一个，并保留两类字符第一次出现的先后顺序；
- 只出现一种：保留两个；
- 单个 `!` 或 `?` 保持不变。

```text
!!?   -> !?
!!??  -> !?
!??   -> !?
??!   -> ?!
!!!   -> !!
????  -> ??
!     -> !
?     -> ?
```

#### 省略号点

基础归一化后的连续中点 run：

```text
1–2 点  -> 保持原数量
3–6 点  -> 3 点
7 点以上 -> 6 点
```

实现使用 `U+E000` 作为评分键里的内部点 token，从而保证归一化幂等。展示或送入翻译
前使用 `render_translation_normalized()` 把内部 token 还原为 `・`。

#### 长音与波浪线

```text
ー       -> ー
ーー以上  -> ーー
〜       -> 〜
〜〜以上  -> 〜〜
```

两类字符不互相折叠。

### 明确保留的差异

V2.1 不折叠正文字符、假名大小、清浊音、大小写、`I/l/1`、`O/0`，也不允许字符
插入、删除或语言模型改写。`!?` 与 `?!` 保持不同；单个与重复的 `!/?` 保持不同。

### 公开实现

- [`semantic_visual_v2_eval.py`](evaluation/semantic_visual_v2_eval.py)：基础 V2 规则。
- [`semantic_visual_v2_translation_eval.py`](evaluation/semantic_visual_v2_translation_eval.py)：V2.1 翻译标点规则。
- [`normalization_presets_semantic_visual_v2.yaml`](evaluation/normalization_presets_semantic_visual_v2.yaml)：字符映射。
- [`test_translation_punctuation_v2_1.py`](evaluation/test_translation_punctuation_v2_1.py)：固定测试向量和幂等测试。

## English

### Purpose

This policy compares Japanese manga OCR for reading and translation. It ignores
typographic variation that does not change the body text, reading, or translation.
The same function is applied symmetrically to references and predictions. Raw OCR
output remains available.

All public EM/CER results in this project use this policy unless a table is
explicitly marked `raw`. EM is exact match after normalization. CER is corpus-level
character-weighted Levenshtein distance after normalization.

### Translation-oriented punctuation rules

- A mixed run of two or more `!/?` keeps one of each in first-seen order:
  `!!?`, `!!??`, and `!??` become `!?`; `??!` becomes `?!`.
- A repeated run containing only one type keeps exactly two: `!!!` becomes `!!`
  and `????` becomes `??`. A single mark remains unchanged.
- A normalized ellipsis run keeps one or two dots as-is, maps 3–6 dots to three,
  and maps more than six dots to six.
- One prolonged-sound mark or wave line remains one; a run of two or more becomes
  two. Prolonged marks and wave lines remain distinct families.

The base V2 layer performs the documented Unicode, width, kana, quote, line,
ellipsis, unit, and approved glyph-alias mappings. The YAML file is the normative
mapping table. The Python implementation and fixed test vectors are distributed
with this document.

The policy does not rewrite lexical characters or fold small kana, voicing,
case, `I/l/1`, or `O/0`. It keeps `!?` distinct from `?!` and keeps a single
exclamation/question mark distinct from a repeated run.

