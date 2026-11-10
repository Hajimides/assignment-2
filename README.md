# 《活着》句子情感分类实验

CHI3242 Assignment 2 的代码、数据、报告和实验结果。**已完成 60 条训练标签及 40 条独立文本的本人抽样核查，报告已包含在 [`report.md`](report.md) 和 [`report.pdf`](report.pdf) 中。**

作业要求：[Assignment 2: Train Your Own Text Classifier](https://mcjkurz.github.io/teaching/courses/2026-2027/CHI3242/assignments/assignment-2/)。截止时间为 **2026 年 11 月 11 日上午 9:00**。

## 研究问题和分析单位

从其他中文小说中选取具有明确情绪的句子，训练正面／负面二元逻辑回归分类器，再将它应用于余华《活着》，观察模型预测随文本位置的变化，并检查其错分与不确定判断。

分析单位统一为**句子**，不是整部作品、人物道德或事件好坏。分句规则：以 `。！？!?`、省略号或原始换行为界，句末闭引号归入前句；逗号、分号、冒号不单独断句。原始换行作为边界，因此单独一行的对白也可能是短句。训练候选保留 12–90 个字符的句子，便于在有限语境下可靠标注；独立文本保留所有含汉字的正文句子，并标记短句。这种长度差异属于适用范围限制。

| 标签 | 操作定义 | 边界处理 |
| --- | --- | --- |
| `positive`（正面） | 句内主导方向是愉快、喜爱、感激、欣慰、满足、积极期待，或明确的宽慰与情绪改善 | 对行为的道德评价不决定标签；愿望须有积极情绪线索，普通请求和纯目标不自动算正面 |
| `negative`（负面） | 句内主导方向是悲伤、痛苦、恐惧、忧虑、愤怒、厌恶、失望、孤独或羞惭 | 注意否定与转折；勉强笑、希望落空不能只因有“笑”“希望”判正面 |
| `exclude`（仅用于训练资料筛选） | 中性或客观叙述、没有明确方向的疑问／假设、情绪混杂、反讽难判、需更多上下文、文本残缺／污染 | 不进入二元模型训练与测试；它不是模型的第三类 |

哭泣通常作为悲伤线索，但若句内明确说感激或哭后舒畅，按主导方向判断；无法确定则排除。情绪改善明确时按转折后的方向判断，无法分出主导方向时排除。标签本身也可能存在争议，必须保留原句并核查。

## 资料来源与标注过程

来源：[mcjkurz/qhchina-data](https://github.com/mcjkurz/qhchina-data)，固定版本 `4f4f2b61b560d5daff3d920eb2dfa72a7a47d7b4`。逐文件原始链接和 SHA-256 见 [`data/sources.json`](data/sources.json)。

| 作品 | 正面 | 负面 | 合计 |
| --- | ---: | ---: | ---: |
| 鲁迅《呐喊》 | 30 | 47 | 77 |
| 老舍《骆驼祥子》 | 43 | 45 | 88 |
| 巴金《寒夜》 | 36 | 62 | 98 |
| 沈从文《边城》 | 42 | 42 | 84 |
| 钱钟书《围城》 | 35 | 44 | 79 |
| 茅盾《子夜》 | 37 | 46 | 83 |
| **合计** | **223** | **286** | **509** |

准备过程如下：

1. 从六部作品按相同规则分句，以较宽的情绪词作为检索线索；候选抽样种子为 `3242`。检索命中不等于标签。
2. 初次取每部作品、每种检索方向至多 50 句；AI 阅读候选后，明确正面句不足 200，因此在**切分和训练之前**为每部作品补取至多 25 条正面线索候选，再逐句判断。
3. 最终 732 条候选均有明确的 AI 标注决定：509 条进入二元数据，223 条排除。没有生成虚构训练句子；保留原句、作品、原文行号、字符偏移、标注方法和理由。
4. 标注决定见 [`data/annotation_decisions.csv`](data/annotation_decisions.csv)；候选见 [`data/candidates.csv`](data/candidates.csv)；训练资料见 [`data/labeled_sentences.csv`](data/labeled_sentences.csv)。`scripts/record_annotations.py` 是 AI 决定的可执行记录，**不是**关键词自动打标签器。
5. 学生本人已检查 [`data/human_review.csv`](data/human_review.csv)：每部作品每类各 5 条，共 60 条，约占当前标签的 11.8%。60 条均与 AI 初标一致，抽查一致率为 100%；这些条目的 `annotation_method` 已更新为 `human_reviewed`。这是一致率，不代表全部 509 条标签的正确率为 100%。

**抽样偏差：** 这批数据有意寻找明确情绪，并非六部小说的随机代表性样本。中性与复杂句被排除，测试分数主要衡量对这类情绪明确句子的区分能力。其中 60 条已经学生本人核查，其余仍为 AI 标签，整批数据不是权威金标准。

《活着》完整留作独立分析，未用于标注训练、学习词表、计算 IDF 或调参。本次输入为项目根目录原有的 `余华_活着.txt`，SHA-256 为 `565003d70a3ab75e670ac9f90129f3cf3e4a31d15dd650dcc79dee223f71ee89`。程序只排除首行书名和末尾“已完结”标记，保留其他原文，包括 12 条被规则标记的疑似文本异常句，供核查。

**全文与公开范围：** 原始小说文件不随本项目再发布，也不为上游文学文本附加新的许可。`data/raw/`、原始《活着》及 `output/private/` 被 Git 忽略。公开版 `output/huozhe_predictions.csv` 仅包含句子位置、概率、标签等，不含全文。完整逐句文本输出在本地 `output/private/huozhe_full_predictions.csv`；浏览页可以校验并在本地载入原始 TXT。训练与核查用的有限摘句均保留作品出处。

## 方法与防止数据泄漏

- `train_test_split(test_size=0.2, stratify=labels, random_state=42)`，先切分再拟合：407 条训练、102 条测试。训练集正面 178／负面 229，测试集正面 45／负面 57。
- 主模型：`jieba` 分词（`HMM=False`）→ 词 unigram 和 bigram TF-IDF（`min_df=2`、`max_features=10000`、`sublinear_tf=True`）→ `LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=2000)`。实际词表 1,465 个特征。不删除停用词，保留“不”“没”等否定信息。
- 词表、IDF 和回归参数只从 80% 训练集学习。使用 `Pipeline` 保证对测试和独立文本应用同一变换。参见 [scikit-learn 防止数据泄漏说明](https://scikit-learn.org/1.7/common_pitfalls.html) 和 [LogisticRegression 文档](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.LogisticRegression.html)。
- 主模型和参数在第一次测试评估前固定。另保留多数类基线、200 个常见词词频逻辑回归作对照；没有根据测试集成绩更换主模型。
- 额外诊断：仅在 407 条训练资料内做留一作品交叉验证，每折重新学习特征。见 `output/cross_work_training_cv.csv`。主测试集仍是要求的 80／20 分层随机切分。
- 检查规范化文本重复、训练／测试之间的字符 5-gram Jaccard ≥ 0.8 近重复，以及训练／《活着》的完整句子重合；本次均为 0。该检查不代表完全排除同作品语体相似性。

## 当前实验数值

这些数字用于核对实验输出，不代替报告中的本人分析。若改动标签或代码，应重新运行并以 `output/metrics.json` 为准。

| 模型 | 测试准确率 | 宏平均 F1 |
| --- | ---: | ---: |
| 多数类基线 | 55.88% | 0.3585 |
| 200 个常见词词频 + 逻辑回归 | 76.47% | 0.7588 |
| **TF-IDF + 逻辑回归（主模型）** | **70.59%** | **0.6780** |

主模型混淆矩阵，**行是 AI 参照标签，列是预测标签**：

| | 预测负面 | 预测正面 |
| --- | ---: | ---: |
| 参照负面 | 51 | 6 |
| 参照正面 | 24 | 21 |

| 类别 | Precision | Recall | F1 | 测试条数 |
| --- | ---: | ---: | ---: | ---: |
| 负面 | 0.6800 | 0.8947 | 0.7727 | 57 |
| 正面 | 0.7778 | 0.4667 | 0.5833 | 45 |

准确率 Wilson 95% 区间为 61.13%–78.55%，仅反映有限测试样本的抽样不确定性，不包含标注偏差等系统误差。

《活着》共 3,262 个正文句子：预测正面 554 条、负面 2,708 条；2,019 条（61.89%）的 `P(positive)` 在 `[0.4, 0.6]` 内；52 条没有已知特征。低置信度是核查标记，不是第三个预测标签。概率未经过额外校准；中性／混合句仍会被二元模型强制分到某一类，因此预测负面比例不是小说的真实负面情绪比例。本人核查子集的评估见下文；**《活着》全文上的准确率仍然未知**。

位置曲线按正文字符位置等分 20 个区间，计算区间内逐句正面概率的算术平均。横轴是文本位置，不是故事年代或章节；窗口大小会影响曲线。

## 本人核查后的独立文本评估

40 条按句子顺序等距抽取的句子已全部由学生本人标注：正面 5、负面 15、中性 17、混合 1、不明 2。只有明确正面／负面的 20 条进入二元准确率计算，正确 15 条，准确率为 **75.00%（15/20）**；Wilson 95% 区间约为 53.13%–88.81%。训练集多数类基线（全部预测负面）在这个子集上同样为 75.00%；主模型的正面召回率为 20.00%（1/5），宏平均 F1 为 0.5671。中性／混合／不明的 20 条保留在核查对照表中，不当作正面或负面的参照标签。该结果仅适用于明确正负面的核查子集，不代表全部 40 条或整部小说。

| 本人参照标签 | 预测负面 | 预测正面 | 合计 |
| --- | ---: | ---: | ---: |
| 负面 | 14 | 1 | 15 |
| 正面 | 4 | 1 | 5 |

逐条原句、本人标签、预测和概率见 [`output/analysis_review_predictions.csv`](output/analysis_review_predictions.csv)，其中 5 条二元错分可以按 `included_in_binary_evaluation=1`、`correct=0` 筛选。本人核查仅作评估与训练标签确认，没有用《活着》的本人标签调参或重新训练。训练标签未改变，因此原来的测试指标和全文预测保持不变。

## 运行

建议 Python 3.12。本地已创建 `.venv` 并跑通全部脚本。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/prepare_data.py
.venv/bin/python scripts/run_experiment.py
.venv/bin/python scripts/build_explorer.py
.venv/bin/python scripts/verify_results.py
```

上述第一步数据准备直接使用仓库内冻结的候选和标注，不需要联网。《活着》原文若不在项目根目录，实验脚本会从固定的上游版本下载到被忽略的 `data/raw/`。也可显式指定原始文本：

```bash
.venv/bin/python scripts/run_experiment.py --analysis-text /path/to/余华_活着.txt
```

如需从源文件重新生成候选池（需要联网）：

```bash
.venv/bin/python scripts/prepare_data.py --candidates
.venv/bin/python scripts/record_annotations.py
.venv/bin/python scripts/prepare_data.py
```

完整传递依赖版本记录在 `requirements-lock.txt`，本次实际 Python 和库版本也保存在 `output/metrics.json` 中。

训练后，把保存的同一条 Pipeline 应用到其他本地 UTF-8 文本：

```bash
.venv/bin/python scripts/predict.py /path/to/another.txt
```

## 浏览结果并完成本人核查

直接用浏览器打开 `output/explorer.html` 即可离线浏览；用“载入原始文本”选择《活着》TXT 后可显示原文。页面按 SHA-256 校验文件，处理发生在本地，无需上传。

也可在项目根目录运行：

```bash
.venv/bin/python -m http.server 8765 --bind 127.0.0.1
```

打开 [本地结果浏览页](http://127.0.0.1:8765/output/explorer.html)。本地服务可以自动载入根目录的原始 TXT。

“本人抽样核查”页包含两组资料：

1. **60 条训练标签核查**：填写 `positive`、`negative` 或 `exclude`，可备注歧义；AI 初始标签默认折叠。导出后替换 `data/human_review.csv`，重新运行 `prepare_data.py`，改标会覆盖对应 AI 标签。
2. **40 条独立文本核查**：按句子顺序等距抽取，先独立填 `positive`、`negative`、`neutral`、`mixed` 或 `unclear`。导出后替换 `data/analysis_human_review.csv`。重新运行实验后，只有明确正／负面的已核查句参与独立文本核查子集的准确率计算；该值不能代表全文。

浏览器中的选择只是本地暂存；必须导出并替换文件，才进入实验记录。程序不会覆盖已经存在的核查文件。标注训练数据后也应重新检查误差和结论；反复根据已见测试错误修改标签可能使测试估计偏乐观，修改应遵循原有标注规则并记录理由。

## 文件导航

| 文件 | 用途 |
| --- | --- |
| `report.md`、`report.pdf` | 报告正文与排版后的提交版本 |
| `data/labeled_sentences.csv` | 一行一条文本与标签，附出处和标注过程 |
| `data/candidates.csv`、`data/annotation_decisions.csv` | 候选池、保留／排除决定，可追溯筛选 |
| `data/human_review.csv`、`data/analysis_human_review.csv` | 已填写的本人核查：60 条训练句、40 条独立文本句 |
| `scripts/prepare_data.py` | 下载、分句、候选抽样、应用核查改标 |
| `scripts/run_experiment.py` | 切分、特征、训练、评估、独立文本预测与绘图 |
| `scripts/predict.py` | 将同一模型用于另一文本 |
| `scripts/build_explorer.py` | 生成无外部依赖的本地浏览页 |
| `scripts/verify_results.py` | 检查切分、分数、概率、出处及复现一致性 |
| `output/metrics.json` | 所有分类指标、参数、数据校验和、运行版本 |
| `output/train_split.csv`、`output/test_split.csv` | 固定切分，可检查条目和类别比例 |
| `output/test_predictions.csv`、`output/misclassified.csv` | 测试预测、30 条错分及局部特征贡献 |
| `output/model_comparison.csv`、`output/cross_work_training_cv.csv` | 模型对照、训练集内跨作品诊断 |
| `output/huozhe_predictions.csv`、`output/analysis_summary.json` | 不含全文的独立文本逐句结果与数值汇总 |
| `output/analysis_review_predictions.csv` | 40 条本人核查与预测对照，标记二元评估范围及错分 |
| `output/huozhe_position_bins.csv` | 20 个文本位置区间的曲线数据 |
| `output/feature_weights.csv` | 全部特征系数，正值指向正面 |
| `output/confusion_matrix.png`、`output/huozhe_sentiment_curve.png` | 可用于本人报告排版的图片 |
| `output/explorer.html` | 原文检索、预测筛选、错分、词权重与本人核查 |

AI 协助范围：资料候选筛选、初始标签、代码、实验执行、技术说明、浏览页，以及报告文字的压缩与润色。**本人核查及报告中的阅读判断由学生本人完成并确认。** 仓库中不填写姓名或学号；报告 Markdown 与 PDF 均已加入，按课程要求在 Moodle 提交公开仓库 URL 和 PDF。
