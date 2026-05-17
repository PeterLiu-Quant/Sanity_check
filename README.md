# ROP Sanity Check — 快速使用指南

这是 Retail Option Pressure (ROP) 论文项目的**最小可行实证检验**。目的是用纯公开数据，在 2-3 天内回答一个问题：

> **零售期权流是否在 cross-section 上预测股票收益？**

如果信号显著，就值得投入接下来 12 周写完整论文。如果完全不显著，需要先调整假设再投入。

---

## 1. 环境准备（10 分钟）

### Python 版本
推荐 Python 3.10 或 3.11。建议用 conda 或 venv 建立独立环境。

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate          # Mac/Linux
# 或:  venv\Scripts\activate      # Windows
```

### 安装依赖
```bash
pip install pandas numpy yfinance statsmodels pyarrow
```

如果你打算在 Jupyter 里逐 cell 跑：
```bash
pip install jupyter ipykernel
```

---

## 2. 下载 SLIM 数据（一次性，5 分钟）

1. 打开 https://sabryzgalova.com/data-and-code/
2. 找到 "Retail Trading in Options and the Rise of the Three Big Wholesalers"
3. 点 **Full replication package** 的 Dropbox 链接
4. 下载并解压
5. 在解压后的文件夹里找包含 daily stock-level SLIM 聚合数据的文件
6. 把这个文件改名为 `slim_data.csv`（或保持原名也行）放到项目的 `data/raw/` 文件夹下

**注意**：BPS 的 replication package 里可能有好几个文件——主程序、分钟级数据、ticker-day 聚合等。你要的是 **ticker-day 级别的聚合文件**。如果不确定，先把所有 .csv 都放进来，运行 Cell 1 看哪个的列名匹配。

---

## 3. 检查并调整列名（关键步骤，5 分钟）

打开 `sanity_check.py`，跑到 Cell 1 结束。它会打印 SLIM 数据的实际列名，类似：

```
columns: ['symbol', 'trade_date', 'slim_call_vol', 'slim_put_vol', ...]
```

然后在 Cell 1 的 `COLUMN_MAP` 字典里**编辑右边的字符串**让它匹配实际列名。例如，如果 BPS 文件里 ticker 字段叫 `symbol`，把：

```python
"ticker": "ticker",
```

改成：

```python
"ticker": "symbol",
```

**这一步很重要——如果列名对不上，后面所有 cell 都会失败。**

如果 BPS 文件只提供 SLIM Share（比例）而没有原始 volume，需要看 Cell 2 的注释说明 fallback 方案。

---

## 4. 运行（30 分钟到 2 小时，取决于 ticker 数量）

### 方式 A：一次性跑完
```bash
python sanity_check.py
```

### 方式 B：在 VS Code 里逐 cell 跑（推荐第一次）
打开文件，安装 Python 扩展，每个 `# %%` 标记上方会出现 "Run Cell" 按钮，逐个点击执行。

### 主要时间花在哪里
- Cell 1-2（加载和构造）：< 1 分钟
- **Cell 3（Yahoo Finance 下载）：最慢，10-90 分钟**——取决于 SLIM 涵盖的 ticker 数量。Yahoo 有 rate limit，第一次会比较慢。
- Cell 4-8（分析）：< 5 分钟

如果 Cell 3 中途断了，可以加一行 `prices.to_parquet(DATA_RAW / "prices_cache.parquet")` 保存进度，下次跳过重新下载。

---

## 5. 怎么读结果

脚本结束时会打印一段"SANITY CHECK SUMMARY"，包含四个测试结果：

### T1 — Cross-sectional dispersion
ROP 信号的横截面差异够不够大。如果 IQR < 0.05，说明信号本身没什么变化，预测不出东西很正常。这种情况通常意味着你需要更聚焦的样本（比如只看高零售关注度股票）。

### T2 — Fama-MacBeth regression  
**核心指标**：lagged ROP 是否预测下一天 cross-section 收益。判断标准：
- t-statistic > 2.0 且系数为正 → PASS
- t-statistic 介于 1.5-2.0 → 边缘信号，需要更大样本或更精细的信号
- t-statistic < 1.5 或为负 → FAIL，假设需要重新审视

### T3 — Decile portfolio sort
**核心指标**：高 ROP 组做多、低 ROP 组做空，看每天能赚多少 bps。判断标准：
- 日均 ≥ 3 bps 且年化 Sharpe ≥ 0.8 → PASS
- 日均 1-3 bps → 边缘信号
- 日均 < 1 bps 或为负 → FAIL

### T4 — Reversal horizon
**机制识别指标**：1 天后、3 天后、5 天后、10 天后累积收益的形态。三种可能：
- **强反转**（5 日累积 < 1 日的 50%）→ 支持 H-A（dealer gamma hedging）
- **持续 drift**（5 日累积 > 1 日的 150%）→ 支持 H-B（informed retail）
- **部分反转**（介于两者之间）→ 支持 H-C（attention/lottery mispricing）

T4 本身不是 GO/NO-GO 判断的核心，但**形态会告诉你论文该往哪个方向写**。

---

## 6. 决策树

脚本会自动打印 DECISION：

### GO（T2 和 T3 都通过）
**这是最理想结果。** 立刻开始 12 周计划：
1. 这周内申请 WRDS + OptionMetrics
2. 邮件联系 Bryzgalova / Bogousslavsky 索取更新数据
3. 开始写 introduction 草稿（即使数据还没到，故事可以先讲）

### CONDITIONAL GO（T2 或 T3 之一通过）
信号存在但弱。**不要放弃，但要 sharpen 样本：**
1. 限定 2020-2021 retail boom 子样本
2. 限定 retail-popular tickers（WSB top 200 mentions）
3. 等短期合约切分（要 OPRA tick data）

### NO-GO（两个都不通过）
**先排除技术原因再考虑放弃：**
1. 确认 SLIM 数据 merge 正确（看 panel 的样本量是否合理）
2. 确认样本期覆盖 2020-2021
3. **注意：公开 SLIM 跨所有到期日聚合，可能稀释了短期合约信号**——这种情况下 NO-GO 不代表完整论文不可行，只代表 sanity check 这个低保真度版本看不到信号。还是值得申请 OPRA 数据再试一次。
4. 如果以上都排除还是没信号，**重新审视假设**——也许零售期权流的横截面效应主要不通过个股 underlying 体现（而是通过指数或波动率市场）。

---

## 7. 接下来该干嘛

不管结果如何，**把以下三件事做完再开始下一阶段：**

1. **保存运行日志**：把脚本输出复制到 `running_log.md`，加日期戳。
2. **建立 GitHub private repo**：把整个 `retail_0dte_mvp/` 目录推上去。等论文发表时这就是你的 replication package。
3. **写 200 字的"结果简报"**：用人话把你看到的告诉自己——T2 系数多大、T3 spread 多大、T4 形态像哪种假设。**这 200 字将来会变成 abstract 的草稿**。

---

## 8. 常见问题

**Q: SLIM 数据找不到对应的字段怎么办？**  
A: BPS 的 replication package 里如果只有 SLIM Share 比例数据（没拆 call/put 原始 volume），需要用 SLIM Share 作为信号本身的代理（虽然损失了方向性信息）。这是次优方案。最干净的做法是写邮件给 Bryzgalova 索要 stock-day-call-put 的拆分数据。

**Q: yfinance 下载经常失败？**  
A: 在 Cell 3 的 `yf.download` 里加 `threads=False` 降速；或者用 `pip install yfinance --upgrade` 升级到最新版（旧版有 Yahoo API 变更的问题）。

**Q: T1 的 IQR 很小是为什么？**  
A: 可能是 SLIM 数据已经过 winsorize / normalization 处理过。看 Cell 1 输出的 raw ROP 描述统计——如果 std 已经 < 0.01，说明数据已经被压缩过，要用未处理的原始 volume 重算。

**Q: 跑出来效果不好，还能继续做这个项目吗？**  
A: 看决策树 §6。**记住：sanity check 的 NO-GO 不等于完整论文的 NO-GO**——它只意味着用公开聚合数据看不到效应，但短期合约切分的 OPRA tick data 可能仍然有效应。

---

最后一句：这个脚本会在 2-3 天内给你一个明确的方向信号。**不要犹豫，今天就开始**。
