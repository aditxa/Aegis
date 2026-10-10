# Selected public references and prior art

The repository uses these as scientific/prior-art inspiration. It does not redistribute third-party model weights or claim rights over patented inventions.

## Financial NLP / language models

- Araci, **FinBERT: Financial Sentiment Analysis with Pre-trained Language Models** — https://arxiv.org/abs/1908.10063
- ProsusAI/FinBERT model card — https://huggingface.co/ProsusAI/finbert
- `yiyanghkust/finbert-tone` — https://huggingface.co/yiyanghkust/finbert-tone
- `soleimanian/financial-roberta-large-sentiment` — https://huggingface.co/soleimanian/financial-roberta-large-sentiment
- Wu et al., **BloombergGPT: A Large Language Model for Finance** — https://arxiv.org/abs/2303.17564
- FinGPT — https://github.com/AI4Finance-Foundation/FinGPT
- Xie et al., **PIXIU / FinMA**, NeurIPS 2023 Datasets & Benchmarks — https://proceedings.neurips.cc/paper_files/paper/2023/hash/6a386d703b50f1cf1f61ab02a15967bb-Abstract-Datasets_and_Benchmarks.html

## Financial event / relation extraction

- Hamad et al., **FIRE: A Dataset for Financial Relation Extraction**, Findings of NAACL 2024 — https://aclanthology.org/2024.findings-naacl.230/
- FIRE code/data — https://github.com/hmhamad/FIRE
- Zheng et al., **Doc2EDAG: An End-to-End Document-level Framework for Chinese Financial Event Extraction** — https://aclanthology.org/D19-1032/
- TextEE unified event extraction toolkit / evaluation work — https://aclanthology.org/2024.acl-long.602/

## Market stress / systemic monitoring

- Aldasoro, Hördahl, Schrimpf & Zhu, BIS WP 1250, **Predicting financial market stress with machine learning** (2025) — https://www.bis.org/publications/working-paper-1250-predicting-financial-market-stress-machine-learning
- Aquilina et al., BIS WP 1291, **Harnessing artificial intelligence for monitoring financial markets** (2025) — https://www.bis.org/publications/working-paper-1291-harnessing-artificial-intelligence-monitoring-financial-markets
- BIS, **Under pressure: market conditions and stress** — https://www.bis.org/publications/qr-202209/under-pressure-market-conditions-and-stress
- ECB, **Composite Indicator of Systemic Stress (CISS)** — https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp1426.pdf

## Change points / point processes / contagion

- Adams & MacKay, **Bayesian Online Changepoint Detection** — https://arxiv.org/abs/0710.3742
- Bacry, Mastromatteo & Muzy, **Hawkes processes in finance** — https://arxiv.org/abs/1502.04592
- Bentaleb et al., **A Graph Neural Network for Comparing the Propagation of Stock Market Shocks**, IEEE Access (2024) — https://ieeexplore.ieee.org/document/10737336
- Yuan et al., **Dynamic Graph Learning with Static Relations for Credit Risk Assessment**, AAAI 2025 — https://ojs.aaai.org/index.php/AAAI/article/view/33433

## Market microstructure / limit order books

- Zhang, Zohren & Roberts, **DeepLOB: Deep Convolutional Neural Networks for Limit Order Books** — https://arxiv.org/abs/1808.03668
- Berti & Kasneci, **TLOB: A Novel Transformer Model with Dual Attention for Stock Price Trend Prediction with Limit Order Book Data** (2025) — https://arxiv.org/abs/2502.15757

## Tail risk / jumps / validation

- McNeil & Frey, **Estimation of tail-related risk measures for heteroscedastic financial time series: an extreme value approach** — https://doi.org/10.1016/S0927-5398(00)00012-8
- Barndorff-Nielsen & Shephard, bipower variation / jump detection — https://doi.org/10.1111/j.1468-0262.2006.00691.x
- Romano, Patterson & Candès, **Conformalized Quantile Regression** — https://arxiv.org/abs/1905.03222
- Bailey & López de Prado, **The Deflated Sharpe Ratio** — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551

## Adversarial financial NLP

- Turetken & Leippold, **Battle of transformers: Adversarial attacks on financial sentiment models**, Journal of Banking & Finance 188 (2026), 107698 — https://doi.org/10.1016/j.jbankfin.2026.107698

## Public patents / disclosures

- Thomson Reuters, **Methods and systems for predicting market behavior based on news and sentiment analysis**, EP2786335A1 / US11257161B2 — https://patents.google.com/patent/EP2786335A1/en and https://patents.google.com/patent/US11257161B2/en
- Thomson Reuters, **Financial event and relationship extraction**, US10049100B2 — https://patents.google.com/patent/US10049100B2
- Knowledge-graph stock-risk example, CN112734569A — https://patents.google.com/patent/CN112734569A/en

## Public/free data interfaces

- GDELT DOC 2.0 API — https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
- SEC EDGAR APIs — https://www.sec.gov/search-filings/edgar-application-programming-interfaces
- Reddit API — https://www.reddit.com/dev/api/

## Useful public training datasets

- FIRE relation extraction — linked above.
- Financial PhraseBank / FiQA mirrors are commonly available on Kaggle and Hugging Face; check the original dataset licenses before redistributing or training commercial models.

## v0.3 execution, market-condition and volatility-interface references

- Almgren, R. & Chriss, N. (2001), **Optimal Execution of Portfolio Transactions**, *Journal of Risk*. Canonical execution-cost / volatility-risk framing and market-impact trade-off.
- NSE India, **Impact Cost** methodology. Practical order-book definition of liquidity/impact cost relative to the ideal mid price; impact varies with order size and side.
- NSE India, **India VIX computation methodology / white paper**. Uses NIFTY option bid/ask information, near/next expiries and interpolation to represent expected near-term volatility.
- Cboe, **American-Style Options Implied Volatility Calculation Methodology**. Documents IV inversion and interpolation/extrapolation principles including splines in total implied variance and SABR extrapolation.
- Zhang, Z., Zohren, S. & Roberts, S. (2019), **DeepLOB: Deep Convolutional Neural Networks for Limit Order Books**, IEEE TSP / arXiv:1808.03668.
- Berti, L. & Kasneci, G. (2025), **TLOB: A Novel Transformer Model with Dual Attention for Stock Price Trend Prediction with Limit Order Book Data**, arXiv:2502.15757. Particularly important caution: predictive classification quality can deteriorate when realistic spread/transaction-cost definitions are introduced.
- Aldasoro, I., Hördahl, P., Schrimpf, A. & Zhu, S. (2025), BIS Working Paper 1250, **Predicting financial market stress with machine learning**. Supports liquidity/no-arbitrage market-condition indicators, nonlinear ML and tail-focused evaluation.
- Aquilina, M. et al. (2025), BIS Working Paper 1291, **Harnessing artificial intelligence for monitoring financial markets**. Combines >100 market indicators, an RNN stress forecast and targeted news/LLM context rather than relying on NLP alone.
