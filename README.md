# StatKit

A pure-Python **Streamlit** web app that lets a university student (any field) run the
statistical test they already know by name on their **messy Excel/CSV sheet**, and get
back the raw numbers, a **plain-English sentence**, a chart, and a downloadable **Word**
report.

Built as a favor — **no accounts, no payment, no tracking, and no AI/LLM anywhere in the
shipped tool.** That last point is enforced mechanically (see below), not just promised.

## Run it locally
```bash
python3.12 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/streamlit run app.py
```
Then open the local URL Streamlit prints. Upload a sheet, pick your test, read the result,
download the Word report.

## What it does
- Reads `.xlsx` / `.xlsm` / `.xls` / `.csv` from the upload **in memory** (nothing is written
  to disk), and cleans the usual mess: title/summary/subtotal/footer rows, two-row and
  units-row headers, merged cells, hidden rows, decimal-comma / thousands / percent / unit /
  currency number formats, censored/non-detect values, and more.
- Infers each column's type, then offers only the tests your data can actually support
  (the menu greys out the rest and says why).
- Runs the test **correctly by default** where common libraries are not: Welch's t-test by
  default, chi-square with no Yates correction + automatic Fisher routing, OLS (not
  `linregress`), Wilson proportion CIs, hand-written effect sizes, explicit listwise
  deletion with the N reported, and a stated block instead of a silent NaN on bad input.
- Writes a plain-English sentence that never overclaims (no "caused", "proves", etc.),
  a chart, and a Word report.

Full menu: descriptives, normality & equality-of-variance checks, one-sample / Welch /
paired t-tests, one-way / two-way / repeated-measures ANOVA (with auto post-hoc),
Mann-Whitney, Wilcoxon, Kruskal-Wallis, Friedman, Pearson / Spearman / Kendall +
correlation matrix, chi-square (independence & goodness-of-fit), Fisher's exact, McNemar,
Cochran's Q, one- & two-sample proportion tests, and simple / multiple / logistic
regression — 27 menu entries in all.

## The no-AI / no-I/O guarantee
`tests/test_no_ai_no_io.py` is a build gate with two layers. The load-bearing one is an
**import allowlist**: every import under `statkit/` (and `app.py`) must name an explicitly
sanctioned pure-compute library, so a new or unknown AI/LLM library — or any stdlib
network/disk module — fails **by omission** rather than needing to be on a ban-list. On top
of that, a **call-scan** catches the I/O verbs the allowed libraries themselves expose
(pandas readers/writers, `savefig`/`.save`/`load_workbook` with a disk-path argument,
`io.open`, `st.connection`, `__import__`/`exec`/`eval`, `sys.modules`, …) and requires their
file targets to be in-memory buffers. The call-scan is an honest **drift guard for
maintainers, not a sandbox against a hostile contributor**: it stops an accidental
re-introduction of I/O through a shipped library; the security guarantee against deliberate
obfuscation is the import allowlist. `tests/test_zero_streamlit.py` keeps Streamlit out of
the package (only `app.py` uses it). `tests/test_deploy_readiness.py` extends the rule to the deploy artefact: every pin in `requirements.txt` must be reachable from a shipped import, so a dependency nothing imports cannot ship.

The **no-tracking** promise is enforced by `.streamlit/config.toml`, not just prose: it sets
`browser.gatherUsageStats = false` (no telemetry phoned home), caps uploads at 25 MB, and
runs headless. That file is verified by the gate test too.

## Tests
```bash
./.venv/bin/python -m pytest -q      # 1143 tests (2026-09-23)
```
The statistics are checked against published/textbook constants and independent
scipy/statsmodels cross-calls; the messy-sheet reader is checked against a curated corpus
of real-shaped fixtures with authored ground-truth oracles.

## Deploy
Streamlit Community Cloud, Python 3.12, `requirements.txt` only, no secrets. Steps and
platform facts in [DEPLOY.md](DEPLOY.md). Deploying is the owner's separate, manual step.
