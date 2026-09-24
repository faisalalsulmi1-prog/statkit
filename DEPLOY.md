# Deploy checklist — Chunk 16 (Faisal's hands + separate explicit go)

Everything under "FAISAL'S HANDS" is an external action that needs his own accounts and
his word. Nothing there has been done. The tool runs locally today
(`./.venv/bin/streamlit run app.py`). Host: Streamlit Community Cloud (L7).

## Preconditions — verified locally, enforced by tests
- `requirements.txt` = 52 exact pins from a real `pip freeze` of the py3.12 build venv
  (pandas 3.0.6 / numpy 2.5.3 / scipy 1.18.1 / statsmodels 0.15.0 / matplotlib 3.11.2 /
  python-docx 1.2.0 / openpyxl 3.1.5 / xlrd 2.0.2 / streamlit 1.64.0 + their transitive
  deps). `tests/test_deploy_readiness.py` proves: every pin is reachable from a shipped
  import (no AI/dev package can be smuggled in), every pin == the installed version, and
  the runtime file carries no pytest. No AI libs, no fpdf/reportlab/bidi/reshaper.
- **Python 3.12 is the floor and the tested version**: numpy 2.5.3 and scipy 1.18.1
  declare `Requires-Python >= 3.12`; 3.13+ is untested. Community Cloud has NO file-based
  Python pin (`runtime.txt` / `.python-version` are ignored) — the version is the
  "Advanced settings" dropdown in the deploy dialog (default 3.12 today) and can later be
  changed under the app's settings ("Upgrade Python").
- Dependency-file precedence on Cloud is `uv.lock` > `Pipfile` > `environment.yml` >
  **`requirements.txt`** > `pyproject.toml`. Our `pyproject.toml` is pytest-only and must
  never gain a `[project]` / `[tool.poetry]` table. Cloud installs with `uv`, pip fallback.
- On Linux (CI and Cloud) the resolver adds one unpinned transitive package, `watchdog`
  (a Streamlit requirement with a `platform_system != "Darwin"` marker) — expected.
- `.streamlit/config.toml` is committed and honored by Cloud: telemetry off, uploads
  capped at 25 MB, headless. No secrets, no `packages.txt` (all pins ship Linux wheels).
- `.gitignore` excludes `.venv/`, `.venv-gen/`, `__pycache__/`, `.pytest_cache/`,
  `.DS_Store`, `.streamlit/secrets.toml`, `tests/fixtures/real/`. Commit set ≈ 190
  files / < 2 MB.
- `scipy.stats.studentized_range` present (Games-Howell) — confirmed.
- The app runs headlessly through 6 end-to-end AppTest scenarios; full suite
  **1152 tests, 0 failed (2026-09-24)**; both gates (no-AI/no-I/O allowlist,
  zero-streamlit package) and the readiness gate green.
- CI (`.github/workflows/ci.yml`) runs the same suite on ubuntu / Python 3.12 — the first
  green run after the push is the proof that the pins resolve on Linux.

## FAISAL'S HANDS — held for your word (do in this order)
0. Decide: public or private repo (private needs an extra Streamlit authorization,
   step 6; free-tier private apps are limited). Decide whether STATE.md / PLAN.md /
   docs/ (the build log) go in the repo; if yes, refresh the STATE.md top banner first.
1. Optional but recommended (needs network): prove the pins resolve from PyPI in a fresh
   venv:
   ```bash
   /opt/homebrew/bin/python3.12 -m venv /tmp/sk-dry && /tmp/sk-dry/bin/pip install -r requirements.txt \
     && cd ~/statkit && /tmp/sk-dry/bin/streamlit run app.py
   ```
2. Git identity is NOT set on this Mac:
   ```bash
   git config --global user.name "Your Name"; git config --global user.email "you@example.com"
   ```
3. Make it a repo (deliberately not one yet) on branch `main`, and check nothing
   ignored leaks in:
   ```bash
   cd ~/statkit && git init -b main && git add -A && git status --short | grep -E '\.venv|__pycache__|\.pytest_cache|\.DS_Store' ; echo "(blank above = clean)"
   git status --short | wc -l        # ≈ 190
   git commit -m "StatKit v1"
   ```
4. Create the GitHub repo and push (`gh auth login` first if needed):
   ```bash
   gh repo create statkit --public --source=. --remote=origin --push     # or --private
   ```
5. Wait for CI to go green (`gh run watch`, or the Actions tab). Red = stop; the pins or
   the suite do not hold on Linux — report before touching Cloud.
6. https://share.streamlit.io -> Continue with GitHub. Private repo only: Settings ->
   Linked accounts -> Source control -> "Connect here" -> Authorize.
7. Create app -> "Yup, I have an app" -> repository `<you>/statkit`, branch `main`, main
   file path `app.py` -> App URL: choose a custom subdomain (e.g. `statkit`) ->
   **Advanced settings -> Python 3.12** -> Secrets: leave empty -> Deploy.
8. Smoke test the live URL: upload `tests/fixtures/generated/gen_h1_3.xlsx` (or any
   sheet), pick a test, read the sentence, download the Word report and open it — its
   last line must read "Computed in Python 3.12 with pandas 3.0.6 ...".
9. Record the live URL in STATE.md. Chunk 16 DoD met.

## After it's live (platform facts, 2026)
- Apps sleep after 12 h without a viewer; anyone wakes it with one click (~1 min).
- Logs: "Manage app" (lower-right of the running app). Reboot/delete: workspace menu.
- Every push to `main` redeploys almost immediately; a change to `requirements.txt`
  triggers a full rebuild — keep it deliberate.
- Resources: roughly 0.7–2.7 GB RAM, <= 2 CPU. Fine for 25 MB student sheets.
- Streamlit's own telemetry is off by config; the Cloud workspace still shows the app
  owner basic viewer counts — platform-level, unrelated to any student data, which never
  leaves memory.
- If Python 3.12 ever leaves security support, Cloud force-upgrades the app; re-verify
  the pins on the new version before that happens.
