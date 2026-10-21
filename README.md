# Fantasy Football Trade Calculator

Grades a trade in your Sleeper league by simulating the rest of the season 10,000
times with and without it. See `spec.md` for how it works.

## Setup (once)

1. Install uv (the tool that manages Python for this project):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. In this folder, install everything:
   ```bash
   uv sync
   ```

## Run

```bash
uv run streamlit run app.py
```
A browser tab opens. Paste your Sleeper league ID (the number in your league's URL
on sleeper.com).

## Check it works

```bash
uv run pytest
```
