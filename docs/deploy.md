# Running and hosting the app

The draft board is a Streamlit app (`src/app/draft_board.py`). This page covers where it can run and the one
constraint on hosting it. Data rights: see [`DATA.md`](../DATA.md).

## Local (recommended)

```bash
./dev setup && ./dev app          # or: pip install -e . && nba-fantasy app
```

Without data the sidebar's "Use synthetic demo data" option works out of the box. With data, run
`./dev bootstrap` first; data lives under `NBA_DATA_DIR` (default `~/dev-data/nba-fantasy-2026`), never in the repo.

## Dev container

`.devcontainer/devcontainer.json` gives a Python 3.12 container with dependencies installed and port 8501
forwarded. Data is written to `/workspaces/data` inside the container, so run `nba-fantasy bootstrap` there or
skip it and use the demo mode.

## Hosting it publicly

Publishing the **app** is a different act from publishing the **data**, but a hosted app with real
projections still shows player-level output derived from NBA.com, ESPN, FantasyPros and similar sources,
which their terms restrict (see [`DATA.md`](../DATA.md)). So:

- A public deployment should run **synthetic demo data only**, or be private (auth-gated, personal use).
- Never bake a built dataset, board CSV, cookies or `.env` into an image or hosting bundle.
- Streamlit Community Cloud and similar hosts build from the repo, so they get the demo mode only; that is the
  safe default.
