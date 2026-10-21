# syntax=docker/dockerfile:1
# Hugging Face Space for the fantasy football trade calculator (setup steps: README.md).
# This is the only file the Space needs: it downloads the app from GitHub and serves it on 7860.
FROM python:3.12-slim

RUN pip install --no-cache-dir uv==0.13.4

# Spaces run as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    UV_PYTHON_DOWNLOADS=never \
    FFCALC_DATA=/home/user/data \
    FFCALC_HOSTED=1

# pinned to the latest commit on main, so "Factory rebuild" picks up new pushes
ADD --chown=user https://github.com/Sag-Bhatia/fantasy-trade-calculator.git#main /home/user/app
WORKDIR /home/user/app
RUN uv sync --frozen --no-dev

# learn from past seasons now so the first visit is quick (tables use this league's scoring);
# if a data source is down during the build, the app builds them on the first visit instead
ARG LEAGUE=1400221038771437568
RUN uv run --no-sync python -m ffcalc history --league $LEAGUE || echo "history builds on first visit"

EXPOSE 7860
CMD ["uv", "run", "--no-sync", "python", "-m", "ffcalc", "serve", "--host", "0.0.0.0", "--port", "7860", "--no-browser"]
