# Fantasy Football Trade Calculator

Tells you whether a trade in your **Sleeper** league is worth making. It plays out the
rest of your season thousands of times with the trade and without it, using your
league's real rosters, scoring, schedule, playoff format and free agents. Then it shows
how your title odds and playoff odds change, and why.

It also lists trades that would help both you and the other team, so they have a reason
to say yes.

Everything runs on your own computer and uses free data (Sleeper, ESPN, FantasyCalc,
nflverse). Nothing to sign up for.

## 1. Install (once)

You need a Mac or Linux terminal. On a Mac: open **Terminal** (press ⌘ Space, type
"Terminal", press Enter).

Install **uv**, the tool that sets up Python for this project:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Close the Terminal window and open a new one so it picks up uv. Then go to this folder
and install everything. If the folder is on your Desktop:

```bash
cd ~/Desktop/"Fantasy Football Trade Calculator App"
```

```bash
uv sync
```

## 2. Start the app

From this folder:

```bash
uv run streamlit run app.py
```

Your browser opens at http://localhost:8501. If it doesn't, open that address yourself.
To stop the app, go back to Terminal and press Ctrl C.

**The very first time** you load a league it downloads four past seasons of data to learn
how much scores swing and how often players get hurt. That takes 2–3 minutes. After
that, it's fast.

## 3. Find your league ID

Paste your Sleeper **league ID** into the box on the left.

- **On a computer:** open your league at sleeper.com. The web address looks like
  `https://sleeper.com/leagues/1400221038771437568/team`. The long number is your league ID.
- **Only use the phone app?** The easiest way is to log in at sleeper.com once in a
  browser and copy the number from the address bar, or use the username search below.
- **Don't know it?** Open "Don't know your league ID?" in the app's sidebar and type your
  Sleeper username. It lists your leagues and their IDs.

Then pick **your team** from the list below it.

## 4. Use it

- **Grade a trade:** choose the other team, pick the players you'd give and get, and press
  *Grade this trade*. *Quick* takes about 5 seconds; *Full* is more precise and takes
  about 15. You get:
  - a verdict: take it, pass, or close to even;
  - how your title and playoff odds change, and theirs;
  - a chart of why: starters, depth, waiver wire, schedule and byes, injury risk,
    playoff weeks;
  - week-by-week scores, a market-value check, and any pickups or cuts the trade forces.
- **Your league:** everyone's playoff and title odds today, and trades already made in
  your league.
- **Find trades:** searches every roster for deals that raise both teams' title odds.
  It takes about a minute.
- **How accurate is it?:** the accuracy report, measured on real past seasons.

Optional: if you want FantasyPros projections in the mix, download a weekly projections
CSV from fantasypros.com and drop it in on the *Your league* page.

## Command line (optional)

```bash
uv run python -m ffcalc teams --league 1400221038771437568
```

```bash
uv run python -m ffcalc evaluate --league 1400221038771437568 --team 1 --give "Zay Flowers" --get "Josh Downs, Khalil Shakir"
```

```bash
uv run python -m ffcalc find --league 1400221038771437568 --team 1
```

Rebuild the accuracy report (about 5 minutes):

```bash
uv run python -m ffcalc.backtest
```

Run the tests (offline):

```bash
uv run pytest
```

## Good to know

- Sleeper leagues only (no ESPN or Yahoo leagues yet), season-long redraft. Keeper,
  dynasty, draft picks and IDP slots aren't modeled.
- The "chance they say yes" number is a rough rule of thumb, not a measurement.
- ESPN and Sleeper projections come from unofficial endpoints. If one stops answering,
  the app keeps working with the sources it can still reach, using cached data where it has it.
- How it works: `spec.md`. Design decisions and test results: `NOTES.md`. Accuracy: `REPORT.md`.
