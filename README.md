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
uv run python -m ffcalc serve
```

Your browser opens at http://localhost:8000. If it doesn't, open that address yourself.
To stop the app, go back to Terminal and press Ctrl C. (The app's screens come already
built, so you don't need anything else installed.)

**The very first time** you load a league it downloads four past seasons of data to learn
how much scores swing and how often players get hurt. That takes 2–3 minutes. After
that, it's fast.

## 3. Find your league ID

Paste your Sleeper **league ID** into the box on the welcome screen.

- **On a computer:** open your league at sleeper.com. The web address looks like
  `https://sleeper.com/leagues/1400221038771437568/team`. The long number is your league ID.
- **Only use the phone app?** The easiest way is to log in at sleeper.com once in a
  browser and copy the number from the address bar, or use the username search below.
- **Don't know it?** Type your Sleeper username into the "Don’t know the ID?" box instead.
  It lists your leagues; click yours.

Then pick **your team**. The app remembers your league next time.

## 4. Use it

- **Grade a trade:** choose the other team, tap the players you'd give and get, and press
  *Grade this trade*. *Quick* takes about 5 seconds; *Full* is more precise and takes
  about 15. You get:
  - a verdict: take it, pass, or close to even;
  - how your title and playoff odds change, and theirs;
  - a chart of why: starters, depth, waiver wire, schedule and byes, injury risk,
    playoff weeks;
  - week-by-week scores, a market-value check, and any pickups or cuts the trade forces;
  - if it isn't a win for both teams, a **Make it fair** button that tries adding or
    removing one player (about half a minute) and shows versions both teams win.
- **Your league:** everyone's playoff and title odds today, and trades already made in
  your league.
- **Find trades:** searches every roster for deals that raise both teams' title odds.
  It takes a minute or two. Click any result to grade it in detail.
- **How accurate is it?:** the accuracy report, measured on real past seasons.

Optional: if you want FantasyPros projections in the mix, download a weekly projections
CSV from fantasypros.com and add it on the *Your league* page. The app uses it until you
stop the app.

On a phone or another computer? The app only runs on the computer you started it on.

## Put it online for free (optional)

So your league-mates can use it without installing anything, host it on **Vercel**'s free
Hobby plan (no credit card). Vercel copies this app from your GitHub repo and runs it.

1. Go to [vercel.com/signup](https://vercel.com/signup), choose **Hobby**, and sign up with
   **Continue with GitHub** (the account that owns this repo).
2. On your Vercel dashboard, press **Add New…** → **Project**. Find
   `fantasy-trade-calculator` under **Import Git Repository** and press **Import**. If it
   isn't listed, press **Adjust GitHub App Permissions** and give Vercel access to it.
3. Leave every setting as it is (Vercel reads `vercel.json` and `pyproject.toml` from the
   repo) and press **Deploy**.
4. The first build takes about 5 minutes, because it also learns from four past seasons.
   When it shows **Congratulations**, press **Continue to Dashboard**, then **Visit**.
   Share that address; it looks like `https://fantasy-trade-calculator.vercel.app`.

Good to know about the online copy:

- **Updates:** every push to GitHub's `main` branch redeploys it by itself.
- **Monthly allowance:** the free plan includes 4 hours of computing a month, about 800
  trade grades or 190 "Find trades" searches. If it runs out, the site pauses until the
  next month. You're never charged.
- **Waking up:** after a quiet stretch the first visit takes a few extra seconds while
  Vercel starts a fresh copy.
- **Other leagues:** the build pre-learns past seasons for this league's scoring. A league
  with different scoring takes 2–3 minutes the first time a fresh copy loads it.
- Each copy runs one grade or search at a time; anyone else waits their turn. Adding a
  FantasyPros file only works when you run the app on your own computer.
- Any host that runs Docker (for example Hugging Face Spaces on its paid PRO plan) can use
  [deploy/huggingface/Dockerfile](deploy/huggingface/Dockerfile) instead.

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

Changing the app's screens (the `web/` folder) needs Node.js; most people never do this.
After an edit, rebuild them:

```bash
cd web && npm install && npm run build
```

## Good to know

- Sleeper leagues only (no ESPN or Yahoo leagues yet), season-long redraft. Keeper,
  dynasty, draft picks and IDP slots aren't modeled.
- The "chance they say yes" number is a rough rule of thumb, not a measurement.
- ESPN and Sleeper projections come from unofficial endpoints. If one stops answering,
  the app keeps working with the sources it can still reach, using cached data where it has it.
- How it works: `spec.md`. Accuracy: `REPORT.md`.
