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

So your league-mates can use it without installing anything, host it on **Hugging Face
Spaces** (free, no credit card). The Space downloads this app from GitHub and runs it.

1. Make a free account at [huggingface.co](https://huggingface.co/join).
2. Go to [huggingface.co/new-space](https://huggingface.co/new-space) and fill in:
   - **Space name:** for example `trade-calculator`
   - **License:** MIT
   - **Select the Space SDK:** **Docker**, then the **Blank** template
   - **Space hardware:** **CPU basic** (free)
   - **Public** if anyone with the link may use it, **Private** if only you

   Then press **Create Space**.
3. On your new Space, open **Files**, choose **+ Contribute → Create a new file**, name it
   `Dockerfile`, and paste in everything from
   [deploy/huggingface/Dockerfile](deploy/huggingface/Dockerfile) (open it on GitHub, press
   **Raw**, select all, copy). Press **Commit new file to main**.
4. The Space starts building. Watch the **Logs** tab: the first build takes about 5–10
   minutes, because it also learns from four past seasons. When it says **Running**, the
   **App** tab shows the trade calculator. Share the address that looks like
   `https://<your-name>-trade-calculator.hf.space`.

Good to know about the online copy:

- **Updates:** after new changes reach GitHub, open the Space's **Settings** and press
  **Factory rebuild** to pick them up.
- **Sleeping:** after 2 days with no visitors the Space goes to sleep; the next visitor
  wakes it, which takes a few minutes.
- **Other leagues:** the build pre-learns past seasons for this league's scoring. A league
  with different scoring takes 2–3 minutes the first time it's loaded.
- At most two grades or searches run at once; anyone else waits their turn. Adding a
  FantasyPros file only works when you run the app on your own computer.

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
