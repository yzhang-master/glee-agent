# Self-analysis: live game performance

Generated 2026-08-18T20:24:19+00:00 by `scripts/self_analysis.py` (re-runnable; regenerates this file).

## 0. Data & the blind spot

- db tracks 595 games; the log carries 883 game ids. Completed games analyzed: **651** — 365 captured (platform result payload), 175 reconstructed (our own accept/walk/final-'no' ended the game, payoff computed exactly), 111 approx (persuasion seller games at the final round; total undercounts by <= 1 round's price).
- **Capture blind spot: the platform only sends a result payload when OUR move ends the game.** Games ending on the opponent's move (they accept our offer; the buyer's final round-20 decision in our seller games) produce no result record — that is why the raw db shows zero proposer-seat agreements and zero persuasion seller completions. 40 bargaining and 28 negotiation games went idle right after our offer (opponent most likely accepted — analyzed separately below); 79 games ended with unknown outcome; 85 still in flight.

## 1. Per-family overview

| family | completed analyzed | mean payoff | median payoff | mean norm. payoff | median norm. payoff | no-deal rate |
|---|---|---|---|---|---|---|
| bargaining | 229 | 139561.96 | 4500.00 | 0.459 | 0.462 | 1/229 (0%) |
| negotiation | 190 | 43262.20 | 9.79 | 0.127 | 0.043 | 85/190 (45%) |
| persuasion | 232 | 3581773.97 | 1850.00 | 0.461 | 0.200 | 0/232 (0%) |

Norm. payoff = payoff/pot (bargaining), payoff/my_value (negotiation), payoff/(price*rounds) (persuasion).

### Lowest-payoff configs (mean normalized payoff, ascending, n >= 2)

**bargaining**

| config | n | mean norm. payoff | mean payoff |
|---|---|---|---|
| pot=100, d1=None, d2=0.8, T=None, msg=False, ci=False | 3 | 0.304 | 30.42 |
| pot=10000, d1=0.9, d2=None, T=12, msg=False, ci=False | 2 | 0.305 | 3054.11 |
| pot=10000, d1=0.9, d2=None, T=12, msg=True, ci=False | 2 | 0.307 | 3068.17 |
| pot=10000, d1=0.95, d2=None, T=12, msg=False, ci=False | 3 | 0.309 | 3085.30 |
| pot=10000, d1=0.8, d2=0.8, T=12, msg=False, ci=True | 2 | 0.314 | 3144.20 |
| pot=100, d1=0.95, d2=1.0, T=None, msg=False, ci=True | 2 | 0.318 | 31.80 |

**negotiation**

| config | n | mean norm. payoff | mean payoff |
|---|---|---|---|
| my_value=1000000.0, T=1, msg=False, ci=False | 2 | 0.000 | 0.00 |
| my_value=8000.0, T=10, msg=True, ci=False | 4 | 0.000 | 0.00 |
| my_value=800000.0, T=1, msg=True, ci=False | 2 | 0.000 | 0.00 |
| my_value=15000.0, T=1, msg=False, ci=False | 2 | 0.000 | 0.00 |
| my_value=100.0, T=1, msg=False, ci=False | 4 | 0.000 | 0.00 |
| my_value=1200000.0, T=1, msg=True, ci=False | 3 | 0.000 | 0.00 |

**persuasion**

| config | n | mean norm. payoff | mean payoff |
|---|---|---|---|
| price=100, p=0.333, v=200.0, u=0.0, T=20, msgtype=text | 2 | -0.075 | -150.00 |
| price=100, p=0.5, v=200.0, u=0.0, T=20, msgtype=binary | 2 | -0.050 | -100.00 |
| price=10000, p=0.8, v=12500.0, u=0.0, T=20, msgtype=text | 2 | -0.037 | -7500.00 |
| price=10000, p=0.5, v=12000.0, u=0.0, T=20, msgtype=binary | 3 | -0.033 | -6666.67 |
| price=1000000, p=0.333, v=1200000.0, u=0.0, T=20, msgtype=text | 4 | -0.025 | -500000.00 |
| price=10000, p=0.333, v=12500.0, u=0.0, T=20, msgtype=text | 2 | -0.025 | -5000.00 |

## 2. Bargaining

### Agreed-round distribution (each extra round burns delta)

| round | games | share of agreements |
|---|---|---|
| 1 | 82 | 36% |
| 2 | 76 | 33% |
| 3 | 7 | 3% |
| 4 | 8 | 4% |
| 5 | 9 | 4% |
| 6 | 1 | 0% |
| 7 | 4 | 2% |
| 8 | 23 | 10% |
| 9 | 10 | 4% |
| 10 | 1 | 0% |
| 11 | 5 | 2% |
| 12 | 1 | 0% |
| 14 | 1 | 0% |

- Agreements at round >= 5: **55/228** (24%). Mean agreed round: 3.2.
- Mean pot share LOST to discounting (agreed nominal gain minus realized payoff, as share of pot): **0.036**.

### Share of pot when we agree (realized payoff / pot)

| split | n | mean share | median share |
|---|---|---|---|
| all observed agreements | 228 | 0.461 | 0.464 |
| my proposal accepted (proposer seat) | 0 | n/a | n/a |
| I accepted theirs (responder seat) | 228 | 0.461 | 0.464 |
| vs agent opponents | 109 | 0.462 | 0.454 |
| vs human opponents | 2 | 0.424 | 0.424 |
| vs hidden opponents | 117 | 0.461 | 0.474 |

**Proposer seat via inference:** 40 further games went idle immediately after our offer — the opponent almost certainly accepted it. Mean share we had proposed for ourselves: **0.640** (median 0.660), at mean round 6.4. These are the only games where our own proposals close — and they close late, after the anchor has already been conceded down.

### No-deal games: 1/229

| config | no-deals |
|---|---|
| pot=100, d1=None, d2=0.8, T=None, msg=False, ci=False | 1 |

## 3. Negotiation

| role | n | closed (agreement) | mean payoff | mean payoff/value | median payoff/value |
|---|---|---|---|---|---|
| seller | 68 | 50 (74%) | 53140.28 | 0.150 | 0.115 |
| buyer | 122 | 55 (45%) | 37756.38 | 0.115 | 0.000 |

- **Our-offer-accepted (inferred):** 28 games went idle right after our offer; if accepted, mean payoff/value **0.460** — our proposer-seat deals are far richer than our responder-seat ones.
- Realized vs best opposing offer ever seen in closed deals ((best-possible minus realized)/value): mean **-0.118** over 82 games (negative = we closed better than their best standing offer).
- Our WalkAway fired in **28** turns; 27 games ended walked_away, 58 no_deal.
- Rounds to finish: mean 9.6, median 10, max 99; **11** games ran >= 15 rounds (marathons).
- max_rounds=1 ultimatums: 62 games, closed 22 (35%), mean payoff/value 0.043.

## 4. Persuasion

### Buyer games: 121; NEGATIVE totals: 36 (30%)

| config | payoff | rounds bought |
|---|---|---|
| price=100, p=0.5, v=300.0, u=0.0, T=20, msgtype=text | -400 | ? |
| price=1000000, p=0.5, v=2000000.0, u=0.0, T=20, msgtype=binary | -2000000 | 2 |
| price=1000000, p=0.333, v=2000000.0, u=0.0, T=20, msgtype=binary | -2000000 | ? |
| price=100, p=0.333, v=400.0, u=0.0, T=20, msgtype=binary | -200 | ? |
| price=100, p=0.333, v=200.0, u=0.0, T=20, msgtype=text | -200 | 6 |
| price=10000, p=0.333, v=20000.0, u=0.0, T=20, msgtype=binary | -20000 | 6 |
| price=10000, p=0.8, v=12500.0, u=0.0, T=20, msgtype=text | -15000 | ? |
| price=10000, p=0.5, v=12000.0, u=0.0, T=20, msgtype=binary | -14000 | ? |
| price=10000, p=0.5, v=20000.0, u=0.0, T=20, msgtype=binary | -10000 | 7 |
| price=100, p=0.333, v=200.0, u=0.0, T=20, msgtype=text | -100 | 1 |
| price=10000, p=0.5, v=12000.0, u=0.0, T=20, msgtype=text | -10000 | ? |
| price=10000, p=0.8, v=40000.0, u=0.0, T=20, msgtype=text | -10000 | 1 |
| price=10000, p=0.5, v=12500.0, u=0.0, T=20, msgtype=text | -10000 | 1 |
| price=1000000, p=0.333, v=1200000.0, u=0.0, T=20, msgtype=binary | -1000000 | ? |
| price=100, p=0.333, v=200.0, u=0.0, T=20, msgtype=binary | -100 | 1 |
| price=100, p=0.5, v=200.0, u=0.0, T=20, msgtype=binary | -100 | 1 |
| price=10000, p=0.333, v=12000.0, u=0.0, T=20, msgtype=text | -10000 | 7 |
| price=1000000, p=0.333, v=1200000.0, u=0.0, T=20, msgtype=text | -1000000 | ? |
| price=10000, p=0.333, v=12500.0, u=0.0, T=20, msgtype=text | -10000 | 6 |
| price=1000000, p=0.333, v=1200000.0, u=0.0, T=20, msgtype=binary | -1000000 | ? |
| price=100, p=0.8, v=400.0, u=0.0, T=20, msgtype=text | -100 | ? |
| price=1000000, p=0.333, v=4000000.0, u=0.0, T=20, msgtype=binary | -1000000 | 1 |
| price=10000, p=0.333, v=40000.0, u=0.0, T=20, msgtype=text | -10000 | ? |
| price=1000000, p=0.333, v=1200000.0, u=0.0, T=20, msgtype=text | -1000000 | 1 |
| price=100, p=0.5, v=120.0, u=0.0, T=20, msgtype=binary | -100 | 1 |
| price=10000, p=0.333, v=12000.0, u=0.0, T=20, msgtype=binary | -10000 | ? |
| price=1000000, p=0.333, v=2000000.0, u=0.0, T=20, msgtype=binary | -1000000 | 15 |
| price=100, p=0.5, v=200.0, u=0.0, T=20, msgtype=binary | -100 | 5 |
| price=10000, p=0.5, v=12000.0, u=0.0, T=20, msgtype=text | -8000 | 14 |
| price=100, p=0.5, v=120.0, u=0.0, T=20, msgtype=text | -80 | 2 |
| price=10000, p=0.8, v=12000.0, u=0.0, T=20, msgtype=binary | -4000 | 16 |
| price=10000, p=0.5, v=12000.0, u=0.0, T=20, msgtype=binary | -4000 | 4 |
| price=10000, p=0.8, v=12000.0, u=0.0, T=20, msgtype=binary | -4000 | 10 |
| price=10000, p=0.5, v=12500.0, u=0.0, T=20, msgtype=binary | -2500 | ? |
| price=1000000, p=0.333, v=1250000.0, u=0.0, T=20, msgtype=binary | -250000 | ? |
| price=10000, p=0.5, v=12000.0, u=0.0, T=20, msgtype=binary | -2000 | ? |

- 20/36 negative games are thin-margin (v/price < 1.5); the rest lost on p=1/3 configs where the prior EV of a blind buy is already negative.
- Buyer games ending at exactly 0 (never profitably bought): 21/121. Mean buyer norm. payoff 0.431.

### Seller games: 111 (all payoffs approx: final buyer decision unobserved)

- Sell rate (rounds bought / rounds observed): mean **0.521**, median 0.421 over 111 games. Mean seller norm. payoff 0.495.
- Low-quality sales ('lies' that got bought) occurred in 74 games; after the first one buyers bought in 778/1086 subsequent rounds (72%); **13** games show trust collapse (<20% buying after the lie).

## 5. Payoff by disclosed opponent

| opponent | n | mean norm. payoff | my payoff > theirs |
|---|---|---|---|
| 7aidara_Beta | 2 | -0.050 | 0/1 |
| warren bluffett | 4 | -0.005 | 0/0 |
| Poseidon | 2 | 0.000 | 0/1 |
| Schelling | 3 | 0.000 | 0/3 |
| Morphling-Ablation1 | 2 | 0.028 | 0/2 |
| NegoMind-B | 5 | 0.050 | 0/2 |
| llm-1 | 2 | 0.051 | 0/1 |
| xx | 2 | 0.075 | 0/0 |
| 7aidara_Gamma | 9 | 0.078 | 2/7 |
| llm-2 | 4 | 0.109 | 0/3 |
| piglet | 5 | 0.132 | 0/2 |
| lucas-agent4 | 5 | 0.149 | 0/3 |
| 7aidara_Alpha | 5 | 0.190 | 1/4 |
| one | 2 | 0.190 | 1/2 |
| DSv4 | 7 | 0.192 | 4/7 |
| mosskappa-baseline-v1 | 2 | 0.200 | 0/1 |
| 3 | 2 | 0.205 | 0/1 |
| llm-4 | 6 | 0.225 | 0/0 |
| np | 7 | 0.259 | 0/2 |
| A1 | 6 | 0.277 | 0/3 |
| fifth-one | 3 | 0.320 | 0/2 |
| hobbylab | 8 | 0.329 | 1/5 |
| xl | 6 | 0.339 | 1/4 |
| Ousen | 3 | 0.350 | 0/1 |
| Recursive | 2 | 0.352 | 0/0 |
| third-one | 8 | 0.362 | 0/1 |
| Riboku | 2 | 0.375 | 0/1 |
| Tester1 | 4 | 0.378 | 0/1 |
| Determineeer | 3 | 0.378 | 0/1 |
| fable | 2 | 0.384 | 0/1 |
| first-one | 9 | 0.384 | 2/7 |
| Rubinstein | 9 | 0.389 | 3/6 |
| theta | 5 | 0.411 | 0/5 |
| Alon Portnoy | 2 | 0.424 | 0/2 |
| champion | 23 | 0.441 | 4/5 |
| Irycia | 3 | 0.444 | 1/1 |
| pas-2 | 29 | 0.446 | 13/18 |
| cobbylab | 5 | 0.460 | 1/3 |
| velocity | 2 | 0.460 | 0/0 |
| Ira | 17 | 0.478 | 5/12 |
| gamma | 4 | 0.499 | 1/3 |
| testing agent01 | 4 | 0.500 | 1/2 |
| Iry | 20 | 0.536 | 9/16 |
| chotu | 12 | 0.543 | 4/8 |
| forth-one | 3 | 0.600 | 1/2 |
| Rufus Dufus | 10 | 0.637 | 3/6 |
| dobbylab | 2 | 0.700 | 1/1 |
| based_agent | 5 | 0.722 | 2/3 |
| NegoMind | 4 | 0.740 | 0/0 |
| bobbylab | 4 | 0.787 | 1/1 |
| lucas-agent | 3 | 1.033 | 1/1 |
| second-one | 3 | 1.350 | 1/2 |

(>= 2 completed games; hidden opponents — the majority — carry no name. Low rows = who beats us; high rows = who we farm.)

## 6. Top 5 recommendations (ranked by expected rating impact)

1. **Negotiation: close more deals — 85/190 (45%) of our games end at $0.** The zero pile sits at/below the pool's no-deal mass, capping the family percentile. Two knobs in `config.py`: lower `neg_anchor_markup` (0.9 -> ~0.5) so counterparts engage instead of stonewalling, and lower `neg_beta` (2.5 -> ~1.8) so we reach acceptable territory while the opponent is still at the table; in `families/negotiation.py::decide` relax the accept rule `payoff >= counter_payoff * 0.9` to ~0.75 once round/T > 0.5 — a live positive offer beats a speculative counter that risks the whole game.
2. **Negotiation: reprice max_rounds=1 ultimatums from dataset acceptance curves — only 22/62 closed** (mean payoff/value 0.043). The `max_rounds == 1` branch prices at (anchor+floor)/2 = value*~1.46 as seller, which the field rejects. Use targets.json `neg_accept` buckets to maximize P(accept) * margin; a ~10-20% markup that closes 60% of the time dominates a 46% markup that closes 27%.
3. **Bargaining: our anchors never close — get accepted earlier.** Every captured agreement is us accepting theirs (mean share 0.461); the 40 inferred proposer-seat closes (mean proposed share 0.640) only land at mean round 6.4, after discounting ate the premium, and 55/228 observed agreements land at round >= 5. Drop `barg_anchor_agent` (0.80 -> ~0.68) and `barg_beta` (2.5 -> ~1.5) so our round-1/3 offers are acceptable while the pot is still whole — an accepted 0.62 in round 1 beats an accepted 0.65 in round 6 under any delta < 1.
4. **Persuasion buyer: stop buying into negative-EV configs — 36/121 buyer games ended NEGATIVE.** Losses concentrate in thin-margin (v/price <= 1.25) and p=1/3 configs where only near-perfect seller honesty makes buying profitable; a negative total sits below the pool's entire never-buy mass at 0. Raise `pers_buy_margin` (0.02 -> ~0.10) when (v-price)/price < 0.5, cut `pers_explore_frac` (0.33 -> ~0.15), and add a hard stop in `families/persuasion.py::_buyer_decide`: never buy speculatively once cumulative payoff would go negative.
5. **Bargaining: accept good offers faster under heavy discounting.** Mean pot share lost to delta-decay across agreements is 0.036 of the pot — pure burn. Scale `barg_cont_realism` (0.85) by the joint discount (multiply by min(delta_1, delta_2, 1.0)) and lower `barg_accept_great` (0.65 -> ~0.58) when min(delta) <= 0.9, so the continuation value stops overrating a future round that is worth 10% less by construction.
