# CryptoPulse RL — project update spec

Context for the coding agent: this is an existing project (real-time crypto dashboard in React.js/Tailwind CSS, with an LLM-based signal generator and a contextual-bandit execution layer, validated via walk-forward backtesting against buy-and-hold and LLM-only baselines). The tasks below extend it into a LangGraph-based multi-agent pipeline and tighten the RL execution layer and evaluation.

## 1. Multi-agent architecture (LangGraph, supervisor pattern)

Replace the current single-shot signal generation call with a graph of specialist agents fanning into a supervisor node.

**Nodes:**

- `market_data_agent` — fetches OHLCV, order book, volume (Phase 1: via Bright Data; Phase 2: via a dedicated crypto MCP server — see section 3).
- `sentiment_agent` — fetches news/sentiment data (Phase 1: via Bright Data; Phase 2: via a dedicated finance/sentiment MCP server — see section 3).
- `onchain_agent` (optional) — fetches on-chain/DeFi activity data if this feature is added; same phased sourcing as the other data agents.
- `technical_analysis_node` — plain function/tool node (no LLM call) computing indicators (RSI, MACD, realized volatility) from `market_data_agent` output.
- `signal_generator_agent` — LLM node that synthesizes technical + sentiment (+ on-chain, if present) context into a structured trade signal.
- `supervisor_agent` — aggregates outputs, resolves disagreement between technical and sentiment signals, and produces the final signal object passed to execution.

**Requirements:**

- `market_data_agent` and `sentiment_agent` (and `onchain_agent`, if present) should run in parallel (fan-out), not sequentially.
- State should be a typed schema (TypedDict or Pydantic model), not free text, so the supervisor resolves conflicts on structured fields.
- `signal_generator_agent` must use structured output (e.g. `with_structured_output`) with schema: `{ action: BUY | SELL | HOLD, confidence: float, rationale: str, horizon: str }`.
- Add a caching + retry layer around the data agents with a fallback-to-last-known-value path. This covers two distinct risks: Bright Data's monthly request cap in Phase 1 (section 3), and rate limits/reliability of the specialized free MCP servers in Phase 2 — don't let a failed or budget-exhausted call crash the graph.

## 2. Tool wiring — how agents actually get access to data

Listing an MCP server does not give any agent access to it. Each agent must be explicitly wired up:

1. Create an MCP client per server using `langchain-mcp-adapters` (or equivalent).
2. Let the client discover the server's tools and convert them into LangChain `Tool` objects (this happens automatically via the adapter).
3. Bind only the relevant tools to each agent. Do not bind all tools to all agents — scoping tool access per agent also keeps, e.g., the sentiment agent from being able to place trades.
4. Add a `ToolNode` in the graph to actually execute tool calls the LLM decides to make, and route results back into that agent's state.

This wiring must exist before any agent can fetch anything — treat it as a prerequisite step, not an implementation detail to fill in later.

## 3. Data sourcing — phased rollout

**Phase 1 — Bright Data MCP only.** Use the existing Bright Data API key/MCP server as the sole data source for all agents initially:

- `market_data_agent`: use Bright Data's web-unlocker/scraping tools to pull price/OHLCV from a page like CoinGecko or an exchange's public site (structured extraction, not raw HTML parsing where avoidable).
- `sentiment_agent`: use Bright Data's search + scrape tools to pull news/social content directly rather than a dedicated sentiment API.
- `onchain_agent` (if added): scrape an explorer or DeFiLlama's public dashboard through Bright Data rather than a separate on-chain MCP server.
- **Budget constraint**: Bright Data's free MCP tier is capped at 5,000 requests/month, then billed per request. Build the caching/retry/fallback layer (section 1) around this ceiling from day one — don't poll more often than the data actually changes, and cache aggressively across agents so the same underlying fetch isn't repeated per agent per cycle.
- Log every Bright Data call (endpoint/page, timestamp, cost-relevant weight) so usage against the 5,000/month cap is visible, not discovered after the fact.

**Phase 2 — add specialized free MCP servers only where Bright Data falls short.** Once Phase 1 is running, evaluate gaps (e.g. Bright Data scraping proves brittle for a given source, or usage is approaching the monthly cap) and add targeted, purpose-built servers:

- **Market data**: `mcp-server-ccxt` (open-source CCXT wrapper, free, no key) or CoinGecko's community MCP (free Demo tier: ~30 calls/minute, ~10,000 calls/month) as a more structured, cheaper-to-poll alternative to scraping.
- **Sentiment/news**: Alpha Vantage's MCP server (free tier: 25 requests/day, 5/minute — confirmed against Alpha Vantage's own documentation) if scraped sentiment proves noisy.
- **On-chain/DeFi**: DeFiLlama MCP — fully free, no API key, MIT license, effectively no rate limit. The most reliable free option found so far; prefer this over scraping DeFiLlama through Bright Data once wired in.
- **On-chain wallet/tx data**: Etherscan MCP (free tier).
- **Macro context**: FRED MCP (free, unlimited-use key) for rates/inflation as a market-regime feature for the bandit.
- **Cross-asset context**: yfinance MCP (fully free, no auth) for equities correlation (e.g. BTC vs Nasdaq) — note its data carries a ~15-minute delay, fine for backtest/regime features but not for time-sensitive live use.
- Do not add community marketplace MCP listings that bundle multiple data types without first verifying the maintainer and repo — unlike the servers above, several have no verifiable provenance.
- Constraint for Phase 2 servers specifically: free tier or open-source only, no paid upgrades. This constraint does not apply to Bright Data since that's an existing account being used within its free allotment.

## 4. Execution layer — contextual bandit refinement

- Keep the bandit's arms **memoryless**: discrete allocation levels (e.g. 0%, 25%, 50%, 100% position size) chosen fresh each period, with reward computed over a fixed short horizon net of transaction costs/slippage. Do not let the bandit also decide holding duration — that introduces state dependence the bandit assumption doesn't cover.
- Reward should be risk-adjusted (e.g. rolling Sharpe-like term or PnL net of costs), not raw PnL.
- If multi-period holding decisions are wanted later, treat that explicitly as a separate (semi-)MDP component, not folded into the bandit.

## 5. New baseline — meta-labeling classifier

Add a third comparison arm alongside buy-and-hold and LLM-only:

- Train a supervised classifier (meta-labeling, López de Prado style) that predicts whether to act on the LLM signal, using the same features available to the bandit.
- This baseline has no live-exploration risk and gives a clean answer to "why RL over supervised learning" — include it in the same backtest suite.

## 6. Backtesting / evaluation

**Backtesting does not reuse the live Bright Data/MCP pipeline.** The live agents (section 3) are for real-time inference only. Backtesting needs a separate offline path:

1. **Build a historical dataset once, outside the live graph.** Pull historical OHLCV via CCXT's REST endpoints (free, built for historical candles — better suited to this than scraping a live page). For historical news/sentiment, this is a real open question, not a solved detail — options: (a) a one-time bulk historical scrape via Bright Data of an archived news source, spent deliberately against the 5,000/month budget rather than continuously, or (b) a free historical news/sentiment archive if one is identified during Phase 2 evaluation (e.g. Alpha Vantage's news-sentiment endpoint has historical coverage on the free tier, within its daily cap). Decide and document which approach is used before building the rest of the backtest pipeline — don't leave this implicit. Store everything locally (parquet/CSV/SQLite) indexed by timestamp, as a standalone data-collection job, not something invoked per backtest run.
2. **Every stored row needs a point-in-time "as-of" timestamp**, not just a value — use the timestamp of when the information was actually available, not when it was collected, or backtest results will silently leak future information into the past.
3. **Replay the cached dataset through the agent logic offline.** At each historical timestep, feed the signal-generator agent only data that was available as of that timestep, read from the cache — not a live Bright Data/MCP call. The LLM call itself still runs (that's what's under test); the data-fetching agents do not hit any live API during a backtest.
4. **Walk-forward windows**: for each sequential window, fit the bandit and the meta-labeling baseline using only data before that window starts, evaluate forward through the window, then roll forward. Nothing evaluated in a window should have informed that window's training step.
5. **Budget LLM calls for the backtest itself** — cache identical (timestep, context) → signal outputs, and/or backtest at a coarser frequency (daily/hourly) if minute-level would produce an unmanageable number of LLM calls over a multi-month history.

**Metrics and checks:**

- Report risk-adjusted metrics: Sharpe ratio, max drawdown, win rate — not just cumulative PnL.
- Explicitly check for and document: look-ahead bias (see point 2 above), realistic transaction costs/slippage assumptions, and whether the test windows span different volatility regimes.
- Produce a results table comparing: buy-and-hold, LLM-only, meta-labeling baseline, contextual-bandit execution — across all test windows.

## 7. Deliverable checklist

- [ ] LangGraph supervisor graph replacing the single-call signal generator
- [ ] MCP clients created and tools bound per-agent (not shared across all agents), with a ToolNode executing calls
- [ ] Phase 1: all agents fetching via Bright Data MCP, with usage logging against the 5,000/month cap
- [ ] Caching/retry/fallback layer in place before Phase 1 goes live
- [ ] Structured output schema enforced on signal generator
- [ ] Bandit arms redefined as memoryless discrete allocations
- [ ] Reward function switched to risk-adjusted, cost-aware
- [ ] Phase 2 (only if Phase 1 gaps identified): specialized free MCP servers added for weak spots (CCXT/CoinGecko, DeFiLlama, FRED, etc.)
- [ ] Meta-labeling baseline implemented and included in backtests
- [ ] Historical news/sentiment sourcing method decided and documented (section 6, point 1 — not left implicit)
- [ ] Historical dataset built and cached offline (CCXT historical OHLCV + cached news/sentiment, point-in-time timestamped) — separate from the live Bright Data pipeline
- [ ] Walk-forward backtest report with Sharpe/drawdown/win-rate across all four strategies
