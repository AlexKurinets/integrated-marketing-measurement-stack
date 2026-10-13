# Integrated marketing measurement stack

One measurement framework instead of three that disagree. The stack fits a hierarchical Bayesian marketing mix model at the market level, designs and reads geo matched-market tests with a synthetic-control counterfactual, attributes site response to individual TV airings, and reconciles the three under one documented precedence rule. A weekly refresh publishes the results to warehouse tables, a results API and an MCP server, so analysts and an assistant can ask questions and get numbers that come from the model rather than from memory.

This is a reference implementation of the architecture behind a platform I built for a mid-sized direct-to-consumer meal-kit subscription brand. The production build is under NDA, so a few integrations are abstracted behind adapters (the syndicated audience feed and the attribution platform) and the warehouse is Postgres rather than BigQuery. It runs on synthetic data: every market, network, creative, person, phone number and figure in it is made up.

[![ci](https://github.com/AlexKurinets/integrated-marketing-measurement-stack/actions/workflows/ci.yml/badge.svg)](https://github.com/AlexKurinets/integrated-marketing-measurement-stack/actions/workflows/ci.yml)

## Architecture

```
                 +------------------------------+
                 |  Weekly market panel          |      spend, impressions, new subscriptions,
                 |  (Postgres here, BigQuery     |      promo depth, price index, syndicated
                 |   in production)              |      competitor GRPs, spot logs, minute sessions
                 +---------------+--------------+
                                 |
          +----------------------+----------------------+
          v                      v                      v
+-------------------+  +--------------------+  +--------------------+
| Hierarchical      |  | Geo matched-market |  | Spot-level TV      |
| Bayesian MMM      |<-| tests              |  | response           |
| (NumPyro)         |  | matching, power,   |  | baseline + lift    |
| adstock, Hill,    |  | synthetic control, |  | per airing, placebo|
| pooled by market  |  | placebo inference  |  | corrected          |
+---------+---------+  +---------+----------+  +---------+----------+
          |     calibration ^    |                       |
          +----------------------+-----------------------+
                                 v
                 +------------------------------+
                 |  Harmonization layer          |   tests calibrate the MMM
                 |  reconciliation table         |   MMM allocates across channels
                 |  + budget scenarios           |   attribution allocates within
                 +---------------+--------------+
                                 v
        +------------------+-----+------+-------------------+
        |  warehouse tables |  results API |   MCP server     |
        |  (model_runs,     |  (FastAPI)   |   (9 tools)      |
        |   reconciliation) |              |                  |
        +------------------+--------------+-------------------+

  weekly refresh (measurement/pipeline/weekly.py, Airflow in production):
  read finished geo tests -> fit calibrated MMM -> attribute spots -> reconcile -> publish
```

| Layer | Where | What it does |
| --- | --- | --- |
| Warehouse | `sql/` | Weekly media and outcomes by market, spot logs, minute-level sessions, geo tests, model runs and their outputs. Reporting views in `sql/010_marts.sql`. |
| Adapters | `measurement/adapters/` | Protocols for the syndicated data feed and the attribution platform, with synthetic implementations. The production adapters are the part that is left out. |
| MMM | `measurement/mmm/` | NumPyro model with geometric adstock and Hill saturation per channel, market-level coefficients partially pooled around a national mean on the log scale, market intercepts, controls, Fourier seasonality and trend. Completed geo tests enter as likelihood terms. |
| Geo tests | `measurement/geo/` | Matched-market selection, simulation-based power analysis, synthetic control with non-negative weights, placebo inference, and a readout that becomes an MMM calibration target. |
| TV response | `measurement/tv/` | Baseline-and-lift attribution of site sessions to airings, overlap split by impressions, placebo-minute correction, roll-ups by network, daypart and creative. |
| Harmonization | `measurement/harmonize/` | The reconciliation table and the precedence rule, plain-English explanations, and budget scenarios on the fitted response curves. |
| Delivery | `measurement/pipeline/`, `measurement/api/`, `measurement/mcp_server/` | Weekly refresh, FastAPI results API, MCP server. Both servers call one `MeasurementService`. |

## Design decisions

**Market-level hierarchy rather than a national model.** A national weekly series gives you about a hundred observations for a dozen parameters. The market panel gives you thousands, and partial pooling on the log scale lets a small market borrow strength from the rest of the country instead of producing a noisy coefficient of its own. The same panel is what the geo tests need, so one data model feeds both.

**Priors do real work.** An MMM cannot tell a large constant media effect from a large baseline using the panel alone, so two priors are set deliberately and kept in `config/`: a benchmark prior on each channel's conversions per thousand dollars (from past models and tests), and a prior on the share of outcomes that would happen with no paid media (from pre-media history or go-dark tests). Both are visible, both are arguable, and the geo tests then do the rest.

**Geo tests are likelihood terms, not post-hoc adjustments.** For each completed test, the model computes its own implied lift for the tested channel, markets and weeks as the difference between the response under actual spend and under counterfactual spend (the per-capita spend of the untreated markets). That implied lift is observed with the test's standard error, inflated by a configurable factor because a six-week read on one creative mix is not the long-run channel effect. The MMM is then consistent with the tests by construction, and the reconciliation reports how far it still sits from each read.

**Synthetic control with normalized donors.** Each control market is divided by its own pre-period mean before the weights are fitted, so a treated series that is the sum of several mid-sized markets can still be matched by a convex combination of controls, and the pre-period level of the treated series sets the scale. Inference is by placebo: the same procedure is run on random untreated markets, and the spread of their lifts gives the standard error and p-value.

**Power before spend.** `measurement/geo/power.py` replays history with a known lift injected and counts how often the readout detects it. The minimum detectable effect from that table decides whether a proposed test is worth running at the budget on offer.

**Spot-level attribution is corrected for its own bias.** Clipping lift at zero means pure noise looks like a small positive response. The method is run at random clean minutes too, and that expected noise lift is subtracted from every airing, so the roll-ups are not inflated. Individual airings remain noisy; the network, daypart and creative roll-ups are what the readout uses.

**One precedence rule, written down.** Tests calibrate. The MMM allocates across channels. Attribution allocates within a channel. Last-touch counts are reported next to the harmonized figure so the over-crediting of search and the under-crediting of TV are visible rather than hidden. The rule lives in `measurement/harmonize/reconcile.py` and the explanations in `explain_channel` read it back in plain English.

**Servers are thin.** The API and the MCP server both wrap `MeasurementService`, which only reads published tables. The MCP tools are read-only except for two scenario tools that run on the fitted response curves and change nothing.

## Results from the production build

The figures the platform produced for the brand, rounded:

- Weekly refresh in about 45 minutes, replacing a quarterly agency deliverable with a six-week turnaround.
- Calibrated MMM and geo test agreement on linear TV within about 15 percent, from about 40 percent apart.
- Six geo tests designed, powered and read in the first year.
- Blended CAC down about 10 percent over two quarters after moving roughly 15 percent of linear TV budget into CTV and stronger dayparts.
- Five months to build with one data engineer and one analyst on the client side, then eight months of weekly cadence support.

The synthetic footprint here is 40 markets by 104 weeks by default (210 DMAs in production), with the same per-capita media spend.

## Running it

With Docker:

```
docker compose up --build
```

That starts Postgres (set `IMMS_DB_PORT`, `IMMS_API_PORT` or `IMMS_MCP_PORT` if those ports are taken on your machine), seeds 40 invented markets and two years of weekly data (including two completed geo tests and one designed one), runs a weekly refresh with the `quick` sampling profile (a few minutes on CPU), and then serves the API on port 8000 and the MCP server on port 8001.

```
curl http://localhost:8000/runs/latest/reconciliation
curl http://localhost:8000/runs/latest/explain
curl "http://localhost:8000/runs/latest/tv?dimension=daypart"
curl "http://localhost:8000/runs/latest/simulate?from_channel=linear_tv&to_channel=ctv&weekly_amount=5000"
```

Locally:

```
python -m venv .venv && . .venv/bin/activate      # .venv\Scripts\activate on Windows
pip install -e ".[dev]"
docker compose up -d db
export DATABASE_URL=postgresql+psycopg://imms:imms@localhost:5432/imms
imms-seed --markets 40 --weeks 104
imms-refresh --profile quick                       # full profile: 2 chains x 500 warmup x 500 samples
imms-api                                           # http://localhost:8000/docs
MCP_TRANSPORT=streamable-http imms-mcp             # http://localhost:8001/mcp
```

For an MCP client that speaks stdio, point it at `imms-mcp` with `DATABASE_URL` set. The tools are `list_model_runs`, `get_channel_results`, `get_reconciliation`, `explain_channel`, `list_geo_tests`, `get_geo_test`, `get_tv_response`, `simulate_budget_shift` and `propose_reallocation`.

Settings live in `config/`: channels and benchmark priors, the baseline-share prior, sampling profiles and calibration, geo test design defaults, spot attribution windows.

## Testing

```
pytest -q
```

The suite builds one small synthetic panel (20 markets, 60 weeks) and runs one weekly refresh with the `test` sampling profile, then checks the pieces against the data-generating process: the transforms agree between NumPy and JAX, the synthetic control recovers an injected lift and its placebo p-value behaves under the null, power rises with lift, spot attribution recovers the injected response in aggregate and ranks the creatives correctly, the reconciliation applies the precedence rule, the allocator respects its bounds, the MMM's channel contributions sit near the truth with sane sampler diagnostics, and the API and MCP tools return what the service returns. Database round trips run against Postgres when `DATABASE_URL` is set and are skipped otherwise. CI runs the suite against a Postgres service and then brings the Compose stack up and smokes the API and the MCP endpoint.

## What is deliberately left out

- The production adapters for the syndicated audience feed and the attribution platform. The protocols are here; the implementations are not.
- The Airflow DAG and the BigQuery SQL. The refresh is a plain function and the SQL is Postgres so the whole thing runs on a laptop.
- The R notebooks used for CausalImpact cross-checks of early geo readouts. Those were ported into the Python pipeline.
- Dashboards. The reporting views in `sql/010_marts.sql` are what they read from.
- Authentication on the API and the MCP server.

## License

MIT, Alex Kurinets.
