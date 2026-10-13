-- Measurement warehouse schema. In production this lived in BigQuery with the same table shapes;
-- the reference build uses Postgres so the whole stack runs locally.

CREATE TABLE IF NOT EXISTS markets (
    market_id        TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    region           TEXT NOT NULL,
    population       INTEGER NOT NULL,
    tv_households    INTEGER NOT NULL
);

-- Weekly spend and impressions by market and channel (long format).
CREATE TABLE IF NOT EXISTS weekly_media (
    market_id        TEXT NOT NULL REFERENCES markets(market_id),
    week_start       DATE NOT NULL,
    channel          TEXT NOT NULL,
    spend            NUMERIC(14, 2) NOT NULL,
    impressions      BIGINT NOT NULL,
    PRIMARY KEY (market_id, week_start, channel)
);

-- Weekly outcomes and controls by market.
CREATE TABLE IF NOT EXISTS weekly_outcomes (
    market_id          TEXT NOT NULL REFERENCES markets(market_id),
    week_start         DATE NOT NULL,
    new_subscriptions  INTEGER NOT NULL,
    promo_depth        NUMERIC(6, 4) NOT NULL,
    competitor_grps    NUMERIC(10, 2) NOT NULL,
    price_index        NUMERIC(8, 4) NOT NULL,
    PRIMARY KEY (market_id, week_start)
);

-- Spot-level TV airings (post-log) for the response model.
CREATE TABLE IF NOT EXISTS tv_airings (
    airing_id        TEXT PRIMARY KEY,
    aired_at         TIMESTAMP NOT NULL,
    network          TEXT NOT NULL,
    daypart          TEXT NOT NULL,
    creative         TEXT NOT NULL,
    impressions      BIGINT NOT NULL,
    cost             NUMERIC(12, 2) NOT NULL
);

-- Minute-level site sessions, the response signal for spot attribution.
CREATE TABLE IF NOT EXISTS minute_sessions (
    minute_ts        TIMESTAMP PRIMARY KEY,
    sessions         INTEGER NOT NULL
);

-- Weekly last-touch conversions by channel from the attribution platform (abstracted behind an adapter).
CREATE TABLE IF NOT EXISTS attribution_weekly (
    week_start               DATE NOT NULL,
    channel                  TEXT NOT NULL,
    last_touch_conversions   INTEGER NOT NULL,
    PRIMARY KEY (week_start, channel)
);

-- Geo matched-market tests: design on the way in, readout on the way out.
CREATE TABLE IF NOT EXISTS geo_tests (
    test_id             TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    channel             TEXT NOT NULL,
    treatment_markets   JSONB NOT NULL,
    control_markets     JSONB NOT NULL,
    pre_start           DATE NOT NULL,
    test_start          DATE NOT NULL,
    test_end            DATE NOT NULL,
    incremental_spend   NUMERIC(14, 2) NOT NULL,
    status              TEXT NOT NULL DEFAULT 'designed',
    readout             JSONB
);

-- One row per weekly refresh.
CREATE TABLE IF NOT EXISTS model_runs (
    run_id         TEXT PRIMARY KEY,
    as_of_week     DATE NOT NULL,
    created_at     TIMESTAMP NOT NULL DEFAULT NOW(),
    profile        TEXT NOT NULL,
    diagnostics    JSONB NOT NULL
);

-- Channel-level MMM output for a run.
CREATE TABLE IF NOT EXISTS channel_results (
    run_id             TEXT NOT NULL REFERENCES model_runs(run_id) ON DELETE CASCADE,
    channel            TEXT NOT NULL,
    spend              NUMERIC(14, 2) NOT NULL,
    incremental_mean   NUMERIC(14, 2) NOT NULL,
    incremental_low    NUMERIC(14, 2) NOT NULL,
    incremental_high   NUMERIC(14, 2) NOT NULL,
    roi_mean           NUMERIC(10, 4) NOT NULL,
    decay              NUMERIC(8, 4) NOT NULL,
    half_sat           NUMERIC(10, 4) NOT NULL,
    slope              NUMERIC(8, 4) NOT NULL,
    beta               NUMERIC(10, 4) NOT NULL,
    spend_scale        NUMERIC(14, 4) NOT NULL,
    PRIMARY KEY (run_id, channel)
);

-- Market-level channel coefficients from the hierarchy.
CREATE TABLE IF NOT EXISTS market_effects (
    run_id        TEXT NOT NULL REFERENCES model_runs(run_id) ON DELETE CASCADE,
    market_id     TEXT NOT NULL,
    channel       TEXT NOT NULL,
    beta_market   NUMERIC(10, 4) NOT NULL,
    PRIMARY KEY (run_id, market_id, channel)
);

-- Harmonized view of MMM, geo tests and attribution for a run.
CREATE TABLE IF NOT EXISTS reconciliation (
    run_id                    TEXT NOT NULL REFERENCES model_runs(run_id) ON DELETE CASCADE,
    channel                   TEXT NOT NULL,
    spend                     NUMERIC(14, 2) NOT NULL,
    mmm_incremental           NUMERIC(14, 2) NOT NULL,
    mmm_low                   NUMERIC(14, 2) NOT NULL,
    mmm_high                  NUMERIC(14, 2) NOT NULL,
    test_incremental          NUMERIC(14, 2),
    test_low                  NUMERIC(14, 2),
    test_high                 NUMERIC(14, 2),
    attributed_conversions    NUMERIC(14, 2) NOT NULL,
    harmonized_incremental    NUMERIC(14, 2) NOT NULL,
    harmonized_source         TEXT NOT NULL,
    harmonized_roi            NUMERIC(10, 4) NOT NULL,
    mmm_vs_test_gap           NUMERIC(8, 4),
    PRIMARY KEY (run_id, channel)
);

-- Spot-level TV response rolled up by network, daypart and creative.
CREATE TABLE IF NOT EXISTS tv_rollup (
    run_id                TEXT NOT NULL REFERENCES model_runs(run_id) ON DELETE CASCADE,
    dimension             TEXT NOT NULL,
    key                   TEXT NOT NULL,
    airings               INTEGER NOT NULL,
    impressions           BIGINT NOT NULL,
    cost                  NUMERIC(14, 2) NOT NULL,
    attributed_sessions   NUMERIC(14, 2) NOT NULL,
    cost_per_response     NUMERIC(12, 4),
    PRIMARY KEY (run_id, dimension, key)
);

CREATE INDEX IF NOT EXISTS idx_weekly_media_week ON weekly_media (week_start);
CREATE INDEX IF NOT EXISTS idx_weekly_outcomes_week ON weekly_outcomes (week_start);
CREATE INDEX IF NOT EXISTS idx_tv_airings_time ON tv_airings (aired_at);
