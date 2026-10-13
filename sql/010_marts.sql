-- Reporting marts built on the base tables. These feed the weekly readout and the dashboards.

CREATE OR REPLACE VIEW national_weekly AS
SELECT
    o.week_start,
    SUM(o.new_subscriptions)             AS new_subscriptions,
    AVG(o.promo_depth)                   AS promo_depth,
    AVG(o.competitor_grps)               AS competitor_grps,
    AVG(o.price_index)                   AS price_index
FROM weekly_outcomes o
GROUP BY o.week_start;

CREATE OR REPLACE VIEW national_weekly_media AS
SELECT
    m.week_start,
    m.channel,
    SUM(m.spend)        AS spend,
    SUM(m.impressions)  AS impressions
FROM weekly_media m
GROUP BY m.week_start, m.channel;

-- The most recent run with the harmonized figure beside the three inputs.
CREATE OR REPLACE VIEW latest_reconciliation AS
SELECT r.*
FROM reconciliation r
JOIN (
    SELECT run_id FROM model_runs ORDER BY created_at DESC LIMIT 1
) latest ON latest.run_id = r.run_id;
