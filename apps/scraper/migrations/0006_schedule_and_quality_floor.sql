-- What the scheduler reads, and what one version is measured by.
--
-- last_attempt_at advances on every run attempt, whether the page changed, stayed byte
-- identical, or failed. The scheduler compares it against sources.fetch_every, which is the
-- authoritative interval: code supplies it when the row is created and never overwrites it
-- after, so an operator can back a source off without a redeploy.
--
-- text_chars and chunk_count record what a version extracted. A later run compares its own
-- extraction against the newest row and refuses to replace the index when it yields
-- materially less. Both are null on versions written before this migration, which reads as
-- unknown and skips the comparison.

set lock_timeout = '5s';
set statement_timeout = '10min';

alter table sources add column last_attempt_at timestamptz;

alter table source_versions add column text_chars int;
alter table source_versions add column chunk_count int;
