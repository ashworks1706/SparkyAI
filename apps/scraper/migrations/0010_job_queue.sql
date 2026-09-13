-- One queue for everything the scraper does: live queries the engine queues, the indexing of
-- their results, and scheduled source runs. A claim takes the highest priority first, then the
-- oldest. The engine inserts at the default priority; the scraper queues its own work below it.
--
-- Adding a column with a constant default does not rewrite jobs.

set lock_timeout = '5s';
set statement_timeout = '10min';

alter table jobs add column priority smallint not null default 0;

drop index jobs_queued_idx;
create index jobs_queued_idx on jobs (priority desc, created_at) where status = 'queued';

-- At most one scheduled run of a source waits or runs at a time.
create unique index jobs_one_source_run_idx on jobs ((input->>'source'))
    where kind = 'source_run' and status in ('queued', 'running');
