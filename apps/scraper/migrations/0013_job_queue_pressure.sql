-- concurrent: this file runs outside a transaction, so the index builds take no write lock.
--
-- Two indexes over jobs, the one table every process writes to. The first serves claiming a job
-- and reading how deep a kind's backlog is; it carries kind, which the old queued index did not,
-- so it replaces that one rather than sitting beside it. The second serves pruning finished jobs.

set lock_timeout = '5s';
set statement_timeout = '30min';

create index concurrently if not exists jobs_kind_queued_idx
    on jobs (kind, priority desc, created_at)
    where status = 'queued';

create index concurrently if not exists jobs_terminal_idx
    on jobs (updated_at)
    where status in ('done', 'failed', 'cancelled');

drop index if exists jobs_queued_idx;
