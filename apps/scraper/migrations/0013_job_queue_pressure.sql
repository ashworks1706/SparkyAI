-- concurrent: this file runs outside a transaction, so the index builds take no write lock.
--
-- Two indexes over jobs: one for claiming and backlog depth by kind, one for pruning finished jobs.

set lock_timeout = '5s';
set statement_timeout = '30min';

create index concurrently if not exists jobs_kind_queued_idx
    on jobs (kind, priority desc, created_at)
    where status = 'queued';

create index concurrently if not exists jobs_terminal_idx
    on jobs (updated_at)
    where status in ('done', 'failed', 'cancelled');

drop index if exists jobs_queued_idx;
