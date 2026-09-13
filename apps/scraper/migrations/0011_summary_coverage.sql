-- The last message a summary stands in for. History loads the newest summary and every message
-- after the position it covers. A summary written before this column has none and covers every
-- message before its own position.

set lock_timeout = '5s';
set statement_timeout = '10min';

alter table messages add column covers_seq bigint;

alter table messages add constraint messages_covers_only_summaries
    check (covers_seq is null or role = 'summary') not valid;
alter table messages validate constraint messages_covers_only_summaries;
