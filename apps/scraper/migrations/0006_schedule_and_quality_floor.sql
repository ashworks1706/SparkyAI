-- last_attempt_at advances every attempt; text_chars/chunk_count refuse a much smaller replacement.

set lock_timeout = '5s';
set statement_timeout = '10min';

alter table sources add column last_attempt_at timestamptz;

alter table source_versions add column text_chars int;
alter table source_versions add column chunk_count int;
