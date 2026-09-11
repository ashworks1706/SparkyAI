-- A compacted turn stands in for the turns it replaced.
--
-- The turns it replaced stay in messages. A load returns the newest summary and everything
-- after it, so the summary supersedes what came before without deleting it.

set lock_timeout = '5s';
set statement_timeout = '10min';

alter table messages drop constraint messages_role_check;
alter table messages
  add constraint messages_role_check
  check (role in ('system', 'user', 'assistant', 'tool', 'summary')) not valid;
alter table messages validate constraint messages_role_check;

-- Finding the newest summary is the first thing a load does.
create index messages_summary_idx on messages (conversation_id, created_at desc)
  where role = 'summary';
