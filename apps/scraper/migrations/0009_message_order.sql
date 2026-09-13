-- The order messages were written in. The messages of one turn are inserted in one transaction
-- and share created_at; seq is assigned in insert order, and history is read by it.
--
-- Adding the identity column rewrites messages under an exclusive lock. Existing rows are
-- numbered in physical order, the order this append-only table was written in.

set lock_timeout = '5s';
set statement_timeout = '10min';

alter table messages add column seq bigint generated always as identity;

create index messages_conversation_seq_idx on messages (conversation_id, seq);
create index messages_summary_seq_idx on messages (conversation_id, seq desc)
    where role = 'summary';

drop index messages_conversation_id_created_at_idx;
drop index messages_summary_idx;
