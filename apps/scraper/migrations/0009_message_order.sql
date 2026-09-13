-- Write order of messages. One turn's messages share created_at; seq breaks ties by insert order.

set lock_timeout = '5s';
set statement_timeout = '10min';

alter table messages add column seq bigint generated always as identity;

create index messages_conversation_seq_idx on messages (conversation_id, seq);
create index messages_summary_seq_idx on messages (conversation_id, seq desc)
    where role = 'summary';

drop index messages_conversation_id_created_at_idx;
drop index messages_summary_idx;
