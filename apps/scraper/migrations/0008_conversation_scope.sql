-- Who may see a conversation, and whether it is still open.
--
-- visibility is public for a conversation held where others can read it, and private for one
-- only its owner sees. ended_at is set when the owner resets a channel; an ended conversation
-- is never picked up again by channel, only by its id.
--
-- A column added with a constant default is recorded in the catalog, so neither statement
-- rewrites the table. Every existing row reads as public and open.

set lock_timeout = '5s';
set statement_timeout = '10min';

alter table conversations
    add column visibility text not null default 'public'
        check (visibility in ('public', 'private'));
alter table conversations add column ended_at timestamptz;

-- Conversations from the OpenAI-compatible surface are one-to-one, so they are private.
update conversations set visibility = 'private' where channel_id = 'openai';

-- The newest open conversation of one user in one channel is what a continuing turn looks up.
create index conversations_open_idx
    on conversations (tenant_id, user_id, channel_id, visibility, updated_at desc)
    where ended_at is null;
