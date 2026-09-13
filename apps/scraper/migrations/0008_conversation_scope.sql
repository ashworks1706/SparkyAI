-- visibility is public or private to the owner. ended_at marks a reset; picked up again only by id.

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
