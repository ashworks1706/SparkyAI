-- Source of truth. Vector indexes in Qdrant are rebuildable from here + object storage.

create table users (
    id            uuid primary key default gen_random_uuid(),
    tenant_id     text not null,
    discord_id    text not null,
    roles         text[] not null default '{}',
    created_at    timestamptz not null default now(),
    unique (tenant_id, discord_id)
);

create table conversations (
    id            uuid primary key default gen_random_uuid(),
    tenant_id     text not null,
    user_id       uuid not null references users(id),
    channel_id    text not null,
    created_at    timestamptz not null default now(),
    updated_at    timestamptz not null default now()
);
create index on conversations (tenant_id, user_id, updated_at desc);

create table messages (
    id               uuid primary key default gen_random_uuid(),
    conversation_id  uuid not null references conversations(id) on delete cascade,
    role             text not null check (role in ('system','user','assistant','tool')),
    content          jsonb not null,
    created_at       timestamptz not null default now()
);
create index on messages (conversation_id, created_at);

create table memories (
    id            uuid primary key default gen_random_uuid(),
    tenant_id     text not null,
    user_id       uuid not null references users(id) on delete cascade,
    kind          text not null check (kind in ('episodic','semantic','profile','task')),
    content       text not null,
    sensitivity   text not null default 'normal',
    confidence    real not null default 1.0,
    source_msg    uuid references messages(id),
    created_at    timestamptz not null default now(),
    expires_at    timestamptz
);
create index on memories (tenant_id, user_id, kind);

create table sources (
    id            uuid primary key default gen_random_uuid(),
    key           text not null unique,
    url           text not null,
    category      text not null,
    fetch_every   interval not null default '24 hours',
    enabled       boolean not null default true
);

create table source_versions (
    id               uuid primary key default gen_random_uuid(),
    source_id        uuid not null references sources(id) on delete cascade,
    content_hash     text not null,
    fetched_at       timestamptz not null default now(),
    snapshot_key     text not null,
    parser_version   text not null,
    chunker_version  text not null,
    embedding_model  text not null,
    previous_id      uuid references source_versions(id)
);
create index on source_versions (source_id, fetched_at desc);

create table jobs (
    id            uuid primary key default gen_random_uuid(),
    kind          text not null,
    status        text not null check (status in ('queued','running','done','failed','cancelled')),
    owner         text,
    input         jsonb,
    result        jsonb,
    error         text,
    attempts      int not null default 0,
    deadline      timestamptz,
    created_at    timestamptz not null default now(),
    updated_at    timestamptz not null default now()
);
create index on jobs (status, created_at);

create table confirmations (
    id            uuid primary key default gen_random_uuid(),
    request_id    uuid not null,
    user_id       uuid not null references users(id),
    action        jsonb not null,
    payload_hash  text not null,
    status        text not null check (status in ('pending','confirmed','denied','expired')),
    created_at    timestamptz not null default now(),
    expires_at    timestamptz not null,
    resolved_at   timestamptz
);
create index on confirmations (user_id, status);
