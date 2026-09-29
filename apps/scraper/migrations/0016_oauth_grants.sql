-- Per-user OAuth grants (Canvas and other providers) and the short-lived state of a login.
-- The engine writes both; the scraper never touches them. Tokens are secrets: never logged.

create table oauth_grants (
    tenant_id      text not null,
    user_id        text not null,
    provider       text not null,
    access_token   text not null,
    refresh_token  text,
    scopes         text[] not null default '{}',
    expires_at     timestamptz,
    updated_at     timestamptz not null default now(),
    primary key (tenant_id, user_id, provider)
);

create table oauth_states (
    state          text primary key,
    tenant_id      text not null,
    user_id        text not null,
    provider       text not null,
    created_at     timestamptz not null default now(),
    expires_at     timestamptz not null
);
create index on oauth_states (expires_at);
