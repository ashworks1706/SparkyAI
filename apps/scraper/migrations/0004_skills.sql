-- A skill has parameters, ordered steps, a domain; reviewed before offered, enabled starts false.

set lock_timeout = '5s';
set statement_timeout = '10min';

create table skills (
    key          text primary key,
    title        text not null,
    -- What the skill applies to, shown to the model beside the key so it can pick one.
    domain       text not null,
    description  text not null,
    -- [{name, description, required, example}] rendered into text the model reads. Not JSON Schema.
    params       jsonb not null default '[]'::jsonb,
    -- [{title, detail}] in order. The model follows them with the capabilities it has.
    steps        jsonb not null default '[]'::jsonb,
    enabled      boolean not null default false,
    created_at   timestamptz not null default now(),
    updated_at   timestamptz not null default now()
);
