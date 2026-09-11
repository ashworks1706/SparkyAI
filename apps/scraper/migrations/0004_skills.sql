-- Skills: saved procedures the model fetches and follows.
--
-- A skill is parameters, an ordered list of steps, and the domain it applies to. It is not
-- code the model wrote and nothing writes this table at runtime: a skill is reviewed before
-- it is offered, so enabled starts false and a person turns it on.

set lock_timeout = '5s';
set statement_timeout = '10min';

create table skills (
    key          text primary key,
    title        text not null,
    -- What the skill applies to, shown to the model beside the key so it can pick one.
    domain       text not null,
    description  text not null,
    -- [{name, description, required, example}] rendered into the text the model reads. Not
    -- JSON Schema: the model supplies these values while it follows the steps.
    params       jsonb not null default '[]'::jsonb,
    -- [{title, detail}] in order. The model follows them with the capabilities it has.
    steps        jsonb not null default '[]'::jsonb,
    enabled      boolean not null default false,
    created_at   timestamptz not null default now(),
    updated_at   timestamptz not null default now()
);
