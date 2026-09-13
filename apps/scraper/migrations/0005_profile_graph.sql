-- The profile graph: a node holds one entity plus its embedding, an edge holds one relation.

set lock_timeout = '5s';
set statement_timeout = '10min';

-- Dimension matches chunks.embedding: nodes are embedded with the same model.
create table profile_nodes (
    id            uuid primary key default gen_random_uuid(),
    tenant_id     text not null,
    user_id       uuid not null references users(id) on delete cascade,
    kind          text not null,
    label         text not null,
    embedding     vector(1024) not null,
    confidence    real not null default 1.0,
    created_at    timestamptz not null default now(),
    updated_at    timestamptz not null default now(),
    -- Repeated extraction of the same entity updates one row instead of adding another.
    unique (tenant_id, user_id, kind, label)
);
create index on profile_nodes (tenant_id, user_id, confidence desc, updated_at desc);
create index on profile_nodes using hnsw (embedding vector_cosine_ops)
  with (m = 16, ef_construction = 64);

create table profile_edges (
    id            uuid primary key default gen_random_uuid(),
    tenant_id     text not null,
    from_node     uuid not null references profile_nodes(id) on delete cascade,
    to_node       uuid not null references profile_nodes(id) on delete cascade,
    relation      text not null,
    confidence    real not null default 1.0,
    created_at    timestamptz not null default now(),
    unique (from_node, to_node, relation)
);
create index on profile_edges (tenant_id, from_node);
create index on profile_edges (tenant_id, to_node);
