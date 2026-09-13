-- A chunk row is a page leaf (level 0) or summary of rows below (level 1+); parent_id null at top.

set lock_timeout = '5s';
set statement_timeout = '10min';

alter table chunks add column level int not null default 0;
alter table chunks add column parent_id uuid references chunks(id) on delete set null;

-- Retrieval filters by level next to tenant and category.
create index on chunks (tenant_id, category, level);
-- Walking a summary back down to what it covers.
create index on chunks (parent_id);
