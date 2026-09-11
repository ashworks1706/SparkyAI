-- The hierarchical index: a chunk row is a leaf cut from the page or a summary of the rows
-- below it.
--
-- level 0 is a leaf. level 1 and above are summaries the scraper writes: cluster the level
-- below, summarize each cluster with a model call, embed the summary, repeat. parent_id points
-- at the summary a row was folded into, and is null at the top of a tree and on any row no
-- summary covers.
--
-- Summaries live in chunks beside the leaves rather than in a table of their own, so one
-- retrieval searches every level at once and the query matches whatever granularity answers
-- it. A source rebuilds its own tree: replace_chunks deletes by source_id, which takes that
-- source's leaves and summaries together.
--
-- A column added with a non-volatile default is recorded in the catalog, so neither statement
-- rewrites the table. parent_id is null on every existing row, so the foreign key has nothing
-- to check.

set lock_timeout = '5s';
set statement_timeout = '10min';

alter table chunks add column level int not null default 0;
alter table chunks add column parent_id uuid references chunks(id) on delete set null;

-- Retrieval filters by level next to tenant and category.
create index on chunks (tenant_id, category, level);
-- Walking a summary back down to what it covers.
create index on chunks (parent_id);
