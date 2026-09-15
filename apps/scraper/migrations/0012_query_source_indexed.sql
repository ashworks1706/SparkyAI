-- Whether a live result of a query source is written to the retrieval index. The engine reads it
-- to check the search tool it offers for that source against what the scraper serves.

set lock_timeout = '5s';
set statement_timeout = '10min';

alter table query_sources add column indexed boolean not null default true;
