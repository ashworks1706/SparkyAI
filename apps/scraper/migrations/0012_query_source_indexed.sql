-- Whether a live result of a query source is written to the retrieval index.

set lock_timeout = '5s';
set statement_timeout = '10min';

alter table query_sources add column indexed boolean not null default true;
