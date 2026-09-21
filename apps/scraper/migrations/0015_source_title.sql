-- The page title of a source, written by the pipeline and read back as the citation label.
-- A source with no stored title is cited by its key.

set lock_timeout = '5s';
set statement_timeout = '10min';

alter table sources add column title text;
