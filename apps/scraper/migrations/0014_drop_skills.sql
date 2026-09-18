-- Removes the skills table; the engine offers no saved procedures.

set lock_timeout = '5s';
set statement_timeout = '10min';

drop table if exists skills;
