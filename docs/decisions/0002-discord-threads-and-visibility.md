# Discord conversations live in threads; public answers carry no personal memory

2026-09-10.

The bot kept one conversation per user in process memory. It was lost on restart and followed a
user across every channel. Answers were posted in the channel, visible to everyone, while the
prompt carried that user's memory and profile graph.

A conversation now belongs to a place. `/ask` in a text channel opens a thread from the
question and answers inside it; an @mention does the same from the mentioned message; `/ask` or
an @mention inside a thread continues that thread. `/ask private:true` answers ephemerally and
keeps its own conversation for that user in that channel. The engine resolves which conversation
to continue from tenant, user, channel, and visibility (`continue_channel`), so the bot holds no
conversation state and a restart loses nothing. `/reset` ends the open conversations of that user
in that channel.

Every request carries a visibility. A public request, which is anything another member can read,
recalls no memory and no profile graph unless `agent.recall_in_public` is set. Only an ephemeral
answer is private. A thread is public even when the asker opened it: members of the channel can
read it, and moderators can read private threads.

A conversation id is owned. The engine refuses to continue, load, or confirm in a conversation
that belongs to another user or tenant, or that was opened in another channel or at another
visibility, and answers 404 rather than confirming it exists. A private conversation cannot be
continued as public.

Two turns from one user in one channel that arrive together, before either has created a
conversation, can each start one. The next turn continues the more recent. The cooldown makes
this rare, and a split conversation loses context, not data.

Users see and remove what the profile graph holds with `/memory` and `/forget`.

An @mention needs no privileged intent: Discord delivers the content of a message that mentions
the bot. The bot needs Create Public Threads, Send Messages in Threads, and Read Message History.
