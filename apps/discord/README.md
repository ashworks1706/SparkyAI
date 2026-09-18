# apps/discord

The Discord bot. A serenity gateway client and an HTTP client of the engine, and nothing else:
it never links the engine crate and holds no agent logic. It decides who may ask, renders what
comes back, and relays the progress of a turn as it happens.

```bash
just discord                        # needs SPARKY_DISCORD__TOKEN and GUILD_ID in .env
cargo run -p discord
cargo test -p discord
```

## What it does

- Decides whether a message is for the bot: a mention, a reply to it, a thread it opened, or a
  direct message when `bot.direct_messages` is on, within `bot.channels` when that is set.
- Sends one `POST /chat` per turn with the asker, the channel, the roles they hold, the message
  they replied to, and any images.
- Renders the stream into one card that it edits as the turn runs: the thinking line, the steps,
  then the answer and its citations. `bot.edit_every_ms` paces the edits, and a message longer
  than `bot.max_message_chars` is split on line boundaries.
- Shows a confirmation as buttons when the engine holds an action, and calls `POST /confirm`
  with the token when someone presses one.
- Records one span per interaction and one span per product event, both exported through
  `telemetry`.

## Layout

```
src/core/       config, telemetry, types, tests. Imports nothing else in the crate.
src/bot/        the serenity handler: addressing, turns, threads, accounts, confirmations
src/engine/     the HTTP client of apps/engine, including the progress stream
src/render/     the card, the step list, splitting, Discord markdown
src/access/     who may ask, and where the answer is visible
src/analytics/  product events, one exported span each
```

## Configuration

`bot` in `sparky.toml` holds everything about behaviour: which channels, how often the card is
edited, the cooldown, thread archiving, how many images reach the model, and
`bot.write_capability`, the role name the engine's policy reads for write-side tools. It must
match `policy.write_roles` on the engine side or a write tool is refused for everyone.

`.env` holds `SPARKY_DISCORD__TOKEN`, `SPARKY_DISCORD__GUILD_ID`, and
`SPARKY_ENGINE__SERVICE_TOKEN`, which must be the same string the engine was started with.
