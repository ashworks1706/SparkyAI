<p align="center">
  <img src="apps/web/public/brand/sparkyai-logo.png" alt="SparkyAI dragon logo" width="180">
</p>

# SparkyAI

SparkyAI is an open-source Discord copilot for ASU students and club moderators. It indexes public university sources, answers with dated citations, and requires confirmation before consequential actions.

The Rust rebuild on `main` is under active development and is not ready for end users. See the [roadmap](docs/ROADMAP.md) and [architecture](docs/ARCHITECTURE.md). The original Python prototype remains on [`archive/v1`](https://github.com/ashworks1706/SparkyAI/tree/archive/v1).

## Development

Install [just](https://just.systems), then run:

```bash
git clone https://github.com/ashworks1706/SparkyAI.git
cd SparkyAI
just doctor
just bootstrap
just check
```

`just` lists every recipe. `just cli` opens the developer console; `just engine`, `just discord`, and `just web` run individual apps.

## Local state

Generated runtime and training files live under `.sparky/`:

```text
.sparky/
  logs/              developer-console logs
  traces/            replayable engine traces
  training/data/     exported and verified datasets
  training/evals/    generated eval reports
  training/outputs/  checkpoints, TensorBoard logs, and model exports
```

The directory is ignored by Git. Source-controlled eval cases and baselines remain in `apps/training/evals/`.

## License

[MIT](LICENSE)
