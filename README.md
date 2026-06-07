# active-inference

> **Act to reduce uncertainty. The Free Energy Principle in motion.**

[![crates.io](https://img.shields.io/crates/v/active-inference.svg)](https://crates.io/crates/active-inference)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Active inference: agents select actions that minimize expected free energy. Rather than separate perception and action, this framework unifies them — perceiving is minimizing current surprise, acting is minimizing expected future surprise.

## The Active Inference Loop

1. **Observe** sensory state through Markov blanket
2. **Infer** most likely hidden causes (perception = minimization)
3. **Predict** expected free energy for candidate actions
4. **Select** action that minimizes expected surprise
5. **Execute** action → new observation → repeat

This is how biological agents work. Now your AI agents can too.

## License

MIT © [SuperInstance](https://github.com/SuperInstance)

Part of the [Exocortex](https://github.com/SuperInstance/exocortex) project.
