# active-inference

> **Act to reduce uncertainty. The Free Energy Principle in motion.**

[![crates.io](https://img.shields.io/crates/v/active-inference.svg)](https://crates.io/crates/active-inference)
[![docs.rs](https://docs.rs/active-inference/badge.svg)](https://docs.rs/active-inference)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A Rust library implementing active inference — the framework that unifies perception and action under a single imperative: minimize expected free energy. Agents don't just perceive the world; they act on it to reduce future uncertainty. Implements policy enumeration, expected free energy evaluation, precision-weighted action selection, and Bayesian state estimation.

---

## Table of Contents

- [What is Active Inference?](#what-is-active-inference)
- [Why Does This Matter?](#why-does-this-matter)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Mathematical Background](#mathematical-background)
- [Installation](#installation)
- [Related Crates](#related-crates)
- [License](#license)

---

## What is Active Inference?

Active inference (Friston et al., 2015) extends the Free Energy Principle from perception to action. An agent doesn't just update beliefs to match observations — it acts on the world to make observations match its beliefs. The result: perception and action are two sides of the same minimization process.

The **active inference loop**:

```
   ┌──────────────────────────────────────────────┐
   │                                              │
   │  1. Observe    →  sensory state arrives      │
   │  2. Infer      →  update beliefs (perception)│
   │  3. Predict    →  compute G(π) for policies  │
   │  4. Select     →  pick π with lowest G       │
   │  5. Execute    →  first action of chosen π   │
   │       │                                      │
   │       └──────→ new observation → repeat ──→  │
   │                                              │
   └──────────────────────────────────────────────┘
```

The key equation is **expected free energy** G(π) for a policy π:

```
G(π) = risk(π) + ambiguity(π)
     = KL[q(s|π) || p(s)] + E[H[p(o|s)]]
```

- **Risk**: divergence from preferred states (pragmatic value)
- **Ambiguity**: expected uncertainty in observations (epistemic value)

Agents prefer policies that reach preferred states AND reduce uncertainty. This naturally produces exploration behavior without any explicit exploration bonus.

## Why Does This Matter?

**Unified framework**: No separate reward function, exploration bonus, or value network. A single quantity (expected free energy) produces goal-directed behavior, exploration, and risk avoidance.

**Biological plausibility**: Active inference describes how real nervous systems work — dopamine encodes precision, not reward. Place cells minimize expected free energy. Saccades are epistemic actions.

**Balanced exploration-exploitation**: The epistemic (information-seeking) and pragmatic (goal-seeking) components of G(π) naturally balance exploration and exploitation without tuning parameters.

**Robotics and embodied AI**: An active inference agent placed in a new environment will explore until it has a good model, then exploit that model — just like animals do.

## Architecture

```
active-inference
│
├── policy module              ← Policy representation and enumeration
│   ├── Policy                     Action sequence with horizon
│   ├── enumerate_policies()       All possible action sequences
│   ├── filter_policies()          Constrain to valid policies
│   └── policy_hash()              Deterministic policy fingerprint
│
├── expected_free_energy       ← Core FEP computations
│   ├── expected_free_energy()     G(π) = risk + ambiguity
│   ├── risk()                     KL[q(s|π) || p(s)]
│   ├── information_gain()         H[prior] − H[posterior]
│   ├── epistemic_value()          Information-seeking drive
│   ├── pragmatic_value()          Goal-seeking drive
│   └── rank_policies()            Sort policies by expected free energy
│
├── precision module           ← Uncertainty-weighted decision making
│   ├── precision_weight()         Weight values by confidence
│   ├── softmax_with_precision()   Boltzmann selection with precision
│   ├── update_precision()         Meta-learning of confidence
│   └── precision_from_variance()  Convert uncertainty to precision
│
├── action_selection           ← Choosing what to do
│   ├── select_action()            Greedy: pick best policy
│   ├── select_action_stochastic() Explore: Boltzmann sampling
│   └── action_probabilities()     Softmax over policy values
│
└── perception module          ← State estimation from observations
    ├── estimate_state()           Bayesian state inference
    ├── bayesian_filter()          Sequential belief updating
    ├── confidence()               Entropy-based confidence
    └── belief_divergence()        Belief change magnitude
```

## Quick Start

```rust
use active_inference::{
    policy::{Policy, enumerate_policies},
    expected_free_energy::{expected_free_energy, rank_policies, pragmatic_value},
    action_selection::select_action,
    perception::{estimate_state, bayesian_filter},
};

// Define the world: 3 states, 2 actions, 2 observations
// Transition function: given state and action, what's next?
let transition = |s: usize, a: usize| -> Vec<f64> {
    match (s, a) {
        (0, 0) => vec![0.9, 0.1, 0.0],  // action 0: mostly stay
        (0, 1) => vec![0.1, 0.8, 0.1],  // action 1: move to state 1
        (1, 0) => vec![0.1, 0.8, 0.1],
        (1, 1) => vec![0.0, 0.1, 0.9],  // action 1: move to state 2
        _ => vec![0.33, 0.34, 0.33],
    }
};

// Enumerate all policies (2 actions × 3 timestep horizon = 8 policies)
let policies = enumerate_policies(2, 3);

// Define preferences (agent wants to be in state 2)
let preference = vec![0.1, 0.2, 0.7];

// Rank policies by expected free energy
let ambiguity = |_: usize| 0.5; // uniform ambiguity
let ranked = rank_policies(&policies, &transition, &preference, &ambiguity);

// Select the best action
let best = select_action(&policies, &ranked);
println!("Best policy index: {}", best);

// Perception: observe state from likelihood matrix
let likelihood = vec![
    vec![0.9, 0.1], // state 0 → likely obs 0
    vec![0.5, 0.5], // state 1 → ambiguous
    vec![0.1, 0.9], // state 2 → likely obs 1
];
let prior = vec![0.33, 0.34, 0.33];
let estimate = estimate_state(1, &likelihood, &prior);
println!("Most likely state: {}", estimate.map_state);
```

## API Reference

### Policy

| Method/Function | Returns | Description |
|-----------------|---------|-------------|
| `Policy::new(actions)` | `Policy` | Create action sequence |
| `enumerate_policies(n_actions, horizon)` | `Vec<Policy>` | All possible policies |
| `filter_policies(policies, predicate)` | `Vec<Policy>` | Filter by constraint |
| `policy_hash(policy)` | `u64` | Deterministic hash |

### Expected Free Energy

| Function | Returns | Description |
|----------|---------|-------------|
| `expected_free_energy(π, T, pref, H)` | `f64` | G(π) = risk + ambiguity |
| `risk(state_dist, preference)` | `f64` | KL[q(s\|π) ‖ p(s)] |
| `information_gain(H_prior, H_post)` | `f64` | Epistemic drive |
| `epistemic_value(before, after)` | `f64` | Uncertainty reduction |
| `pragmatic_value(dist, pref)` | `f64` | Goal proximity |
| `rank_policies(policies, ...)` | `Vec<f64>` | G(π) for each policy |

### Precision & Selection

| Function | Returns | Description |
|----------|---------|-------------|
| `precision_weight(values, prec)` | `Vec<f64>` | Weight by confidence |
| `softmax_with_precision(values, γ)` | `Vec<f64>` | Boltzmann with inverse temperature |
| `update_precision(current, error, lr)` | `f64` | Precision learning |
| `select_action(policies, values)` | `usize` | Greedy selection |
| `select_action_stochastic(policies, values, T)` | `usize` | Softmax exploration |

### Perception

| Function | Returns | Description |
|----------|---------|-------------|
| `estimate_state(obs, likelihood, prior)` | `StateEstimate` | Bayesian inference |
| `bayesian_filter(prev, obs, likelihood, transition)` | `Vec<f64>` | Sequential filtering |
| `confidence(dist)` | `f64` | 1 − normalized entropy |
| `posterior_entropy(dist)` | `f64` | H[p(s\|o)] |
| `predict_belief(belief, transition)` | `Vec<f64>` | Belief propagation |

## Mathematical Background

### Expected Free Energy

For a policy π over horizon T, expected free energy is:

```
G(π) = Σ_t D_KL[q(s_t|π) || p(s_t)] + E_q[H[p(o_t|s_t)]]
```

- First term (**risk**): how far will beliefs be from preferences?
- Second term (**ambiguity**): how uncertain will observations be?

### Active Inference vs Reinforcement Learning

| Aspect | Reinforcement Learning | Active Inference |
|--------|----------------------|------------------|
| Objective | Maximize reward R(s,a) | Minimize free energy G(π) |
| Exploration | ε-greedy, curiosity bonus | Emerges from ambiguity term |
| Policy | π(a\|s) → maximize Q(s,a) | π* = argmin G(π) |
| Perception | State estimation (separate) | Belief update = free energy min |
| Action | Separate from perception | Unified with perception |

### Bayesian Filtering

Sequential state estimation updates beliefs given new observations:

```
p(s_t | o_1:t) ∝ p(o_t | s_t) Σ_{s_{t-1}} p(s_t | s_{t-1}) p(s_{t-1} | o_1:t-1)
```

This is the perception step: predict (transition) then update (likelihood). The `bayesian_filter` function implements this exact recursion.

### Precision as Attention

Precision (inverse variance) modulates how strongly prediction errors update beliefs:

```
Δq ∝ γ · ε    where γ = 1/σ²
```

High precision → trust the input → large belief update. Low precision → ignore the input → maintain prior. Dopamine in the brain is thought to encode precision.

## Installation

```bash
cargo add active-inference
```

Or add to your `Cargo.toml`:

```toml
[dependencies]
active-inference = "0.1"
```

## Related Crates

Part of the **SuperInstance Exocortex** ecosystem:

- **[markov-blanket](https://github.com/SuperInstance/markov-blanket)** — Statistical boundary between agent and world
- **[free-energy](https://github.com/SuperInstance/free-energy)** — Variational free energy computation
- **[signal-transduction](https://github.com/SuperInstance/signal-transduction)** — Signal cascading for agent systems
- **[morphogenesis](https://github.com/SuperInstance/morphogenesis)** — Turing patterns for agent development
- **[dream-cycle](https://github.com/SuperInstance/dream-cycle)** — Sleep consolidation for agent memory

## License

MIT © [SuperInstance](https://github.com/SuperInstance)

Part of the [Exocortex](https://github.com/SuperInstance/exocortex) project.
