//! Active inference for goal-directed behavior.
//!
//! Implements policy evaluation, expected free energy computation, precision
//! weighting, action selection, and perception (state estimation).

// ── Module: policy ───────────────────────────────────────────────────────────

pub mod policy {
    #[derive(Clone, Debug)]
    pub struct Policy {
        pub actions: Vec<usize>,
        pub horizon: usize,
    }

    impl Policy {
        pub fn new(actions: Vec<usize>) -> Self {
            let horizon = actions.len();
            Policy { actions, horizon }
        }
        pub fn empty() -> Self { Policy { actions: vec![], horizon: 0 } }
        pub fn is_empty(&self) -> bool { self.actions.is_empty() }
        pub fn len(&self) -> usize { self.actions.len() }
    }

    pub fn enumerate_policies(n_actions: usize, horizon: usize) -> Vec<Policy> {
        if horizon == 0 || n_actions == 0 { return vec![Policy::empty()]; }
        let mut policies = Vec::new();
        let mut current = vec![0usize; horizon];
        loop {
            policies.push(Policy::new(current.clone()));
            let mut carry = true;
            for i in (0..horizon).rev() {
                if carry {
                    current[i] += 1;
                    if current[i] >= n_actions { current[i] = 0; } else { carry = false; }
                }
            }
            if carry { break; }
        }
        policies
    }

    pub fn filter_policies(policies: &[Policy], f: impl Fn(&Policy) -> bool) -> Vec<Policy> {
        policies.iter().filter(|p| f(p)).cloned().collect()
    }

    pub fn policy_hash(policy: &Policy) -> u64 {
        let mut h: u64 = 0;
        for &a in &policy.actions { h = h.wrapping_mul(31).wrapping_add(a as u64); }
        h
    }
}

// ── Module: expected_free_energy ─────────────────────────────────────────────

pub mod expected_free_energy {
    use crate::policy::Policy;

    pub fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
        p.iter().zip(q.iter())
            .filter(|(pi, _)| **pi > 0.0)
            .map(|(pi, qi)| { let qs = qi.max(1e-10); pi * (pi / qs).ln() })
            .sum()
    }

    pub fn expected_free_energy(
        policy: &Policy,
        transition: &dyn Fn(usize, usize) -> Vec<f64>,
        preference: &[f64],
        ambiguity: &dyn Fn(usize) -> f64,
    ) -> f64 {
        let mut state_dist = vec![1.0];
        let mut total_risk = 0.0;
        let mut total_ambiguity = 0.0;
        for &action in &policy.actions {
            let mut next_dist = vec![0.0; preference.len()];
            for (s, &p) in state_dist.iter().enumerate() {
                if p > 0.0 {
                    let trans = transition(s, action);
                    for (ns, &t) in trans.iter().enumerate() {
                        if ns < next_dist.len() { next_dist[ns] += p * t; }
                    }
                }
            }
            total_risk += kl_divergence(&next_dist, preference);
            for (s, &p) in next_dist.iter().enumerate() {
                if p > 0.0 { total_ambiguity += p * ambiguity(s); }
            }
            state_dist = next_dist;
        }
        total_risk + total_ambiguity
    }

    pub fn risk(state_dist: &[f64], preference: &[f64]) -> f64 { kl_divergence(state_dist, preference) }

    pub fn information_gain(prior_entropy: f64, posterior_entropy: f64) -> f64 { prior_entropy - posterior_entropy }

    pub fn epistemic_value(before: &[f64], after: &[f64]) -> f64 {
        before.iter().zip(after.iter()).map(|(&b, &a)| b - a).sum()
    }

    pub fn pragmatic_value(state_dist: &[f64], preference: &[f64]) -> f64 {
        state_dist.iter().zip(preference.iter())
            .map(|(s, p)| s * p.max(1e-10).ln())
            .sum()
    }

    pub fn rank_policies(
        policies: &[Policy], transition: &dyn Fn(usize, usize) -> Vec<f64>,
        preference: &[f64], ambiguity: &dyn Fn(usize) -> f64,
    ) -> Vec<(usize, f64)> {
        let mut ranked: Vec<(usize, f64)> = policies.iter().enumerate()
            .map(|(i, p)| (i, expected_free_energy(p, transition, preference, ambiguity)))
            .collect();
        ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        ranked
    }
}

// ── Module: precision ────────────────────────────────────────────────────────

pub mod precision {
    pub fn precision_weight(values: &[f64], precisions: &[f64]) -> Vec<f64> {
        values.iter().zip(precisions.iter()).map(|(v, p)| v * p).collect()
    }

    pub fn softmax_with_precision(values: &[f64], precision: f64) -> Vec<f64> {
        let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = values.iter().map(|v| (precision * (v - max_val)).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|e| e / sum).collect()
    }

    pub fn update_precision(current: f64, error: f64, lr: f64) -> f64 { (current + lr * error).max(0.01) }
    pub fn precision_from_variance(variance: f64) -> f64 { if variance > 0.0 { 1.0 / variance } else { f64::MAX } }
    pub fn precision_diagonal(variances: &[f64]) -> Vec<f64> { variances.iter().map(|v| precision_from_variance(*v)).collect() }
    pub fn expected_precision(shape: f64, rate: f64) -> f64 { if rate > 0.0 { shape / rate } else { f64::MAX } }

    pub fn precision_weighted_average(values: &[f64], precisions: &[f64]) -> f64 {
        let ws: f64 = values.iter().zip(precisions.iter()).map(|(v, p)| v * p).sum();
        let ps: f64 = precisions.iter().sum();
        if ps > 0.0 { ws / ps } else { 0.0 }
    }

    pub fn normalize_precisions(precisions: &[f64]) -> Vec<f64> {
        let sum: f64 = precisions.iter().sum();
        if sum > 0.0 { precisions.iter().map(|p| p / sum).collect() }
        else { vec![1.0 / precisions.len() as f64; precisions.len()] }
    }
}

// ── Module: action ───────────────────────────────────────────────────────────

pub mod action {
    use crate::policy::Policy;

    pub fn select_action(policies: &[Policy], values: &[f64]) -> usize {
        values.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i).unwrap_or(0)
    }

    pub fn select_action_stochastic(policies: &[Policy], values: &[f64], temperature: f64) -> usize {
        let probs = super::precision::softmax_with_precision(values, 1.0 / temperature);
        let mut cumsum = 0.0;
        for (i, &p) in probs.iter().enumerate() { cumsum += p; if cumsum >= 0.5 { return i; } }
        policies.len() - 1
    }

    pub fn first_action(policy: &Policy) -> Option<usize> { policy.actions.first().copied() }
    pub fn action_at(policy: &Policy, t: usize) -> Option<usize> { policy.actions.get(t).copied() }

    pub fn execute_policy<F: Fn(usize, usize) -> usize>(policy: &Policy, initial: usize, transition: F) -> Vec<usize> {
        let mut states = vec![initial];
        let mut current = initial;
        for &a in &policy.actions { current = transition(current, a); states.push(current); }
        states
    }

    pub fn action_probabilities(values: &[f64], temperature: f64) -> Vec<f64> {
        super::precision::softmax_with_precision(values, 1.0 / temperature)
    }

    pub fn actions_equal(a: usize, b: usize) -> bool { a == b }

    pub fn action_label(action: usize, labels: &[&str]) -> Option<String> {
        labels.get(action).map(|s| s.to_string())
    }
}

// ── Module: perception ────────────────────────────────────────────────────────

pub mod perception {
    #[derive(Clone, Debug)]
    pub struct StateEstimate {
        pub state_distribution: Vec<f64>,
        pub entropy: f64,
        pub most_likely: usize,
    }

    pub fn estimate_state(observation: usize, likelihood: &[Vec<f64>], prior: &[f64]) -> StateEstimate {
        let n = prior.len();
        let mut posterior = vec![0.0; n];
        for s in 0..n {
            let p_obs = likelihood.get(s).and_then(|d| d.get(observation)).copied().unwrap_or(0.0);
            posterior[s] = prior[s] * p_obs;
        }
        let sum: f64 = posterior.iter().sum();
        if sum > 0.0 { for p in &mut posterior { *p /= sum; } }
        let entropy = -posterior.iter().filter(|&&p| p > 0.0).map(|&p| p * p.ln()).sum::<f64>();
        let most_likely = posterior.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i).unwrap_or(0);
        StateEstimate { state_distribution: posterior, entropy, most_likely }
    }

    pub fn bayesian_filter(prev: &[f64], obs: usize, likelihood: &[Vec<f64>], transition: &[Vec<f64>]) -> Vec<f64> {
        let n = prev.len();
        let mut predicted = vec![0.0; n];
        for s in 0..n {
            for sp in 0..n {
                predicted[s] += transition.get(sp).and_then(|t| t.get(s)).copied().unwrap_or(0.0) * prev[sp];
            }
        }
        let mut posterior = vec![0.0; n];
        for s in 0..n {
            posterior[s] = predicted[s] * likelihood.get(s).and_then(|d| d.get(obs)).copied().unwrap_or(0.0);
        }
        let sum: f64 = posterior.iter().sum();
        if sum > 0.0 { for p in &mut posterior { *p /= sum; } }
        posterior
    }

    pub fn posterior_entropy(dist: &[f64]) -> f64 { -dist.iter().filter(|&&p| p > 0.0).map(|&p| p * p.ln()).sum::<f64>() }

    pub fn confidence(dist: &[f64]) -> f64 {
        let h = posterior_entropy(dist);
        let max_h = (dist.len() as f64).ln();
        if max_h > 0.0 { 1.0 - h / max_h } else { 1.0 }
    }

    pub fn belief_divergence(before: &[f64], after: &[f64]) -> f64 {
        before.iter().zip(after.iter())
            .filter(|(b, _)| **b > 0.0)
            .map(|(b, a)| { let as_ = a.max(1e-10); b * (b / as_).ln() })
            .sum()
    }

    pub fn predict_belief(belief: &[f64], transition: &[Vec<f64>]) -> Vec<f64> {
        let n = belief.len();
        let mut pred = vec![0.0; n];
        for s in 0..n {
            for sp in 0..n {
                pred[s] += transition.get(sp).and_then(|t| t.get(s)).copied().unwrap_or(0.0) * belief[sp];
            }
        }
        pred
    }

    pub fn uniform_belief(n: usize) -> Vec<f64> { vec![1.0 / n as f64; n] }

    pub fn marginal_observation(belief: &[f64], likelihood: &[Vec<f64>]) -> Vec<f64> {
        let n_obs = likelihood.first().map(|d| d.len()).unwrap_or(0);
        let mut marginal = vec![0.0; n_obs];
        for s in 0..belief.len() {
            if let Some(od) = likelihood.get(s) {
                for (o, &p) in od.iter().enumerate() { if o < marginal.len() { marginal[o] += belief[s] * p; } }
            }
        }
        marginal
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test] fn test_policy_creation() { let p = policy::Policy::new(vec![0,1,2]); assert_eq!(p.len(), 3); assert_eq!(p.horizon, 3); }
    #[test] fn test_policy_empty() { let p = policy::Policy::empty(); assert!(p.is_empty()); }
    #[test] fn test_enumerate_policies() { assert_eq!(policy::enumerate_policies(2, 2).len(), 4); }
    #[test] fn test_enumerate_policies_single() { assert_eq!(policy::enumerate_policies(3, 1).len(), 3); }
    #[test] fn test_filter_policies() {
        let ps = policy::enumerate_policies(2, 2);
        assert_eq!(policy::filter_policies(&ps, |p| p.actions[0] == 0).len(), 2);
    }
    #[test] fn test_policy_hash_same() { let p1 = policy::Policy::new(vec![0,1]); let p2 = policy::Policy::new(vec![0,1]); assert_eq!(policy::policy_hash(&p1), policy::policy_hash(&p2)); }
    #[test] fn test_policy_hash_diff() { let p1 = policy::Policy::new(vec![0,1]); let p2 = policy::Policy::new(vec![1,0]); assert_ne!(policy::policy_hash(&p1), policy::policy_hash(&p2)); }
    #[test] fn test_policy_len() { let p = policy::Policy::new(vec![0,1,2,3]); assert_eq!(p.len(), 4); }
    #[test] fn test_policy_not_empty() { let p = policy::Policy::new(vec![0]); assert!(!p.is_empty()); }
    #[test] fn test_enumerate_3x3() { assert_eq!(policy::enumerate_policies(3, 3).len(), 27); }

    #[test] fn test_kl_same() { assert!(expected_free_energy::kl_divergence(&[0.5,0.5], &[0.5,0.5]).abs() < 1e-10); }
    #[test] fn test_kl_diff() { assert!(expected_free_energy::kl_divergence(&[0.9,0.1], &[0.5,0.5]) > 0.0); }
    #[test] fn test_risk() { assert!(expected_free_energy::risk(&[0.9,0.1], &[0.5,0.5]) > 0.0); }
    #[test] fn test_information_gain() { assert!((expected_free_energy::information_gain(2.0, 1.0) - 1.0).abs() < 1e-10); }
    #[test] fn test_epistemic_value() { assert!((expected_free_energy::epistemic_value(&[2.0,1.5], &[1.0,0.5]) - 2.0).abs() < 1e-10); }
    #[test] fn test_pragmatic_value() { assert!(expected_free_energy::pragmatic_value(&[0.5,0.5], &[0.9,0.1]) < 0.0); }
    #[test] fn test_rank_policies() {
        let ps = policy::enumerate_policies(2, 2);
        assert_eq!(expected_free_energy::rank_policies(&ps, &|_,_| vec![0.5,0.5], &[0.5,0.5], &|_| 0.0).len(), 4);
    }
    #[test] fn test_efe_basic() {
        let p = policy::Policy::new(vec![0]);
        assert!(expected_free_energy::expected_free_energy(&p, &|_,_| vec![0.5,0.5], &[0.5,0.5], &|_| 0.0) >= 0.0);
    }

    #[test] fn test_pw() { let w = precision::precision_weight(&[1.0,2.0], &[0.5,2.0]); assert!((w[0]-0.5).abs()<1e-10); assert!((w[1]-4.0).abs()<1e-10); }
    #[test] fn test_softmax() { let s = precision::softmax_with_precision(&[1.0,2.0,3.0], 1.0); assert!((s.iter().sum::<f64>()-1.0).abs()<1e-10); assert!(s[2]>s[0]); }
    #[test] fn test_softmax_high() { let s = precision::softmax_with_precision(&[1.0,3.0], 100.0); assert!(s[1]>0.99); }
    #[test] fn test_softmax_low() { let s = precision::softmax_with_precision(&[1.0,3.0], 0.01); assert!((s[0]-s[1]).abs()<0.1); }
    #[test] fn test_update_prec() { assert!((precision::update_precision(1.0, 0.5, 0.1) - 1.05).abs() < 1e-10); }
    #[test] fn test_prec_from_var() { assert!((precision::precision_from_variance(0.25) - 4.0).abs() < 1e-10); }
    #[test] fn test_prec_diag() { let d = precision::precision_diagonal(&[0.5,1.0,2.0]); assert!((d[0]-2.0).abs()<1e-10); assert!((d[2]-0.5).abs()<1e-10); }
    #[test] fn test_expected_prec() { assert!((precision::expected_precision(2.0, 1.0) - 2.0).abs() < 1e-10); }
    #[test] fn test_pw_average() { assert!((precision::precision_weighted_average(&[1.0,3.0], &[1.0,3.0]) - 2.5).abs() < 1e-10); }
    #[test] fn test_norm_prec() { let n = precision::normalize_precisions(&[1.0,3.0]); assert!((n.iter().sum::<f64>()-1.0).abs()<1e-10); }
    #[test] fn test_pw_zero() { assert_eq!(precision::precision_weight(&[1.0,2.0], &[0.0,0.0]), vec![0.0,0.0]); }

    #[test] fn test_select_action() { let ps = policy::enumerate_policies(3,1); assert_eq!(action::select_action(&ps, &[0.1,0.5,0.3]), 1); }
    #[test] fn test_first_action() { assert_eq!(action::first_action(&policy::Policy::new(vec![2,1,0])), Some(2)); }
    #[test] fn test_action_at() { let p = policy::Policy::new(vec![0,1,2]); assert_eq!(action::action_at(&p,1), Some(1)); assert_eq!(action::action_at(&p,5), None); }
    #[test] fn test_execute_policy() { assert_eq!(action::execute_policy(&policy::Policy::new(vec![0,1]), 0, |s,a| s+a+1), vec![0,1,3]); }
    #[test] fn test_action_probs() { let p = action::action_probabilities(&[0.5,1.5], 1.0); assert!((p.iter().sum::<f64>()-1.0).abs()<1e-10); }
    #[test] fn test_actions_equal() { assert!(action::actions_equal(1,1)); assert!(!action::actions_equal(1,2)); }
    #[test] fn test_action_label() { assert_eq!(action::action_label(1, &["left","right","up"]), Some("right".into())); }
    #[test] fn test_select_stochastic() { let ps = policy::enumerate_policies(2,1); assert!(action::select_action_stochastic(&ps, &[0.1,0.9], 1.0) < ps.len()); }

    #[test] fn test_estimate_state() {
        let l = vec![vec![0.9,0.1], vec![0.1,0.9]];
        let e = perception::estimate_state(0, &l, &[0.5,0.5]);
        assert_eq!(e.most_likely, 0);
    }
    #[test] fn test_estimate_entropy() {
        let l = vec![vec![0.9,0.1], vec![0.1,0.9]];
        assert!(perception::estimate_state(0, &l, &[0.5,0.5]).entropy >= 0.0);
    }
    #[test] fn test_bayes_filter() {
        let l = vec![vec![0.9,0.1], vec![0.1,0.9]];
        let t = vec![vec![0.8,0.2], vec![0.2,0.8]];
        let p = perception::bayesian_filter(&[0.5,0.5], 0, &l, &t);
        assert!((p.iter().sum::<f64>()-1.0).abs()<1e-10); assert!(p[0]>p[1]);
    }
    #[test] fn test_post_entropy() { assert!((perception::posterior_entropy(&[0.5,0.5]) - 0.6931).abs() < 0.01); }
    #[test] fn test_confidence_uniform() { assert!(perception::confidence(&[0.5,0.5]).abs() < 0.01); }
    #[test] fn test_confidence_certain() { assert!((perception::confidence(&[1.0,0.0]) - 1.0).abs() < 0.01); }
    #[test] fn test_belief_div_same() { assert!(perception::belief_divergence(&[0.5,0.5], &[0.5,0.5]).abs() < 1e-10); }
    #[test] fn test_predict_belief() {
        let t = vec![vec![0.9,0.1], vec![0.1,0.9]];
        let p = perception::predict_belief(&[1.0,0.0], &t);
        assert!((p[0]-0.9).abs()<1e-10);
    }
    #[test] fn test_uniform_belief() { let b = perception::uniform_belief(3); assert!((b.iter().sum::<f64>()-1.0).abs()<1e-10); }
    #[test] fn test_marginal_obs() {
        let m = perception::marginal_observation(&[0.5,0.5], &vec![vec![0.9,0.1], vec![0.1,0.9]]);
        assert!((m[0]-0.5).abs()<1e-10);
    }
    #[test] fn test_state_dist_sum() {
        let l = vec![vec![0.8,0.2], vec![0.3,0.7]];
        let e = perception::estimate_state(0, &l, &[0.6,0.4]);
        assert!((e.state_distribution.iter().sum::<f64>()-1.0).abs()<1e-10);
    }
    #[test] fn test_belief_div_diff() { assert!(perception::belief_divergence(&[0.9,0.1], &[0.5,0.5]) > 0.0); }
    #[test] fn test_bayes_multi() {
        let l = vec![vec![0.9,0.1], vec![0.1,0.9]];
        let t = vec![vec![0.7,0.3], vec![0.3,0.7]];
        let mut b = vec![0.5,0.5];
        for _ in 0..5 { b = perception::bayesian_filter(&b, 0, &l, &t); }
        assert!(b[0] > 0.5);
    }
}
