# Project Proposal

## Title

**Reinforcement Learning for Adaptive Traffic Signal Control under Nonstationary Traffic Demand**

## Team

**Team Black Swan**

Members: Zichao Zhang, Shawn Wu, Tyler Tao, Jinghan Lei

## Abstract

Adaptive traffic signal control is a natural sequential decision-making problem because each signal decision affects not only immediate vehicle departures but also future queue growth, cumulative waiting time, and subsequent control choices. In this project, we study a single-intersection traffic light control task under stochastic and nonstationary traffic demand and formulate it as a finite-horizon Markov Decision Process. At each time step, the agent observes traffic conditions, including queue lengths, the current signal phase, and phase duration, and chooses whether to keep the current phase or switch to the alternative phase. We build a lightweight simulator with stochastic arrivals, phase-dependent departures, and switching losses, and we compare a reinforcement learning controller against strong non-learning baselines, including fixed-cycle and queue-based heuristic policies. Our primary objective is not to propose a novel RL algorithm, but to conduct a controlled empirical study of when RL can match or outperform traditional rule-based control, particularly under asymmetric and nonstationary traffic patterns. We expect RL to perform competitively in stationary settings and to show clearer advantages when demand changes over time.

## 1. Introduction

Traffic signal control has long been approached with fixed-cycle schedules and hand-designed heuristics. These methods are often effective when traffic conditions are stable and predictable, but they can degrade when flow becomes asymmetric, bursty, or time-varying. In such settings, a controller must continuously balance short-term and long-term tradeoffs: serving one direction immediately may worsen congestion elsewhere in the near future, while frequent switching may reduce local queues at the cost of lost service time during signal transitions.

This project investigates whether reinforcement learning can provide a more adaptive and robust control strategy than conventional baselines in a simplified but meaningful traffic setting. We focus on a single intersection with two signal phases. This scope is deliberate. A single-intersection environment is complex enough to capture delayed rewards, switching costs, and long-horizon policy tradeoffs, yet simple enough to support systematic experimentation, strong baselines, and interpretable analysis within the time frame of a course project.

The broader motivation for this work is not to claim that RL universally dominates traditional traffic control, but rather to identify the regimes in which it is genuinely useful. In particular, we want to understand whether RL offers an advantage when traffic demand is nonstationary, directionally imbalanced, or subject to sudden bursts that are difficult to address with a single hand-crafted rule.

## 2. Research Questions

The project is organized around the following research questions:

1. Can an RL-based controller match or exceed strong heuristic baselines under stationary traffic demand?
2. Does RL provide greater robustness than fixed-cycle and queue-based control when traffic demand becomes asymmetric or nonstationary?
3. How sensitive is the learned policy to reward design, especially queue-based versus waiting-time-based objectives?
4. How much state information is necessary for effective control, and does adding features such as arrival-rate summaries or phase duration materially improve performance?

These questions frame the project as a controlled empirical study rather than a pure implementation exercise.

## 3. Problem Formulation

We formulate the adaptive traffic signal control task as a finite-horizon Markov Decision Process (MDP). At each time step `t`, the environment represents the current intersection status, the controller selects an action, and the system evolves according to stochastic vehicle arrivals and phase-dependent departures.

### 3.1 State

The default state representation includes:

- queue lengths for the north, south, east, and west approaches
- the current signal phase
- the duration of the current phase
- recent or current arrival-rate summaries for each direction

Formally, a state can be written as:

`s_t = (q_t^N, q_t^S, q_t^E, q_t^W, p_t, d_t, a_t^N, a_t^S, a_t^E, a_t^W)`

where `q_t^i` denotes the queue length for direction `i`, `p_t` is the current phase, `d_t` is the elapsed duration of that phase, and `a_t^i` summarizes traffic arrivals. An expanded variant may additionally include waiting-time information for ablation studies.

### 3.2 Action Space

We use a discrete two-action control scheme:

- `keep`: maintain the current signal phase
- `switch`: change to the other signal phase

The intersection is modeled with two phases:

- Phase 0: north-south green
- Phase 1: east-west green

This action space is intentionally simple. It captures the essential control decision while keeping the learning problem well aligned with a DQN-style approach.

### 3.3 Transition Dynamics

The transition function is governed by four main components:

- stochastic vehicle arrivals for each direction
- service capacity for directions that currently have green
- queue updates from arrivals and departures
- a switching penalty that models yellow-light or clearance loss

At a high level, each queue evolves according to:

`q_(t+1)^i = q_t^i + arrivals_t^i - departures_t^i`

Arrivals are sampled from configurable stochastic processes, while departures depend on the active phase and a maximum per-step service capacity. If the controller chooses to switch phases, the environment incurs a temporary loss of service to reflect the real operational cost of switching too frequently.

### 3.4 Reward

Our default reward is based on congestion:

`r_t = - sum_i q_t^i - alpha * 1[switch]`

This reward penalizes total queue length and discourages unnecessary switching. We also plan to evaluate an alternative reward based on incremental waiting time in order to examine how reward shaping changes learned behavior.

## 4. Environment Design

Rather than relying on real-world traffic data, we construct a lightweight simulator. This choice is intentional and appropriate for the project scope. Our focus is on sequential control rather than traffic prediction, and a synthetic simulator allows us to generate controlled traffic regimes that isolate the strengths and weaknesses of different control strategies.

Each episode corresponds to a fixed-horizon control process, for example 200 time steps with 3 seconds per step. The simulator includes:

- Poisson arrivals for each direction
- configurable piecewise traffic regimes
- a departure model with a maximum service rate per green phase
- a switching-loss mechanism to model clearance time

The use of synthetic but structured traffic demand enables us to evaluate not only average performance but also generalization across qualitatively different conditions. This is especially important for studying robustness.

## 5. Methods

### 5.1 Non-Learning Baselines

To ensure that the evaluation is meaningful, we compare RL against strong baselines rather than weak strawman policies.

The first baseline is **fixed-cycle control**, which switches phases after a predetermined duration. This represents a standard classical approach and provides an important reference point.

The second baseline is a **queue-threshold heuristic**, which switches phases when the opposing direction becomes substantially more congested than the currently served direction. This is a simple but often effective adaptive strategy.

The third baseline is a **max-pressure style heuristic**, which gives priority to the direction with higher aggregate queue pressure once a minimum green duration has been satisfied. This baseline is stronger than fixed-cycle control and serves as a more demanding benchmark for RL.

### 5.2 Reinforcement Learning Method

Our primary learning-based method is **Deep Q-Networks (DQN)**. DQN is appropriate because the action space is small and discrete, and the state representation can be encoded as a compact vector. The Q-network takes the traffic state as input and predicts Q-values for the two actions, `keep` and `switch`.

The model is trained using:

- experience replay
- a target network
- epsilon-greedy exploration

The optimization target follows the standard Bellman update:

`y_t = r_t + gamma * max_a' Q_target(s_(t+1), a')`

We choose DQN for the main study because it is conceptually well matched to the environment and easier to interpret than more complex alternatives. If time permits, we may add Double DQN or PPO as an optional extension, but this is not essential to the core proposal.

## 6. Experimental Plan

### 6.1 Training Setting

We will first train the RL controller in a stationary or mildly varying traffic environment in order to verify that the learning setup is stable. This phase is intended to establish that the agent can learn sensible phase-control behavior before being evaluated in more challenging regimes.

### 6.2 Evaluation Regimes

We plan to test all methods across several traffic regimes:

- **Symmetric low traffic**: light and balanced demand from all directions
- **Symmetric high traffic**: heavy but balanced demand
- **Asymmetric traffic**: one axis carries substantially more traffic than the other
- **Nonstationary traffic**: traffic intensity changes over the course of the episode
- **Burst traffic**: one direction experiences a temporary demand spike

These regimes are designed to reveal not only average-case performance but also the robustness of each controller when the environment shifts.

### 6.3 Evaluation Metrics

We will report metrics that are operationally meaningful and easy to interpret:

- average waiting time
- average queue length
- maximum queue length
- throughput
- number of phase switches

Among these, average waiting time and average queue length will be the primary outcome measures in the final analysis because they best capture practical congestion quality.

### 6.4 Ablation Studies

To better understand what drives performance, we will conduct the following ablations:

1. **Reward design**: queue-based reward versus waiting-time-based reward
2. **State representation**: minimal state versus enriched state
3. **Generalization**: training on stationary traffic and testing on nonstationary traffic
4. **Switching cost sensitivity**: varying the switch penalty to study aggressive versus conservative behavior

These ablations will help us distinguish whether any gains come from the RL framework itself or from specific design choices in the environment and objective.

## 7. Expected Contributions

The main contribution of this project is a controlled empirical analysis of RL for traffic signal control in a manageable yet meaningful setting. More specifically, we aim to contribute:

- a clean and extensible single-intersection simulator for adaptive traffic control
- a suite of representative stationary and nonstationary traffic regimes
- a comparison between RL and strong rule-based baselines
- an analysis of how reward design and state representation influence learned policies

The novelty of the project does not lie in inventing a new RL algorithm. Instead, it lies in building a coherent experimental framework and using it to answer a clear control question: under what traffic conditions does RL offer a meaningful advantage over strong hand-designed control rules?

## 8. Expected Outcomes and Risks

Our expectation is that RL will be competitive with strong heuristic baselines under stationary traffic and may show more noticeable gains under asymmetric or nonstationary demand. At the same time, we do not assume that RL will dominate in every regime. In some simpler settings, heuristics may remain competitive or even preferable because of their simplicity and stability. That possibility is part of the scientific value of the project.

The main project risk is over-expanding the environment. A multi-intersection network would introduce much greater state complexity, coordination challenges, and longer training times. To manage that risk, we define the single-intersection setup as the core deliverable and treat any larger network as an optional extension only if the main pipeline is already stable.

## 9. Timeline

### Week 1

- implement the simulator
- build fixed-cycle and heuristic baselines
- validate transition logic and metrics

### Week 2

- implement DQN
- train on stationary traffic
- verify that the learned policy is reasonable

### Week 3

- evaluate across all traffic regimes
- generate plots and tables
- begin ablation experiments

### Week 4

- complete ablations
- analyze learned policy behavior
- finalize the report and presentation

## 10. Team Responsibilities

To keep development parallel and balanced across the group, we propose the following division of work:

- **Zichao Zhang:** environment design, simulator implementation, and integration of the overall project pipeline
- **Shawn Wu:** baseline controller implementation, traffic-regime construction, and simulator validation
- **Tyler Tao:** RL implementation, DQN training pipeline, and hyperparameter tuning
- **Jinghan Lei:** evaluation, visualization, report writing, and presentation preparation

This split allows the project to progress in parallel while keeping ownership of each component clear.

## 11. Optional Extension

If the single-intersection system is implemented and validated early, we may extend the study to a small `2x2` network of intersections. However, this extension will remain secondary to the main goal of thoroughly evaluating the single-intersection problem.

## 12. Conclusion

This project proposes a focused study of reinforcement learning for adaptive traffic signal control under changing traffic demand. The problem is well suited to RL because it involves delayed consequences, sequential tradeoffs, and the need for dynamic adaptation. At the same time, it is scoped carefully enough to be feasible within a course-project timeline. By combining a controlled simulator, strong non-learning baselines, and multiple traffic regimes, the project aims to produce a clear and credible empirical answer to when RL is genuinely useful for traffic signal control.
