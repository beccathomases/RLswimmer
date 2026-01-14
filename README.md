# RLswimmer (MATLAB)

Small sandbox for experimenting with **reinforcement learning** on a toy
low-Reynolds swimmer model.

Right now everything is **self-contained in MATLAB** with a simple
placeholder “physics” model. The idea is:

- State = discrete paddle angle levels.
- Action = small changes to those angle levels.
- Reward = a proxy for forward swimming (for now).
- RL algorithm = tabular Q-learning or a linear Q-network.

Later, the placeholder reward in `swimmer_step.m` can be replaced by a
call to a **regularized Stokeslet solver** or other fluid solver.

---

## 1. Idea in one paragraph


We consider a 2–paddle swimmer. Each paddle has an angle index
$s_i \in \{-5,\dots,5\}$ and angle $\theta_i = s_i \cdot (\pi/20)$.


- A **state** is the pair \((s_1,s_2)\).
- An **action** is a small change in each index
  \((\Delta s_1,\Delta s_2) \in \{-1,0,1\}^2 \setminus \{(0,0)\}\).
- The environment applies the action, clamps the indices to the allowed
  range, converts to angles, and computes a **reward** that is meant to
  approximate “forward motion minus effort”.

This makes a small, finite MDP that is easy to explore with tabular
methods and simple function approximation.

---

## 2. File structure

Current layout:

- `matlab/`
  - `encode_state.m`, `decode_state.m`  
    Base-11 style mapping between a 2D state \((s_1,s_2)\) and a single
    integer state index in `1..121`.
  - `featurize_state.m`  
    Converts a state index into a small feature vector `phi` (currently
    `[s1; s2]` or similar) for the linear Q-network.
  - `swimmer_step.m`  
    One environment step:
    - decodes the current state,
    - applies an action `a = [d1 d2]` in index space with clamping,
    - converts indices to angles,
    - computes a **placeholder reward** based on phase difference and a
      simple angle penalty,
    - returns the next state index and reward.
    This is the place to plug in a real fluid solver later.
  - `train_swimmer_tabular.m`  
    Tabular Q-learning over all discrete states and actions.
    - Builds the state and action sets.
    - Runs episodes for a fixed horizon `Tmax`.
    - Uses an ε-greedy policy over a `Q(nStates, nActions)` table.
    - Records episode returns.
  - `qUpdateLinear.m`  
    Q-learning update for a **linear Q-network**:
    \[
      Q(s,a) = W(a,:) \,\phi(s) + b(a)
    \]
    where `phi` is the feature vector from `featurize_state`.
  - `train_swimmer_linearQ.m`  
    Same environment as the tabular version, but uses a linear Q-network
    instead of a table:
    - State → features via `featurize_state`.
    - Q-values via `W*phi + b`.
    - ε-greedy over the actions.
    - Updates `W`, `b` using `qUpdateLinear`.
  - `compare_tabular_vs_linear.m`  
    Convenience script that runs both trainers and plots episode returns
    side by side.

- `docs/`  
  Text/design notes for the project (e.g. MDP definition, action set,
  future connection to a fluid solver).



---

## 3. How to run the current experiments

From MATLAB:

```matlab
cd('path/to/RLswimmer/matlab');   % adjust as needed
```

### 3.1 Tabular Q-learning

```matlab
returns_tab = train_swimmer_tabular(500, 40, true);
```

Arguments:

* `nEpisodes` (default 500)
* `Tmax` steps per episode (default 40)
* `doPlot` whether to plot episode returns

This trains a tabular Q-function over the discrete state/action space
and will print occasional summaries to the command window. If
`doPlot == true`, it produces a simple return-vs-episode plot.

### 3.2 Linear Q-network

```matlab
returns_lin = train_swimmer_linearQ(500, 40, true);
```

Same interface, but:

* Q-values are represented by a linear model on top of the features
  `phi(s)`.
* `qUpdateLinear` does the TD update on `W` and `b`.

### 3.3 Compare the two

```matlab
compare_tabular_vs_linear
```

Runs both trainers (with default settings) and plots the episode returns
for tabular vs linear Q-learning on the same figure.

---

## 4. Placeholder physics vs real fluid solver

Right now the “physics” in `swimmer_step.m` is intentionally simple:

```matlab
phase_diff    = theta2 - theta1;
forward_proxy = 0.1 * sin(phase_diff);
angle_penalty = 0.01 * (theta1^2 + theta2^2);
r = forward_proxy - angle_penalty;
```

The long-term plan is to:

1. Replace this block with a call to a **regularized Stokeslet solver**:

   * build kinematics from `(theta1, theta2)`,
   * simulate a short stroke segment,
   * compute forward displacement (and maybe an energetic cost).
2. Keep the RL code (state/action logic, Q-learning loops) unchanged.

So the RL machinery is already set up; only the reward model needs to be
swapped once the fluid solver is wired in.

---

## 5. Possible next steps

* Extend from 2 paddles to 3 paddles (`N=3`, `3^N - 1 = 26` actions).
* Try different feature maps in `featurize_state.m`
  (angles, sines/cosines, interaction terms).
* Add episode termination conditions based on time or energy.
* Connect `swimmer_step.m` to a real fluid simulation and compare the
  learned “good strokes” with known optimal strokes.


