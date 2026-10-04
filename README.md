# Pokémon Red Deep Reinforcement Learning

A deep reinforcement learning project that trains an agent to interact with **Pokémon Red** through the **PyBoy** Game Boy emulator.

This repository currently contains my **V1 recreation of the baseline RL environment and training pipeline**. The goal of this stage is to understand and reproduce the existing approach before experimenting with improvements for long-horizon game progression.

## Project Goal

The long-term goal is to build an RL agent capable completing the game.

The current V1 system combines:

- **PyBoy** — Game Boy emulation and game interaction
- **Gymnasium** — custom reinforcement learning environment interface
- **Stable-Baselines3** — PPO implementation and training
- **PyTorch** — neural-network backend
- **CNN-based visual observations** — processed game frames
- **Parallel environments** — multiple game instances can generate experience for one shared policy
- **RAM-based game-state information** — Pokémon's internal memory is used for reward/state tracking
- **Exploration rewards** — V1 uses screen-based KNN exploration

## Architecture

The basic interaction loop is:


                    PPO Agent
                        │
                      action
                        │
                        ▼
                ┌────────────────┐
                │  RedGymEnv     │
                └───────┬────────┘
                        │
                     PyBoy
                        │
                        ▼
                 Pokémon Red
                        │
                ┌───────┴────────┐
                │                │
             screen           game RAM
                │                │
                └───────┬────────┘
                        ▼
                  observation
                        +
                      reward
                        │
                        ▼
                    PPO update


### Observation

The V1 environment processes the original Game Boy screen from approximately:

144 × 160 × 3


to:


36 × 40 × 3


It keeps a history of three recent frames and combines this visual information with additional exploration/reward memory into the model observation:


128 × 40 × 3


### Action space

The default action space contains six controller actions:


0 → DOWN
1 → LEFT
2 → RIGHT
3 → UP
4 → A
5 → B


These abstract RL actions are translated into PyBoy `WindowEvent` inputs.

### Exploration

The V1 baseline uses a screen-based exploration mechanism:


current screen
      ↓
36 × 40 × 3
      ↓
flatten → 4320-dimensional vector
      ↓
HNSW / KNN similarity search
      ↓
exploration reward


The environment can also use coordinate-based exploration, controlled by its configuration.

## Repository Structure


pokemon-red-rl/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── has_pokedex_nballs.state
│
├── assets/
│   └── ...
│
└── baselines/
    ├── red_gym_env.py
    ├── memory_addresses.py
    ├── run_pretrained_interactive.py
    ├── run_baseline_parallel_fast.py
    ├── tensorboard_callback.py
    └── ...


Generated files such as training sessions, videos, logs, checkpoints, and emulator output should not be committed unless specifically needed as small demonstration artifacts.

The Pokémon Red ROM itself is **not included** in this repository. A legally obtained ROM must be supplied locally.

## Installation

Python 3.10+ is recommended.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The project also requires **FFmpeg** for video-related functionality used by the environment.

## ROM Setup

Place your legally obtained Pokémon Red ROM in the repository root:


pokemon-red-rl/
├── PokemonRed.gb
├── has_pokedex_nballs.state
├── baselines/
└── ...


The V1 configuration expects the ROM at:


../PokemonRed.gb


when running scripts from the `baselines/` directory.

## Running the Pretrained Agent

From the `baselines/` directory:

```bash
cd baselines
python run_pretrained_interactive.py
```

This runs the pretrained policy interactively through PyBoy.

The controls and exact configuration depend on the script and environment setup.

## Training

The V1 training script uses Stable-Baselines3 PPO together with multiple subprocess environments.

The baseline configuration creates multiple independent Pokémon Red environments and uses their experience to train a shared PPO policy.

Run:

```bash
cd baselines
python run_baseline_parallel_fast.py
```

### Parallel training concept


             Shared PPO Policy
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
   PyBoy Env 1   PyBoy Env 2   PyBoy Env 3   ...
       │            │            │
       └────────────┼────────────┘
                    ▼
                experience
                    │
                    ▼
                PPO update


The number of environments and training parameters can be adjusted in the training configuration.

## Tracking Training

The training pipeline records information about agent/game progress.

TensorBoard can be used to inspect training metrics:

```bash
tensorboard --logdir <session_directory>
```

Then open the local TensorBoard address shown in the terminal.

For development, it is useful to track metrics such as:

- episode reward
- exploration
- Pokémon levels
- health
- deaths
- map/event progression
- training throughput

## Current Stage: V1 Recreation

The current objective is to **recreate and understand the V1 baseline implementation**, rather than immediately change the algorithm.

The implementation is being studied component-by-component:

1. `red_gym_env.py` — environment and RL interface
2. `memory_addresses.py` — Pokémon RAM addresses and game-state extraction
3. `run_pretrained_interactive.py` — running a trained policy
4. `run_baseline_parallel_fast.py` — PPO training and parallel environments
5. `tensorboard_callback.py` — training metrics and monitoring

After reproducing the baseline, the next stage is to analyze its failure modes and compare it with the V2 approach.

## Planned Development


V1 baseline
   ↓
Reproduce
   ↓
Validate
   ↓
Analyze failure modes
   ↓
Study V2
   ↓
Reproduce V2
   ↓
Compare V1 vs V2
   ↓
Develop custom improvements
   ↓
Evaluate long-horizon progression


Potential future directions include:

- improved exploration
- reward shaping
- anti-loop mechanisms
- better temporal memory
- curriculum learning
- action abstraction
- hierarchical RL
- larger-scale parallel training
- improved evaluation and diagnostics

These are future experiments and are not part of the current V1 implementation.

## Why This Project

Pokémon Red is a challenging RL environment because useful progress can require long sequences of actions, exploration, combat, resource management, and game-event progression.

The project is therefore intended as a practical study of:

- deep reinforcement learning
- PPO
- visual observations
- custom Gymnasium environments
- emulator integration
- reward engineering
- parallel environment training
- long-horizon decision making


This project is a recreation/study of an existing Pokémon Red reinforcement learning implementation and is intended to understand, reproduce, and eventually extend the underlying ideas.

Original project:

**PWhiddy/PokemonRedExperiments**  
https://github.com/PWhiddy/PokemonRedExperiments

The original repository is licensed under the MIT License. See `LICENSE` for the applicable license and attribution requirements.

## Disclaimer

Pokémon and related intellectual property belong to their respective owners.

This repository does not include the Pokémon Red ROM. Users are responsible for obtaining and using game files legally.
