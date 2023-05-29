# VanillaWorld

[![Build Status](https://github.com/inspirai/VanillaWorld.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/inspirai/VanillaWorld.jl/actions/workflows/CI.yml?query=branch%3Amain)

<div align="center">
  <p>
  <img src="./docs/asserts/img/logo.png" width="320px">
  </p>
  
</div>

> - **EMERGENCE** is when simple mechanics interact to create complex situations.
> - Leveraging emergence means crafting mechanics that don’t just add together, but **MULTIPLY** into a rich universe of possibility.
> - **ELEGANCE** happens when mechanics interact in complex, non-obvious ways.
> 
> -- [Designing Games](https://tynansylvester.com/book/)


This project is our attempt to build a research platform for exploring how to create a self-evolving system. More specifically, we are interested in the following three game systems:

- ❤️ Survival
- 🔨 Craft
- 🗡️ Combat


## Motivation

- Most existing environments are pretty **SLOW**.
- The number of agents in most existing environments is limited.
- Most existing environments contain only one game mechanic. Instead, we want to explore different mechanics and study when and how **EMERGENCE** appears with the latest RL algorithms.

## Key Features

- GPU Native
  - The environments are written in pure Julia code. By utilizing
    KernelAbstractions.jl, the environments can be executed on many different
    devices (on both CPU and GPU). The best efficiency is achieved when policy
    and environment are executed on the same accelerator.
  - With the help of DLPack, the internal observation of environments can be
    shared across different Deep Learning platforms.  Extensive examples are
    provided to help users understand how to use these environments in different
    Reinforcement Learning packages.

- Massive Agents
  - Several builtin environments demonstrate that millions of agents can be
    executed simultaneously on modern accelerators.

- Visualization
  - A grid-based interactive GUI is provided by default for each environment.
    Recording wrappers are also provided to help analysis and debug policies.

- Composability
  - Thanks to multiple dispatch provided by Julia. We can easily create new
    environments by reusing existing components as much as possible.
  
## Resources

Talks:

- TODO: Add link to slides.

Demo Videos

- TODO: Add links

## Related Environments

- [Crafter](https://github.com/danijar/crafter)
  - Single agent
  - Though it contains survival, craft and combat systems, it mainly focuses on the craft part.

- [NeuralMMO](https://neuralmmo.github.io/)
  - The latest version has many game mechanics we want, however, it runs pretty slow and it is not that easy to extend. 

- [Gym-MiniGrid](https://github.com/Farama-Foundation/gym-minigrid)
  - Grid based, close to our design but many environments provided focus on exploration only.
  - Number of agents are limited.

- [Lux](https://github.com/Lux-AI-Challenge/Lux-Design-2022)
  - A specific large scale multi agent environment focusing on cooperation and competition.

- [gymnax](https://github.com/RobertTLange/gymnax) & [brax](https://github.com/google/brax)
  - Implemented in JAX, native GPU/TPU support. 

## Acknowledgement

- The assets used in the demo video are from [1-Bit Pack](https://kenney.nl/assets/1-bit-pack) by [Kenney](https://twitter.com/KenneyNL).