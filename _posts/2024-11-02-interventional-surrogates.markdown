---
layout: post
title: "Interventionally Consistent Surrogates for ABMs"
date: 2024-11-02
last_modified_at: 2024-11-02
categories: [Recent Papers]
---

This blog post outlines a recent paper which I worked on with my fantastic collaborators Joel Dyer, Fabio Zennaro, Yorgos Felekis and Theo Damoulas. This work is set to be published at NeurIPS 2024 - check out the full paper [here](https://arxiv.org/abs/2312.11158)!

## ABMs and the Need for Surrogates

As micro-data becomes increasingly available, agent-based modelling becomes an increasingly attractive framework to model and analyse complex systems. Since agent-based models simulate systems at the agent level, they allow us to they allow us to investigate exceptionally fine-grained policy interventions. Additionally, unlike machine learning models such as neural networks, one can readily incorporate domain knowledge that may not be represented explicitly in data.

Unfortunately ABMs are often computationally expensive to run and difficult to integrate into machine learning pipelines. Moreover, they often produce complex individual-level outputs that are difficult to analyse. In practice, we often want to reason about a complex system at the macro-level, in terms of its emergent properties.

This motivates us to pursue surrogate models which 
- are computationally cheap to run
- easy to integrate into existing ML pipelines
- and reason at the macro-level.

## ABMs as Causal Models
An ABM implcitly defines a structural causal model (SCM) that specifies how agents interact with each other and the environment. Unfortunately, we rarely have explicit access to this SCM, and even if we did, it would be tremendously large. This is in part why we encounter the issues described above with ABMs, and why reasoning causally about ABMs is difficult. Our lives would be made far easier if we had access to a simple causal surrogate model that is interventionally consistent with the original ABM. This line of thought leads naturally to causal abstraction theory!

## Causal Abstraction Theory
Causal abstraction theory aims to relate two structural causal models defined over different sets of variables. There are many frameworks for causal abstraction. In this work we use $$\tau$$-$$\omega$$ abstraction. Investigating the formal connections between different abstraction frameworks constitutes and interesting research direction, and you should check another of Fabio's papers if you are curious about it! 

Consider two SCMs; a base model $$\mathcal{M}$$, and an abstract model $$\mathcal{M}^{\prime}$$. You can think of $$\mathcal{M}$$ as an ABM and $$\mathcal{M}^{\prime}$$ as a simpler surrogate model. A $$(\tau, \omega)$$-abstraction is a pair of maps $$\tau$$ and $$\omega$$. $$\tau$$ maps variable assignments in the base model $$\mathcal{M}$$ to variable assignments in the abstract model $$\mathcal{M}^{\prime}$$. For instance, consider an epidemilogical ABM that tracks the infection status of citizens through time via a set of binary variables, and a simple ODE simulator that tracks only the total number of infected individuals through time. In this case, it is natural to let $$\tau$$ simply count the number of infected individuals in the ABM on each time step. Note that this gives rise to a corresponding variable assignment in the ODE model. Generally speaking, $$\tau$$ should map the microstates of the ABM to macroscopic emergent properties that an expert would like to reason in terms of. We will assume from hereon that $$\tau$$ is some fixed map that a domain expert has chosen.

In contrast, $$\omega$$ maps inteverntions in the base model to interventions in the abstract model. Consider our previous epidemiological example. Our ABM may allow us to perform lockdown interventions to limit disease spread. In contrast,  it is unclear what a lockdown intervention would look like in the ODE model as it does not model at the individual level. $$\omega$$ is responsible for telling us what intervention in the ODE model corresponds to a lockdown. Unlike, $$\tau$$, it may be difficult for a domain expert to naturally specify an $$\omega$$ map. We will come back to this issue soon!

## Interventional Consistency

What makes a good abstraction? Say we want to observe a macroscopic output associated with an intervention $$\iota$$ in the ABM. We have two options. We could map $$\iota$$ to its corresponding abstract intervention $$\omega(\iota)$$ and apply it in the surrogate $$\mathcal{M}^{\prime}$$ to obtain a macroscopic output directly. Contrastingly, we could simulate $$\iota$$ directly via $$\mathcal{M}$$ to obtain a microscopic output and lift it to the macroscopic level via $$\tau$$. Logically, we want both these procedures to produce the same output. In other words, abstracting then abstracting should be the same as intervening then abstracting! more formally, we want the following commutative diagram to hold for any possible base intervention $$\iota$$!

![Average Effort Over Time](/assets/images/IC_Error_diagram-6-1.png)

In the diagram above, $$\mathbb{P}_{\mathcal{M}_{\iota}}$$ and $$\mathbb{P}_{\mathcal{M}^{\prime}_{\omega(\iota)}}$$ correspond to the interventional distributions associated with $$\iota$$ and $$\omega(\iota)$$ respectively. Commutativity is quite a stringent goal and one we are unlikely to achieve exactly for every intervention $\iota$. That is, whichever surrogate model and intervention map $$\omega$$ we choose, the interventional distribution attained by intervening then abstracting may differ from the interventional distribution attained by abstracting then intervening. Our goal should be to minimise the distance between these distributions as much as possible!

![Average Effort Over Time](/assets/images/IC_Error_gap_diagram-1.png)

This naturally leads use to the definition of abstraction error:

$$
    d_{\tau, \omega}(\mathcal{M}, \mathcal{M}^{\prime}) = \mathbb{E}_{\iota \sim \eta} \left[ d\left(\tau_{\#} (\mathbb{P}_{\mathcal{M}_{\iota}}) ,\, \mathbb{P}_{\mathcal{M}^{\prime}_{\omega(\iota)}}\right)\right].
$$

Here, $\eta$ is a distribution over base interventions, whilst $d$ is a divergence between probability distributions. In words, the abstraction error measures the average difference between intervening and abstracting versus intervening then abstracting. The abstraction error depends heavily on the interventional distribution $\eta$, which should be carefully chosen by a domain expert and reflects the interventions they care most about.

## Learning an Abstraction

## Some Experiments