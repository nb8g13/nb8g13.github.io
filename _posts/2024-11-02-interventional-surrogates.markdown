---
layout: post
title: "Interventionally Consistent Surrogates for Agent-based Simulators"
date: 2024-11-02
last_modified_at: 2024-11-02
categories: [Recent Papers]
---

This blog post outlines a recent paper which I worked on with my fantastic collaborators Joel Dyer, Fabio Zennaro, Yorgos Felekis and Theo Damoulas. This work is set to be published at NeurIPS 2024 - check out the full paper [here](https://arxiv.org/abs/2312.11158)!

## ABMs and the Need for Surrogates

As micro-data becomes increasingly available, agent-based modelling is become an increasingly attractive framework to model and analyse complex systems. Since agent-based models simulate systems at the agent level, they allow us to they allow us to investigate exceptionally fine-grained policy interventions. Additionally, unlike machine learning models such as neural networks, one can readily incorporate domain knowledge that may not be represented explicitly in data.

Unfortunately ABMs are often computationally expensive to run and difficult to integrate into machine learning pipelines. Moreover, they produce complex outputs, describing the behaviour of each agent within the system, that are difficult to analyse. In practice, we often want to reason about a complex system at the macro-level, in terms of its emergent properties.

This motivates us to pursue surrogate models which 
- are computationally cheap to run
- easy to integrate into existing ML pipelines
- and reason at the macro-level, in terms of emrgent properties.

## ABMs as Causal Models
An ABM implcitly defines a structural causal model (SCM) that specifies how agents interact with each other and the environment. Unfortunately, we rarely have explicit access to this SCM, and even if we did, it would be tremendously large. This is, in part, why we typically encounter the issues described above with ABMs, and why reasoning causally about ABMs is difficult. Our lives would be made far easier if we had access to a simple causal surrogate model that is interventionally consistent with the original ABM. This line of thought leads naturally to causal abstraction theory!

## Causal Abstraction Theory
Causal abstraction theory aims to relate two structural causal models defined over different sets of variables. There are many frameworks for causal abstraction. In this work we will use $\tau$-$\omega$ abstraction. Investigating the formal connections between different abstraction frameworks constitutes and interesting research direction, and you should check another of Fabio's papers if you are curious about it! 

Consider two SCMs; a base model $\mathcal{M}$, and an abstract model $\mathcal{M}^{\prime}$. You can think of $\mathcal{M}$ as an ABM and $\mathcal{M}^{\prime}$ as a simpler surrogate model. A $(\tau, \omega)$-abstraction is a pair of maps $\tau$ and $\omega$. $\tau$ maps variables assignments in the base model $\mathcal{M}$ to variable assignments in the abstract model $\mathcal{M}^{\prime}$. For instance, consider an epidemilogical ABM that tracks the infection status of citizens through time via a set of binary variables, and a simple ODE simulator that tracks only the total number of infected individuals through time. In this case, it is natural to let $\tau$ simply count the number of infected individuals in the ABM on each time step. Note that this gives rise to a corresponding variable assignment in the ODE model. Generally speaking, $\tau$ should map the microstates of the ABM to macroscopic emergent properties that an expert would like to reason in terms of. We will assume from hereon that $\tau$ is some fixed map that a domain expert has chosen.

In contrast, $\omega$ maps inteverntions in the base model to interventions in the abstract model. Consider, our previous epidemiological example. Our ABM may allow us to perform lockdown intevrentions to limit disease spread. Since our ODE does not model at the individual level it is unclear what a lockdown intervention should look like at the abstract level. $\omega$ is responsible for telling us what intervention in the ODE model corresponds to a lockdown. Unlike, $\tau$, it may be difficult for a domain expert to naturally specify an $\omega$ map. We will come back to this issue soon!

## Interventional Consistency

What makes a good abstraction? Say we want to observe macroscopic output associated with an intervention $\iota$ for the ABM. We have two options. We could map $\iota$ to its corresponding abstract intervention $\omega(\iota)$ and apply it in the surrogate $\mathcal{M}^{\prime}$ to obtain a macroscopic output directly. Contrastingly, we could simulate $\iota$ directly via $\mathcal{M}$ to obtain a microscopic output which we can then lift to the macroscopic level via $\tau$. Logically, we want both these procedures to produce the same output. In other words, we want the following commutative diagram to hold for any possible base intervention $\iota$!

![Average Effort Over Time](/assets/images/IC_Error_diagram-4-1.png)

<object data="{{ site.url }}{{ site.baseurl }}/assets/images/Algebra_I_Reference_Sheet.pdf" width="1000" height="1000" type="IC_EE"></object>