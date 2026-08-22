---
layout: post
title: "RL for LLMs"
date: 2026-08-22 09:00:00-0600
description:
tags:
related_posts: false
---

In this post I try to summarise my own review of different methods for Reinforcement Learning for LLM post-training, particularly focused on algorithms used for reasoning.
### Introduction
Reinforcement Learning has become one of the most impactful methodologies in the post-training of large language models (LLMs). Its role has evolved substantially over time: initially applied narrowly to align model outputs with human preferences via Reinforcement Learning from Human Feedback (RLHF), it has more recently emerged as the principal tool for inducing sophisticated reasoning capabilities. The release of OpenAI's o1 and, most influentially, DeepSeek-R1, demonstrated that RL with verifiable rewards can cause models to develop structured, multi-step reasoning behaviours as an _emergent_ property of the training process itself — not merely as a learned imitation of human demonstrations. This has reframed reasoning as a capability that can be explicitly trained and scaled, opening a new axis of improvement orthogonal to parameter scaling and data scaling during pre-training.

The landscape of RL for LLMs can be broadly divided into two historical phases. The first, spanning roughly 2022–2024, was dominated by RLHF and its variants, with Proximal Policy Optimization (PPO) as the de facto optimiser. PPO was introduced by Schulman et al. (2017) and first applied to LLM alignment by Ziegler et al. (2020) and Ouyang et al. (2022) in the InstructGPT paper, which laid out the three-stage recipe — supervised fine-tuning (SFT), reward model training, and PPO — that became the standard approach for producing instruction-following models such as ChatGPT.

The second phase began with DeepSeek's introduction of Group Relative Policy Optimization (GRPO) and its application to reasoning tasks via Reinforcement Learning with Verifiable Rewards (RLVR). GRPO eliminates the critic network that PPO relies upon, replacing it with a group-relative Monte Carlo baseline. This dramatically reduces memory and compute overhead at the cost of introducing a structural bias, and has become the dominant algorithm in the current reasoning-model literature. However, as this review discusses in depth, both PPO and GRPO share a fundamental algorithmic flaw — they estimate the value of the _old_ policy rather than the _current_ policy — that limits their theoretical soundness and motivates new research directions.
### 2. The RL framework for LLMs

Here is how RL maps onto autoregressive text generation:

- **Policy $\pi\_\theta​$**: the LLM itself, which defines a probability distribution over tokens given a context.
- **State $s\_t$​**: the full context at timestep $t$, i.e., the prompt concatenated with all tokens generated so far: $s\_t=[q\_1,…,q\_m,o\_1,…,o\_{t−1}]$ where $q\_i$ is a context/prompt token, and $o\_i$ is a token generated so far.
- **Action $a\_t$​**: the next token to be generated, drawn from $\pi\_\theta(\cdot  \mid  s\_t)$.
- **Transition function P(S'  \mid  s,a)**: the transition probability function, representing the probability of transitioning  to state s′ when taking action a in state s. This is deterministic in standard text-generating LLMs. **However**, In general, the environment dynamics can be more complex. For example, models with tool access have to call the actual tool and receive the environment feedback in context or even modify their environment such as the cases of SWE and Web agents.
- **Trajectory $\tau$**: the complete sequence of states and actions from prompt to end-of-sequence token.
- **Reward $R$**: a scalar score assigned to the trajectory, typically at the terminal step.
- **Value function $V^\pi(s\_t)$**: the expected cumulative reward when starting in state $s\_t$​ and acting according to policy $\pi$.
- **Advantage $A(s\_t,a\_t)=Q(s\_t,a\_t)−V(s\_t)$**: how much better action $a\_t$ is in state $s\_t$ relative to the policy's average performance from that state.

There are two standard MDP formulations. In the (contextual) **bandit formulation**, the entire completion is a single action and the model receives one scalar reward. In the **MDP (token-level) formulation**, each token is a separate action with its own state, enabling per-token credit assignment. PPO uses the MDP formulation; GRPO and most critic-free methods operate at the response level, broadcasting a single scalar advantage to every token in the completion.
#### 2.1 The RL Objective
The objective is to maximise the expected cumulative reward:
							$J(\pi\_\theta) = \mathbb{E}\_{\tau \sim \pi\_\theta}[R(\tau)]$
To apply gradient ascent, we use the policy gradient theorem:
							$\nabla\_\theta J(\pi\_\theta)=\mathbb{E}\_{\tau \sim \pi\_\theta}\left[\sum\_t \nabla\_\theta \log \pi\_\theta(a\_t  \mid  s\_t) \cdot \psi\_t \right]$

where $\psi\_t$ represents a step-wise weight. Different choices for this weight distinguish the different algorithms used. The key insight is that this gradient requires no differentiation through the reward function or through the discrete sampling process — it only requires log-probabilities of actions under the current policy, making it applicable even when the reward is a non-differentiable external verifier.
### 3. The Training Pipeline

Modern LLM training proceeds in distinct stages:
- **1. Pre-training**: Large-scale next-token prediction over internet-scale corpora. This builds the model's knowledge base.
- **2. Supervised Fine-Tuning (SFT)**: Training on a curated set of high-quality prompt–response pairs. This teaches the model appropriate formatting, instruction-following style, and in the reasoning context can be used to provide the model with chain-of-thought demonstrations before RL is used to enhance reasoning capabilities.
- **3. Reward Model Training**: A separate model (typically initialised from the SFT model with its final layer replaced by a scalar regression head) is trained to predict human preference scores. This model is trained _before_ RL and held fixed during RL training. It provides the terminal reward signal used to score complete responses. This step may be skipped if there is a clear verifier-based reward.
- **4. RL Fine-Tuning**: The policy (the current LLM) is optimised using a policy gradient algorithm against the reward signal from the reward model (RLHF) or from rule-based verifiers (RLVR). A reference model (the SFT checkpoint) is held frozen and the final model is kept close to it (via a KL divergence penalty) to prevent reward hacking.

A critical distinction that is often conflated in practice is the difference between the **reward model** and the **critic**:
- The **reward model** estimates the quality of a _complete_ response. It is trained before RL and remains frozen. It answers the question: "How good is this finished output?"
- The **critic** (or value function) estimates the expected future reward from an _intermediate_ state. It is trained _during_ RL and updated alongside the policy. It answers the question: "Given the tokens generated so far, how likely is this partial completion to ultimately receive a high reward?"

PPO employs all four models simultaneously: the policy, the reference model, the critic, and the reward model. This is the primary source of PPO's memory overhead and motivated the development of critic-free alternatives.

### 4. Policy Gradient methods: Theoretical Foundation

#### 4.1. REINFORCE
The simplest policy gradient algorithm, REINFORCE, sets $\psi\_t = R(\tau)$, the total trajectory reward. Thus its update rule (averaged over a batch of N trajectories) is:

						$\nabla\_\theta J(\pi\_\theta) \approx \frac{1}{N}\sum\_{i=1}^N \sum\_{t=0}^T \nabla\_\theta \log \pi\_\theta(a\_t \mid s\_t) \cdot R(\tau\_i)$

The intuition is that if a trajectory received a high reward, we should increase the probability of all actions taken. However, there are two main issues with this approach. First, it uses the total trajectory reward, meaning past rewards influence updates to future actions, resulting in a violation of causality (e.g. the reward at time t=0 impacts the gradient of $\pi\_\theta(\cdot \mid s\_5$). Secondly, each step of the trajectory $t=0-T$ is updated in the same way, and thus this update rule does not consider per-step credit assignment. We can correct thus by using the **reward-to-go** formulation, that is $G\_t = \sum\_{k=t}^T r\_k$.

The REINFORCE estimator has high variance, and thus one can introduce a non-action dependent baseline $b(s\_t)$. A natural choice for this baseline is $V^{\pi\_\theta}(s\_t)$, which results in $\psi\_t = G\_t-V^{\pi\_\theta}(s\_t) = A\_t$ , that is the advantage function. This is the foundation of all actor-critic methods. The key challenge is that $V$ must itself be estimated.

#### 4.2. Importance Sampling
When the data is generated by a different policy from the one being optimised, the estimator must be corrected by importance sampling weights:
				$\nabla\_\theta J(\pi\_\theta)=\mathbb{E}\_{\tau \sim \pi\_{\theta\_\text{old}}} \left[ \sum\_t \frac{\pi\_\theta(a\_t \mid s\_t)}{\pi\_{\theta\_\text{old}}(a\_t \mid s\_t)} A^{\pi\_{\theta\_\text{old}}}(s\_t,a\_t) \nabla\_\theta \log \pi\_\theta(a\_t \mid s\_t)\right]$
PPO uses a clipped version of the importance sampling ratio $\frac{\pi\_\theta(a\_t \mid s\_t)}{\pi\_{\theta\_\text{old}}(a\_t \mid s\_t)}$. Crucially, it only applies this correction to the policy learning update above, and does not apply such a correction when learning the critic/value function.

### 5. PPO
Proximal Policy Optimization (Schulman et al., 2017) improves over vanilla policy gradients along three dimensions: better advantage estimation via GAE, actor-critic architecture, and constrained policy updates via clipping or KL penalties.

**Generalised Advantage Estimation (GAE).** Rather than using the raw Monte Carlo return $G\_t$ to estimate the advantage through $A\_t = G\_t - V(s\_t)$​, PPO estimates the advantage using a TD-$\lambda$ style interpolation. The per-step TD error is:
$\delta\_t= r\_t + \gamma V(s\_{t+1})-V(s\_t)$ 
and the GAE advantage is estimated as:
							$\hat{A}\_t(\gamma,\lambda)=\sum\_{l=0}^\infty (\gamma \lambda)^l \delta\_{t+l}$

This represents a forward-view estimation of the advantage, which uses the critic estimate. The critic itself is learnt by regressing towards the target $V^\text{target}(s\_t) = \hat{A}\_t + V(s\_t)$, which is equivalent to a standard $\lambda$-return $G\_t^\lambda = (1-\lambda) \sum\_{n=1}^\infty \lambda^{n-1} G\_{t}^{(n)}$. 
#### 5.1. Algorithm walk-through
In the LLM setting (Ziegler et al., 2020; Ouyang et al., 2022), PPO maintains four model instances simultaneously: the policy $\pi\_\theta$, the reference model $\pi\_{ref}$ (the frozen model after SFT), the critic $V\_\phi$ (which scores intermediate completions) and the reward model $r$ (which scores complete responses). The training loop proceeds as follows:

**Step 1 - Rollout generation**
A batch of prompts ${q}$ is sampled from a dataset, and the current policy $\pi\_{\theta\_\text{old}}$ is used to generate completions ${o}$ autoregressively. Once generation of these is complete, the reward model $r$ scores the full completion, giving a terminal reward $R=r(q,o)$. 

**Step 2 - Reward shaping with Reference Model KL**
A per-token KL divergence penalty is subtracted from the terminal reward to prevent reward hacking. In most implementations this is distributed across tokens, giving a per-token reward of:
$r\_t = \cases{R-\beta \log \frac{\pi\_{\theta\_\text{old}}(a\_t \mid s\_t)}{\pi\_{ref}(a\_t \mid s\_t)} \text{.    if.  } t=T \text{(terminal token)}\\ - \beta \log \frac{\pi\_{\theta\_\text{old}}(a\_t \mid s\_t)}{\pi\_{ref}(a\_t \mid s\_t)} \text{.    otherwise}}$

Note that this is a constraint against the **frozen SFT model** not against the immediately preceding policy: it enforces long-range fidelity to the original model's language behaviour.

**Step 3 - Advantage estimation with GAE**
The GAE is calculated as explained above. Typical values are $\gamma \approx 1$ and $\lambda \approx 0.95$ in the LLM setting (**need to check**). The key limitation is that GAE requires summing over all future timesteps from $t$, meaning the complete rollout must be available before any advantage can be computed. This is what makes PPO non-incremental despite using TD errors as building blocks.

**Step 4 - Clipped Policy Objective**
To prevent large policy updates which lead to instabilities, PPO uses a clipped "importance sampling" ratio that discourages large changes in policy:
					$\mathcal{L}(\pi\_\theta) = \mathbb{E}\left[ \min \left( \rho\_t(\theta)\hat{A}\_t,clip(\rho\_t(\theta),1-\epsilon,1+\epsilon)\hat{A}\_t\right)\right]$
where $\rho\_t(\theta)=\frac{\pi\_\theta(a\_t \mid s\_t)}{\pi\_{\theta\_\text{old}}(a\_t \mid s\_t)}$. 
Typically $\epsilon=0.2$. 

This clipping is the **trust-region KL** constraint in disguise: it enforces that $\pi\_{\theta}$ cannot deviate far from $\pi\_{\theta\_{old}}$ within a single update step. This is different from the KL against the reference model, which enforces fidelity to the original post-SFT model across the entire training process.

An alternative formulation (PPO-KLPEN) instead adds a KL divergence penalty $D\_{KL}(\pi\_{\theta\_{old}}(\cdot \mid s\_t) \mid  \mid  \pi\_\theta(\cdot \mid s\_t))$ directly to the loss instead of clipping the ratio between the policies, and the weight of this KL divergence on this loss is controlled by a hyperparameter $\beta$ which is updated adaptively. Note this KL divergence is a mean-seeking KL, where $\pi\_\theta$ is "forced" to cover all the modes of P.

**Step 5 - Value Learning**
The critic is trained to regress towards the $\lambda$-return, which can be shown to be equivalent to:
$V^\text{target}(s\_t) = \hat{A}\_t + V(s\_t)$. Thus we learn the critic by minimizing the loss:
					$\mathcal{L}(\phi)=\left(V\_\phi(s\_t)-sg[V^{target}(s\_t)]\right)^2$
where $sg$ denotes a stop-gradient which acts over the value function estimate $V(s\_t)$ (the advantage is a scalar).

**Pseudocode detail: Multi-epoch updates**
A key feature of PPO is that it reuses each rollout batch across multiple gradient epochs (typically 2-4). This happens for both the critic and the actor. First, rollouts are obtained (data collection) according to the current policy. Then, for each of the 2-4 epochs, we cycle through minibatches to update both the actor and the critic with each minibatch. This is why the policy clipping in PPO is so important: by the 2nd epoch, $\pi\_\theta$ has already been updated away from $\pi\_old$, the policy used to collect the data and thus used to estimate the critic. As discussed later on, this multi-epoch reuse means the critic's regression target — computed from $π\_{\theta\_\text{old}}$​​ rollouts with no IS correction on the value loss — progressively estimates the wrong value function. It should be noted that regardless, PPO only uses the IS ratio for the actor for 1-step, even though it multiplies it by an advantage calcualted through GAE that depends on all future steps, and thus should depend on the entire IS ratio of the trajectory-to-go.
#### 5.2 PPO's Structural Limitation: Non-Incrementality
Despite using TD errors as its building block, PPO is **not incremental** in the spirit of TD learning. GAE is a _forward view_: it sums future TD errors out to the horizon. The algorithm must therefore wait for the complete rollout before computing any advantage or performing any update. This makes PPO — and all current LLM RL algorithms — subject to the same scalability bottleneck: the requirement for full sequential text generation before any parameter update. The original motivation for TD learning — updating online, before the episode ends — is not realised in the LLM context.
### 6. Critic-free Methods: ReMax, RLOO, GRPO and REINFORCE++

Many PPO implementations learn a critic model which is an entirely separate model (effectively a full copy of the LLM for the value head), leading to a very high memory cost. While current implementations such as the one in the transformers RL library use a value head on top of the fixed LLM model (thus not requiring a copy), the field didn't necessarily pay much attention to this and instead aimed to create **critic-free** methods. These methods estimate the advantage entirely from sampled rewards, without learning $V(s\_t)$. All of them work at the **response/outcome** level, that is they simply assign a scalar advantage to the entire completion, and broadcast it uniformly to every token. Thus, there is no per-token credit assignment, and the entire RL framework works as a contextual bandit (even if the LLM field does not acknowledge it). 
#### 6.1. ReMax
ReMax (Li et al, 2023) eliminates the value function by using a **greedy rollout baseline**. The full algorithm for one prompt $q$ is:
1. Sample completion $o \sim \pi\_\theta (\cdot \mid q)$.
2. Generate a greedy completion by doing $\hat{o} \sim \arg \max \pi\_\theta (\cdot \mid s\_t)$ at each completion step.
3. Score both completions with a reward model $r(o)$ and $r(\hat{o})$. 
4. Compute the advantage and apply it uniformly to every token of $o$ such that $\hat{A}\_t = r(o)- r(\hat{o}), \forall t \in o$.
5. Update the policy via $\nabla\_\theta J(\theta) = \mathbb{E}\left[ \sum\_t \hat{A}\_t \nabla\_\theta \log \pi\_\theta(a\_t \mid s\_t)\right]$
Thus the reward of the greedy completion serves as a cheap proxy for $V^{\pi\_\theta}(q)$. Because greedy decoding uses the _current_ policy weights, ReMax is fully on-policy. However, it is still a single-sample estimate and therefore high variance. ReMax requires only two forward passes per prompt, making it the most computationally efficient of the group-sampling alternatives.

#### 6.2. RLOO (REINFORCE Leave-One-Out)
RLOO (Ahmadian et al.,2024) samples G completions $(o\_1,...,o\_G)$ from $\pi\_{old}$ for each prompt $q$. The key insight here is that a naive group mean is a biased estimate of the adantage becuase completion $o\_i$'s reward is included in its own baseline. The leave-one-out construction fixes this. For each completion $o\_i$, the baseline is the mean reward of the other $G-1$ completions such that:
					$\hat{A}\_i = r\_i - \frac{1}{G-1}\sum\_{j \neq i}r\_j$
Through algebrain manipulation, this is equivalent to:
					$\hat{A}\_i =\frac{G}{G-1} (r\_i - \bar{r})$
where $\bar{r} = \frac{1}{G} \sum\_{j=1}^G r\_j$.
This advantage is then **broadcast uniformly** to every token of $o\_i$.
Note that $\bar{r}$ is an unbiased sample estimate of $V^{\pi\_\text{old}}$, so RLOO — like GRPO — estimates the old policy's value, not the current policy's value. At large $G$, RLOO and GRPO converge to the same estimator (**I dont get why if it's biased**); the difference lies in whether the std normalisation is applied and whether $o\_i$ is included in its own baseline.

#### 6.3. GRPO 
GRPO (Shao et al., 2024) is the algorithm that powered DeepSeek-R1 and has become the dominant RL algorithm in the current reasoning literature. The full algorithm proceeds as follows.

**Rollout**. For each prompt $q$, we sample a group of $G$ completions from the old policy. We score each with the reward model (or a rule-based verifier) such that $r\_i = r(q,o\_i)$.

**Advantage computation.** Normalise by the full group statistics:
						$\hat{A}\_{i,t} = \frac{r\_i - mean(r)}{std(r)}, \forall t$
Thus the same scalar is broadcast to every token $t$ of completion $o\_i$.

**Policy update**. Define the probability ratio $\rho\_{i,t}(\theta) = \frac{\pi\_\theta(o\_{i,t} \mid q,o\_{i<t})}{\pi\_{\theta\_\text{old}}(o\_{i,t} \mid q,o\_{i<t})}$. The GRPO loss is:
$\mathcal{L}(\theta) = -\frac{1}{G}\sum\_{i=1}^G \frac{1}{ \mid o\_i \mid }\sum\_{t=1}^{ \mid o\_i \mid }\left[\min \left(\rho\_{i,t}\hat{A}\_{i},clip(\rho\_{i,t},1-\epsilon,1+\epsilon)\hat{A}\_i \right) - \beta \cdot D\_{KL}[\pi\_{\theta}(\cdot \mid s\_{i,t}) \mid  \mid \pi\_{ref}(\cdot \mid s\_{i,t})]\right]$

A few things to note:
1. The KL term is inside the loss, not the reward, unlike in PPO. In PPO, the drift KL is subtracted from the reward before computing advantages, and gradients do not pass through it. In GRPO, it is part of the differentiable loss, so the gradient of the KL estimator directly enters the parameter update. As Shah et al. (2026) show, this interacts poorly with biased KL estimators (specifically the K3 estimator used in most implementations), producing biased policy gradients.
2. **The std normalisation is theoretically weak.** Dividing by the group standard deviation has no direct grounding in the policy gradient theorem — it is a variance-reduction heuristic. As REINFORCE++ shows, it introduces a statistical dependency between numerator and denominator that makes the estimator formally biased.
3. As previously argued, there is no per-token credit assignment, and the group mean estimates $V^{\pi\_{old}}$, not $V^\pi$. 

GRPO's main advantage is computational: by eliminating the critic, and usually relying on rule-based verifiers to model the reward, it reduces the number of model copies from four (PPO) to two (policy and reference), with a corresponding memory saving. The cost is that high variance forces large groups, requiring many forward passes per prompt, and that it relies on a heavily biased estimation.
#### 6.4. REINFORCE++

REINFORCE++ (Hu et al., 2025) starts from an identification of three compounding flaws in GRPO's local (prompt-level) normalisation:

1. **Theoretical bias.** Because the numerator $r\_i - mean(r)$ and the denominator $std(r)$are both computed from the same small group of $G$ samples, they are statistically dependent. The advantage estimator is therefore a ratio of dependent random variables, which is biased. This bias is not asymptotic: it exists for any finite $G$.
2. **Numerical instability.** When $G$ is small (e.g. $G=4/8$) and all sampled completions receive similar rewards (which happens often on hard problems where the model always fails), $std(r) \rightarrow 0$ and the advantage explodes.
3. **Prompt-level overfitting.** Optimising to be "better than peers from the same prompt" is not the same as optimising to achieve globally high reward. The policy can overfit to prompt-specific patterns, performing well within the local group while failing to generalise.

**The fix: global advantage normalisation** REINFORCE++ retains the critic-free structure but normalises advantages across an entire training batch (based on different prompts) rather than per prompt group. The KL w.r.t. to the reference model is added to the reward itself, as in PPO.
In what follows, $k$ is the number of completions in the batch for each prompt.			

As the global batch size grows (e.g., $N=1024$), the batch mean and standard deviation converge to stable constants, making the estimator unbiased. The algorithm has two variants:
1. REINFORCE++ $(k=1)$: one sampled completion per prompt (achieves maximum prompt diversity). The advantage is just the reward minus the KL penalty, normalised globally. Best for RLHF.
2. REINFORCE++ with baseline $(k>1)$: for complex reasoning tasks where group sampling is beneficial, the group mean is first subtracted (local baseline subtraction for reward rescaling), and then the result is normalised by the _global_ standard deviation. The KL regularisation uses the K2 estimator in the loss (which has unbiased gradients for reverse KL), rather than GRPO's K3 estimator.

Empirically, REINFORCE++ achieves substantially better out-of-distribution generalisation than GRPO: when trained on a small set of AIME-24 problems, GRPO achieves near-perfect training accuracy but scores 0.0 on AIME-25, while REINFORCE++ generalises meaningfully. This suggests that GRPO's local normalisation does not merely add variance — it actively drives overfitting through the prompt-level objective mismatch identified above.

### 7. Direct preference Optimization (DPO)
DPO (Rafailov et al., 2023) represents a distinct paradigm: it bypasses the RL training loop entirely by observing that, under a KL-constrained RLHF objective, the optimal policy has a closed form that implies the reward can be expressed directly in terms of the policy. This derivation leads to a contrastive training objective over paired preference data (chosen responses $o\_w$ VS rejected responses $o\_l$ to a specific prompt $q$). The loss is:
							$\mathcal{L} = - \mathbb{E}\left[\log \sigma \left( \beta \log \frac{\pi\_\theta(o\_w \mid q)}{\pi\_{\theta\_{ref}}(o\_w \mid q)} - \beta \log \frac{\pi\_\theta(o\_l \mid q)}{\pi\_{\theta\_{ref}}(o\_l \mid q)}\right) \right]$
DPO can be optimised with standard gradient descent over a fixed, offline dataset, requiring no reward model, no critic, and no on-policy sampling. This makes it far cheaper and simpler to implement than PPO-based RLHF.

However, DPO has a structural limitation that is particularly relevant to reasoning: in its classic formulation, it operates only on **sequence-level, terminal rewards**. It treats the entire generated text as a single unit and cannot assign credit to specific intermediate steps or tokens. This makes it poorly suited to tasks where the quality of a response depends on its reasoning _process_ rather than just its final output.

### 8. Online vs Offline RL for LLMs
A central axis in the LLM alignment literature is the distinction between **online** and **offline** training algorithms. Online algorithms (PPO, GRPO) generate completions in real time from the current policy and use these on-policy samples for training. Offline algorithms (DPO, SFT over fixed datasets) train over pre-computed datasets with no real-time generation.

Empirical evidence consistently shows a **performance gap** in favour of online algorithms. The reason is intuitive: on-policy samples are always relevant to the current policy's distribution, whereas offline data can become stale as the policy changes. Several semi-online or "iterative" variants attempt to bridge this gap: rejection sampling (filtering on-policy completions with a reward model and performing SFT on the best ones), iterative DPO, and reward-weighted regression. These approaches recapture much of the benefit of online sampling while avoiding the full complexity of PPO-based RL.

Online RL has additional costs beyond compute. PPO in its full form requires four simultaneous model copies: the policy, the reference model, the critic, and the reward model. GRPO reduces this to two copies (policy and reference model, assuming rule-based verifiers), which is a large part of why it has displaced PPO in the reasoning literature.

### 9. The Emergence of Reasoning: RL and CoT
#### 9.1 What is Reasoning in the LLM Context?
In the context of LLMs, reasoning refers to the model's ability to produce structured intermediate steps before providing a final answer, a process commonly described as **chain-of-thought (CoT) reasoning**. Rather than mapping a prompt directly to an output, a reasoning model generates a sequence of intermediate statements or computations that illustrate how it arrives at its conclusion. Reasoning models are particularly suited to complex tasks such as mathematical proofs, competitive programming, multi-step logical puzzles, and scientific reasoning, but can often lead to overly verbose answers for simple factual queries.
#### 9.2 The DeepSeek-R1 Finding: Reasoning as an Emergent Property of RL
The most influential empirical finding in recent reasoning research is the emergence of reasoning behaviours from pure RL. **DeepSeek-R1-Zero** was trained exclusively with RL (no SFT stage) on top of the DeepSeek-V3 base model, using two types of verifiable reward signals: an accuracy reward (verified by compiler for code, deterministic checking for maths) and a format reward (ensuring outputs were placed within `<think>` tags). Despite not being explicitly trained to generate reasoning traces, the model spontaneously began producing them — the so-called **"Aha!" moment** in which the model starts self-correcting and reconsidering its approach mid-completion.

This finding is significant because it suggests that the capacity for reasoning may be latent in large pre-trained models, requiring only the right RL signal to be unlocked. However, this interpretation is contested. Recent work (e.g., _Understanding R1-Zero-Like Training_) suggests that the reasoning behaviours observed — including the "Aha" moment — may already be present in base models as a consequence of pre-training on large amounts of chain-of-thought data, with RL serving to _amplify and elicit_ rather than _create_ these capabilities. This is the "sharpening vs. discovery" debate in the RL-for-reasoning literature.

**DeepSeek-R1**, the flagship model, further refined R1-Zero with additional SFT stages and RL training. DeepSeek also released **DeepSeek-R1-Distill**, which performs knowledge distillation from R1 into smaller Llama and Qwen models via SFT on R1's outputs — demonstrating that reasoning capabilities can be transferred without RL at a fraction of the cost.
#### 9.3 The DeepSeek-R1 Pipeline
The R1 training pipeline proceeds in stages:
1. **Cold-start SFT** on a small set of long chain-of-thought demonstrations (to avoid the readability issues of R1-Zero's raw RL outputs).
2. **RLVR training** with GRPO and rule-based accuracy/format rewards.
3. **Rejection sampling SFT** to curate high-quality reasoning traces from the RL checkpoint.
4. **Final RL stage** combining reasoning and general instruction-following rewards.
#### 9.4 RL vs. SFT for Reasoning: Generalisation vs. Memorisation
A central question in the reasoning literature is whether RL genuinely improves a model's ability to _generalise_ its reasoning to out-of-distribution problems, or merely helps it _memorise_ patterns in the training distribution. Evidence from REINFORCE++ experiments (Hu et al., 2025) is instructive: GRPO trained on 30 AIME-24 problems achieves near-perfect training accuracy but completely fails on AIME-25, exhibiting catastrophic overfitting. REINFORCE++ with global normalisation, by contrast, learns more gradually but generalises substantially better. This suggests that certain implementation choices in RL algorithms — particularly how advantages are normalised — directly affect whether RL teaches reasoning or reward hacking.

### 10. Test-time compute scaling and its relation to RL
#### 10.1 Two Axes of Improvement
Reasoning capabilities in LLMs can be improved along two orthogonal axes:
- **Train-time compute**: increased RL training, SFT on reasoning demonstrations, or distillation.
- **Test-time compute (inference-time scaling)**: allocating more computation at inference time to produce better outputs.
These axes are complementary rather than mutually exclusive. Models like OpenAI's o1 and o3 likely employ both heavy train-time RL and explicit inference-time scaling, contributing to their cost relative to conventional models. According to OpenAI, o3 used approximately 10× more training compute than o1.
#### 10.2 Inference-Time Scaling Methods
Inference-time scaling methods improve reasoning without modifying model weights:
- **Chain-of-thought prompting**: including phrases like "think step by step" encourages the model to generate intermediate reasoning steps. This is the simplest form of inference-time scaling and is entirely prompt-based.
- **Majority voting / Best-of-N**: generating multiple independent completions and selecting the answer by majority vote or reward model score. This is a parallel method that improves accuracy at the cost of multiplying inference compute.
- **Beam search and lookahead search**: guided search over the generation tree, often combined with a process reward model to evaluate partial completions.
- **"Wait" token insertion** (Muennighoff et al., 2025): training a model to respond to special tokens that prompt it to continue reasoning before finalising an answer.

#### 10.3 DeepSeek-R1's Position on Inference-Time Scaling
Notably, the DeepSeek-R1 technical report explicitly categorises explicit inference-time scaling methods — including process reward model-based approaches and Monte Carlo Tree Search — as "unsuccessful attempts." DeepSeek-R1's reasoning behaviour emerges naturally from RL training rather than being imposed at inference time. The model's longer responses are an _implicit_ form of inference-time scaling (longer generation = more compute), but explicit test-time budgeting was not applied.
This stands in contrast to OpenAI's models, which are believed to employ explicit inference-time scaling. The distinction has practical significance: DeepSeek-R1's reasoning is baked into the model weights and runs at standard inference cost per token, while o1/o3-style inference-time scaling incurs additional cost proportional to the number of "thinking steps."
#### 10.4 The Relationship Between GRPO and Planning
There is a deep and underappreciated structural connection between GRPO's group rollouts and inference-time planning. GRPO samples $G$ completions per prompt, uses the textual environment to score each one, and derives a baseline from the group statistics. This is functionally equivalent to **sample-based Monte Carlo planning**: the LLM acts as its own world model, generating multiple possible futures and using their outcomes to evaluate the current state. The non-incrementality of this process — the requirement to wait for complete rollouts — is the direct analogue of Monte Carlo planning's cost in classical RL, and is the primary scalability bottleneck of current RL for LLMs.

### 11. The role of the critic in test-time compute methods

Currently much of RL for LLMs research focuses on outcome reward models, which simply evaluate a completion as correct or not (0/1). This often gets propagated equally to each token of the completion, but even if not, there is no per-token supervision obtained. 

A much stronger idea lies in Process Reward Models (PRMs) which provide supervision for each token/intermediate step. If we have access to such a model, we can improve the efficiency of the search process with the following steps:
1. Terminate a solution attempt that is not making progress, or is incorrect prior to reaching the  final answer. 
2. Reset the agent to any intermediate, previously visited, state that has a high likelihood of  success.
In fact, this reward supervision can be seen as a critic that estimates the probability that a particular intermediate state will lead to a solution $v\_\theta(s\_t) \rightarrow [0,1]$. Notice that with these two operations, and the general structure of language, we can implement any tree search procedure. This is the premise of several approaches (Yao et al., 2023; Hao et al., 2023; Zhou et al., 2024a).

### 12. Open Problems and Future Directions

Several open questions define the frontier of RL for LLMs:

**Process Rewards vs. Outcome Rewards.** Current algorithms predominantly use terminal outcome rewards. Process reward models (PRMs), which assign intermediate rewards to reasoning steps, offer finer-grained credit assignment but are harder to train. GRPO is structurally incompatible with PRMs without modification, because it broadcasts a single terminal reward uniformly across tokens. A per-token value function is naturally suited to PRMs and may be where critic-based approaches win most clearly.

**Incrementality and Scalability.** All current algorithms — GRPO, PPO, and their variants — require complete rollouts before any update. A genuinely incremental RL algorithm for LLMs, updating as tokens are generated rather than waiting for the full completion, would eliminate the primary scalability bottleneck and enable truly online learning. This remains an unsolved problem.

**RL's True Role: Sharpening vs. Discovery.** Whether RL for reasoning genuinely discovers new reasoning capabilities or merely amplifies latent abilities learned during pre-training is an open empirical question with significant implications for how much we should expect RL to improve models in data-limited regimes.

**Stability of Off-Policy Corrections.** The "deadly triad" of bootstrapping, function approximation, and off-policy data is known to cause divergence in classical RL. Whether this poses a practical problem in LLM post-training — where episodes are short and terminal-rewarded — is an open question. The stop-gradient bootstrapping and trust-region clip used in PPO may suffice; the answer likely depends on sequence length and the degree of policy staleness.

**Model-Based RL.** **Model-Based RL.** GRPO's group rollouts are implicitly a form of Monte Carlo planning: the LLM samples possible futures, scores them with the environment, and derives a baseline from the results. Explicit model-based RL would replace these expensive text rollouts with a learned world model that simulates trajectories more cheaply. In the classical latent-space planning tradition (e.g., MuZero), this simulation runs over compact learned representations rather than raw observations — but applying this idea to LLMs faces serious obstacles: transformer hidden states are not designed to support forward simulation without a full forward pass, and reward models typically require text input, making it unclear how to query them with a latent vector. The closest existing work either performs MCTS directly over token space (still expensive) or runs implicit reasoning steps in embedding space without decoding to text at intermediate steps (e.g., Coconut, Hao et al., 2024) — but neither constitutes a full model-based RL solution. This remains an early-stage and largely open direction.
## 13. References

- Espeholt et al. (2018). _IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures_ (V-trace). arXiv:1802.01561.
- Guo et al. (2025). _DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning_. arXiv:2501.12948.
- Hu et al. (2025). _REINFORCE++: Stabilizing Critic-Free Policy Optimization with Global Normalization_. arXiv:2501.03262.
- Jaech et al. (2024). _OpenAI o1 Technical Report_.
- Maei, H. (2011). _Gradient Temporal-Difference Learning Algorithms_ (PhD Thesis, University of Alberta).
- Munos et al. (2016). _Safe and Efficient Off-Policy Reinforcement Learning_ (Retrace). NeurIPS 2016.
- Ouyang et al. (2022). _Training Language Models to Follow Instructions with Human Feedback_ (InstructGPT). NeurIPS 2022.
- Precup, Sutton & Singh (2000). _Eligibility Traces for Off-Policy Policy Evaluation_. ICML 2000.
- Rafailov et al. (2023). _Direct Preference Optimization: Your Language Model is Secretly a Reward Model_. NeurIPS 2023.
- Schulman et al. (2015). _High-Dimensional Continuous Control Using Generalized Advantage Estimation_. arXiv:1506.02438.
- Schulman et al. (2017). _Proximal Policy Optimization Algorithms_. arXiv:1707.06347.
- Shah et al. (2026). _A Comedy of Estimators: KL Divergence Estimation in LLM RL_. arXiv:2512.21852.
- Shao et al. (2024). _DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models_ (GRPO). arXiv:2402.03300.
- Williams, R. J. (1992). _Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning_ (REINFORCE). Machine Learning.
- Zhang et al. (2025). _A Survey of Reinforcement Learning for Large Reasoning Models_. arXiv:2509.08827.
- Ziegler et al. (2020). _Fine-Tuning Language Models from Human Preferences_. arXiv:1909.08593.
