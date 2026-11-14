# Analysis (Tasks 7.1 to 7.6)

All numbers come from the logs in `results/`. "PPO" means **run 2** (value-head bias initialised to the mean return of the first rollout). Run 1 (bias 0) is kept as an ablation. All PPO/DPO/reward-model settings are the ones fixed in the assignment. Every number comes from a single run (one seed), so small differences should not be over-read.

**Final comparison (Table 3: 300 held-out posts, greedy decoding)**

| Policy | Mean r_phi | Win rate vs. SFT | KL to SFT | Length | ROUGE-L |
|---|---|---|---|---|---|
| SFT | 4.606 | - | 0.000 | 30.5 | 0.2534 |
| PPO | 6.145 | 0.853 | 5.277 | 33.3 | 0.2109 |
| DPO | 3.603 | 0.380 | 3.098 | 45.2 | 0.2307 |

**Human evaluation (Table 4: 30 posts, blind, random order)**

| | SFT picked | PPO picked | DPO picked |
|---|---|---|---|
| Member 1 | 6 | 15 | 9 |
| Member 2 | 10 | 14 | 6 |

Agreement between members: 8 / 30 (with three options, chance is about 10 / 30). Member 1 is the author; Member 2 is a friend (not a member of the project) who completed the same blind form.

## 7.1 Reward-model accuracy and its errors

Pairwise accuracy on the 1,000 validation pairs is **64.7%** (chance 50%; 52.8% before training). Across the ten evaluation checkpoints validation accuracy fluctuated between 59.9% and 64.7%, so the last digit is noisy.

The errors are not random. Accuracy depends strongly on which summary is longer:

| Pair type (length in tokens) | n | RM accuracy |
|---|---|---|
| chosen longer than rejected | 559 | 78.2% |
| same length | 42 | 64.3% |
| chosen shorter than rejected | 399 | 45.9% |

When the human-preferred summary is the shorter one, the reward model is below chance, so it partly rewards length and detail. A rule that always picks the longer summary would be right on 55.9% of these pairs (ties counted as wrong; about 58% if ties are split evenly), so the model learned something beyond length, but not much.

Three confident mistakes (the reward model scored the rejected summary higher; these are the most confident errors, not typical ones):

1. Post about relatives and grandparents. Chosen (r = 4.46): "My hater grandparents are getting old; my Mom is now demanding that we be respectful of their age and not upset them." Rejected (r = 8.07): "Over the years my family has been incredibly cruel towards my homophobic grandfather. Recently Mom wants us all to stop sharing details of our lives with them [...]. I'm pissed." The rejected summary is longer and more detailed.
2. Throwaway post about isolation. Chosen (r = 3.58): "I don't have good friends and I'm losing my mind. I've been depressed for most of my life [...]." Rejected (r = 7.16): "[30F] Cut family out of life and have strained relationship with siblings. I don't know where to go for support or who to talk to." The rejected summary has a demographic tag and concrete-sounding facts. From the tail of the post I cannot tell which is more faithful.
3. Post about supporting a girlfriend. Chosen (r = 3.41): "How can I be there for my girlfriend in hard times without physically being there?" Rejected (r = 6.73): "Girlfriend has severe depression and anxiety. I try to be there for her, but she gets panic attacks [...]. I don't know what I can do to be more supportive." Again longer and more detailed, and it contains details I could not confirm from the part of the post I saw.

Common pattern: the reward model prefers longer, more detailed summaries (including ones with tags like "[30F]"), while the human labellers sometimes preferred the shorter one.

## 7.2 What PPO did, and is a higher reward a better summary?

**Score.** On training rollouts the mean r_phi rose from 3.6 (iteration 1) to about 5.9 (iteration 50) and 6.7 (mean of the last 10 iterations). On the 300 held-out posts (greedy) it rose from 4.61 (SFT) to 6.15, and PPO scored higher than SFT on 85.3% of posts.

**Length.** In run 2 the mean length stayed between about 25 and 39 tokens. In run 1 (value-head bias 0) it collapsed to 7.6 tokens around iteration 100 (reward dropped to 4.4) and then recovered to about 21. On the evaluation posts PPO is slightly longer than SFT (33.3 vs 30.5).

**KL.** The sampled KL to the reference (summed over the summary, temperature 1) grew to about 30-50 in run 1 and about 30-42 in run 2 after iteration 100. The KL on the greedy evaluation summaries is only 5.3, so the two numbers measure different things (greedy decoding picks the most likely tokens and avoids the word-salad that sampling produced).

**Is a higher reward-model score a better summary? Only partly.**

- *For:* humans picked PPO most often (29 of 60 picks, against 16 for SFT and 15 for DPO), and PPO beats SFT on 85% of posts under the reward model.
- *Against:*
  - (a) In training, the sampled summaries turned into ungrammatical word salad after roughly iteration 90 while the reward stayed at 6-7.
  - (b) ROUGE-L against the human reference fell from 0.2534 to 0.2109.
  - (c) PPO ends 85% of its summaries with a question, against 19% for SFT and for DPO. This habit looks learned from the reward model (I did not check it against the human references).
  - (d) In the qualitative samples PPO sometimes adds details that I could not confirm from the post (sample 2), and it gets the highest score (r = 7.52) in sample 7, where all three summaries get the story wrong.
  - (e) The reward model's top-scored summary was the one a human picked on only 15/30 (Member 1) and 13/30 (Member 2) posts, where chance is 10, i.e. about 1.9 and 1.2 standard deviations above chance. On 6 posts a human picked PPO although the reward model ranked SFT or DPO first.
- *Conclusion:* the reward model carries some signal (PPO really is better than SFT on many posts), but it can be exploited, so a higher score does not mean a better summary. The human evaluation is small (30 posts) and the two raters agreed on only 8/30, so it shows a trend, not a firm result.

## 7.3 Role of beta_KL (reasoning, not run)

Per Task 4.3 the per-token reward is r_t = -beta_KL (log pi_old - log pi_ref) + 1[t = T] r_phi. The first term charges the policy for every token on which it moves away from the SFT reference; summed over a summary it is -beta_KL times the sequence KL. With beta_KL = 0 only the terminal reward-model score is left, so nothing keeps the policy near SFT. I would expect the reward to rise faster, the KL to grow without bound, and the text to degrade and exploit the reward model's quirks (length, trailing questions, specific-sounding details).

Even with beta_KL = 0.05 the leash was weak: at a sequence KL of about 35 the penalty is about 0.05 x 35 = 1.75, while the reward gained was about +3.1 (3.6 to 6.7), so drifting stayed profitable on average. Run 1 showed the failure mode directly: sampled KL up to about 50 and incoherent samples.

## 7.4 Clip fraction

In run 2 the clip fraction (share of generated tokens whose ratio left [0.8, 1.2], averaged over the 4 epochs) was between 0.009 and 0.065 in the iterations printed every 5 steps after iteration 1, so each rollout moved the policy only a little. In iteration 1 it was 0.18, and it was the same in run 1 (0.182 vs 0.183), so it is not caused by the value-head initialisation. A possible reason is the first optimiser steps (Adam's first updates move every parameter by roughly the learning rate); I did not investigate this. In run 1 the clip fraction reached 0.27-0.39 around iterations 90-100, exactly when the summary length collapsed, i.e. large, unstable updates.

Correctness checks: in iteration 1 the KL to the reference was 0.000000. In the first epoch of **every** iteration (all 200 of run 2) the maximum |ratio - 1| was 0.00e+00 and the clip fraction was 0.

The only difference between run 1 and run 2 is the initial value-head bias (same prompts and sampling seeds). Run 1 had a value loss of 15-28 in the early iterations against 2-6 for run 2, so a badly initialised value head is a plausible cause of the collapse. I have one run of each, so this is consistent with the evidence but not proven.

## 7.5 Where the reward went in DPO

DPO starts from the KL-constrained objective of Task 4.3, whose optimal policy is pi*(y|x) = pi_ref(y|x) exp(r(x,y)/beta) / Z(x) (standard DPO derivation, Eqs. 55-58). Solving for the reward gives r(x,y) = beta log(pi*(y|x) / pi_ref(y|x)) + beta log Z(x). The reward is therefore expressed through the policy itself: the "implicit reward" r_hat(x,y) = beta log(pi_theta(y|x) / pi_ref(y|x)). Z(x) is the sum over all sequences y of pi_ref(y|x) exp(r(x,y)/beta), which is intractable, but it depends only on the prompt x. The Bradley-Terry loss uses only r(x, y_w) - r(x, y_l), and the same Z(x) appears in both terms and cancels. So Z(x) never has to be computed and no reward model is trained. Sanity check: the first-step loss was 0.69315 = log 2.

**Accuracy on the same 1,000 validation pairs:** reward model 64.7%, DPO implicit reward 60.6% (60.9% at step 250, then between 55.6% and 60.9%).

**Agreement between the two.** They give the same verdict on 779 / 1,000 pairs: both right on 516, both wrong on 263, only the reward model right on 131, only DPO right on 90. If their errors were independent I would expect about 392 both right and 139 both wrong, so the two models mostly fail on the same pairs.

**Same length bias.** DPO's implicit reward shows the same pattern as the reward model:

| Pair type | n | RM accuracy | DPO implicit accuracy |
|---|---|---|---|
| chosen longer | 559 | 78.2% | 73.3% |
| same length | 42 | 64.3% | 69.0% |
| chosen shorter | 399 | 45.9% | 41.9% |

In these pairs the human-chosen summary is the longer one 55.9% of the time and the shorter one 39.9% of the time, so the preference data itself leans towards longer summaries, and both methods can pick up this shortcut. DPO's generated summaries are also long (below), which fits this, but I did not test it.

**DPO's generations.** DPO's summaries get a lower reward-model score than SFT (3.60 vs 4.61, win rate 38%) and are much longer (45.2 tokens; 207 of 300 hit the 48-token cap, against 34 for SFT and 9 for PPO). Capped summaries score low (mean r = 2.15) while the 93 DPO summaries that finished score 6.83 (for SFT: 5.01 for finished, 1.44 for capped). These are different posts, so this is not a clean comparison. A possible reason is that the reward model only saw summaries ending with an end-of-sequence token in training, so cut-off summaries are out of distribution for it (a hypothesis I did not test). Humans picked DPO about as often as SFT (15 vs 16 picks), so the reward model and the humans disagree about DPO vs SFT.

## 7.6 Memory, time and choice of method

Networks in GPU memory (135M parameters, about 0.54 GB each in fp32; I did not measure peak memory):

- **PPO:** 3 full networks (trainable policy with optimiser state, frozen reference, frozen reward model) plus a small linear value head, and the activations of sampling rollouts.
- **DPO:** 2 networks (trainable policy with optimiser state, frozen reference).

| Stage | Time | Hardware |
|---|---|---|
| SFT (1,250 steps) | 63.0 min | Colab T4 |
| Reward model (2,500 steps) | 102.3 min | Colab T4 |
| PPO run 2 (200 iterations; run 1: 39.7 min) | 43.5 min | Kaggle T4 |
| DPO (2,500 steps) | 146.1 min | Kaggle GPU |

Both PPO and DPO start from SFT. PPO additionally needs the reward model first, so its end-to-end cost (102.3 + 43.5 = 145.8 min) is about the same as DPO's 146.1 min here. PPO also uses far fewer distinct training examples (3,200 prompts versus 20,000 preference pairs).

**Which method for a new project?** Evidence from this assignment: PPO gave the best reward-model score and was picked most by the humans, but it needed a careful value-head initialisation (run 1 collapsed), showed reward hacking when sampling at temperature 1, and needs rollout code and a trained reward model. DPO needs only preference pairs, one fewer network and no rollouts, and it trained without instability; its problem here was longer, often truncated summaries. I would start with DPO for a new project because it is simpler, cheaper and more stable, and use PPO when a reliable reward model exists and there is compute and time for monitoring. In this particular experiment PPO was the better method on both the automatic and the human metrics, so this choice is about robustness and cost, not about the numbers.

## Limitations

- One seed per run; only 30 human-evaluated posts and low agreement (8/30).
- The reward model is only 64.7% accurate and length-biased, and PPO optimises it directly.
- The per-step SFT loss was lost to a Colab disconnect, so Figure 1 uses the 6 evaluation points reconstructed from the console output.
- SFT ran with the library's default bf16 weights; the reward model, PPO and DPO ran in fp32.
- Several explanations above (length bias in DPO, truncation, Adam's first steps, value-head initialisation) are hypotheses that I did not test with extra runs.
