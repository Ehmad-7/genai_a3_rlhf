WHY = [
    # 1
    "SFT uses the whole 48-token budget and is cut off mid-sentence, while PPO gives a complete, shorter summary that keeps the main facts (excitement, inability to achieve an erection, lack of confidence) and ends with a question. PPO's wording is awkward and it drops the hooking-up context, so 'better' here means complete and on-topic, not well written. DPO also hits the cap and mostly restates SFT's idea.",
    # 2
    "SFT collapses into a repetition loop ('spoiled, spoiled, ...') until the cap, so any complete summary is better. PPO is fluent and ends with 'What do I do?', but it adds details (clothing, tanning salon, a sister) that I could not confirm from the post, so it should be checked against the full post. DPO is the most faithful here (it reuses the reference's 'acts like a toddler' idea) but is cut off at the cap.",
    # 3
    "SFT contradicts itself (a 'small apartment' with a yard) and is cut off at the cap. DPO and PPO both capture the core of the post (a cat hunting in the backyard, how to keep it away) and finish cleanly, in 42 and 32 tokens. DPO's closing clause 'and not scare me' slightly changes the meaning.",
    # 4
    "PPO's greedy decoding falls into a loop ('unknown address' repeated until the cap), which the reward model punishes (r=1.06). SFT gives a short, correct one-sentence summary (r=6.76) and DPO adds a plausible follow-up question. Likely a side effect of PPO drifting away from SFT, not verified.",
    # 5
    "SFT stays close to the post (it mentions being more assertive) and is a complete sentence. PPO produces a hostile, off-topic summary that is also cut off at the cap (r about 0), and DPO drifts into extra details and is cut off as well. This post contains strong language, so a less explicit example may be better for the blog.",
    # 6
    "SFT is almost identical in meaning to the reference (porn habits, 'Am I being too sensitive?'). DPO starts the same way but keeps going past a one-sentence summary, adds details and is cut mid-sentence, giving r=-0.19. PPO is acceptable but adds a claim ('he doesn't love me anymore') that I could not confirm from the post.",
    # 7
    "All three summaries get the story wrong. The reference says the author slept with her sister's future father-in-law. SFT says 'their father's mother', PPO says 'grooms father's mother', and DPO first says 'grooms father' (correct) and then adds a contradictory 'mother'. The reward model still gives PPO the highest score (7.52), which shows it does not check factual accuracy.",
    # 8
    "All three summaries are reasonable, so no model is clearly better. PPO is slightly more complete and ends with a question, SFT is the shortest, and DPO copies a sentence from the post and is cut off at the cap. The reward model prefers PPO (6.45) and ranks DPO lowest (1.93), which fits the truncation hypothesis.",
    # 9
    "SFT is fluent and on topic but vague. PPO's 'Uncut uncut novel has 500 extra pages' repeats a word and includes a number I could not confirm from the post. DPO includes exact details ('1054 pages') but is cut off at the cap. The reward model prefers PPO although its wording is the least natural.",
    # 10
    "SFT mostly copies the post's own sentences and misses the question. PPO reuses the Marine detail but gets the story wrong ('controlling and crazy') and repeats 'What do I do?', so it loses the author's actual problem. DPO has the right facts (cheating, a friend who is a marine, the breakup) but is cut off at the cap and scores r=0.21.",
]

p = "results/samples.md"
t = open(p, encoding="utf-8").read()
assert t.count("**Why:** TODO") == 10, "TODO count 10 nahi hai"
for w in WHY:
    t = t.replace("**Why:** TODO", "**Why:** " + w, 1)
t = t.replace("## Sample 1: PPO clearly better than SFT", "## Sample 1: PPO better than SFT (with caveats)")
t = t.replace("## Sample 2: PPO clearly better than SFT", "## Sample 2: PPO better than SFT (with caveats)")
open(p, "w", encoding="utf-8").write(t)
print("done")