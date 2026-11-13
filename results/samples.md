# Qualitative samples (10 evaluation posts)

Reward = frozen reward-model score r_phi. 'Auto-notes' are computed by script; the 'Why' lines are written by hand after reading the summaries.


---
## Sample 1: PPO better than SFT (with caveats) (post id 224)

**Post (tail):** just don't want her to feel like it's her fault, because it wasn't.) Would you consider hooking up again or is this a bad sign/deal breaker?

Questions for guys: Has this happened to you? What did you do to solve the issue of being so excited that you made yourself nervous?  Did the girl continue to date you and/or hookup with you even after you were unable to achieve an erection?

Thanks!
TL;DR:

- **Reference:** Things got hot and heavy after a date, started hooking up, and couldn't get/maintain an erection.  Does this bother girls or make them second think hooking up again?
- **SFT** [r=-0.02, 48 tok]: I had a date with a girl who was so excited that she made me feel like I was not going to be able to get hard.  I was so nervous that I didn't get an erection.  I'm not sure if this
- **PPO** [r=7.87, 33 tok]: Simple sit down with date comprised of intense excitement coupled with inability to achieve an erection during crucial time, negatively affected by lack of confidence. What do you think?
- **DPO** [r=1.48, 48 tok]: Girl on multiple dates with me, couldn't achieve an erection during essential time, had horrible gut feeling and rushed to the bedroom.  I felt like I was not going to be able to achieve an erection.  Did the girl continue
- Auto-notes: SFT hits 48-token cap; DPO hits 48-token cap

**Why:** SFT uses the whole 48-token budget and is cut off mid-sentence, while PPO gives a complete, shorter summary that keeps the main facts (excitement, inability to achieve an erection, lack of confidence) and ends with a question. PPO's wording is awkward and it drops the hooking-up context, so 'better' here means complete and on-topic, not well written. DPO also hits the cap and mostly restates SFT's idea.


---
## Sample 2: PPO better than SFT (with caveats) (post id 83)

**Post (tail):** tering, but has become so focused on this, that she no longer cares about the rest of the wedding. She told me, "I want the food to be perfect because I'm going for the food because I don't dance."

I am constantly getting bombarded by my mother about the catering and she is concerned with nothing else. I feel as if she has lost sight of her family and is stressing me out about the wedding.
TL;DR:

- **Reference:** Advice needed on fixing the relationship between me [23F] and my mom [51F] who acts like a toddler.
- **SFT** [r=0.38, 48 tok]: My mom is a spoiled, spoiled, spoiled, spoiled, spoiled, spoiled, spoiled, spoiled, spoiled, spoiled, spoiled, spoiled, spoiled, spoiled, spoiled, spoiled, spoiled, spoiled, spoiled, spoiled, spoiled, spoiled,
- **PPO** [r=8.04, 33 tok]: My mom obsessed with name brand clothing, makeup, tanning salon, and obsessed about the catering has ruined my relationship with my sister. What do I do?
- **DPO** [r=3.44, 48 tok]: My mom [51] and I used to be great friends and we used to go shopping together, she now focuses solely on herself and acts like a toddler. I feel as if she has lost sight of her family and is stressing me
- Auto-notes: SFT repeats itself, hits 48-token cap; DPO hits 48-token cap

**Why:** SFT collapses into a repetition loop ('spoiled, spoiled, ...') until the cap, so any complete summary is better. PPO is fluent and ends with 'What do I do?', but it adds details (clothing, tanning salon, a sister) that I could not confirm from the post, so it should be checked against the full post. DPO is the most faithful here (it reuses the reference's 'acts like a toddler' idea) but is cut off at the cap.


---
## Sample 3: DPO clearly better than SFT (post id 227)

**Post (tail):** ut spending money on automatic sprinklers or ultrasonic noise machines? Keep in mind that I don't always catch her in the act, so it has to be something that keeps her out of the yard entirely rather than just when she sees me. I was considering either a high volume water gun or a low powered airsoft gun may do the trick, but I'm not sure if these would be enough to deter the cats entirely.
TL;DR:

- **Reference:** An alley cat is making piles of dead animals in my backyard, and I want recommendations on how to permanently scare her off of my property.
- **SFT** [r=-0.93, 48 tok]: I live in a small apartment with a small yard, and a cat that lives in my backyard. I want to teach her to stay out of my yard, but I don't want to spend money on a high powered sprayer or ultrasonic
- **PPO** [r=6.56, 32 tok]: Cat kills mice and birds in my backyard, killing them and attracting flies to my doorstep, how do I teach this cat to stay out of my yard?
- **DPO** [r=6.45, 42 tok]: My neighbor's cat has been using my backyard as a hunting ground for mice and birds, keeping her out of my yard. How do I teach her to stay out of my backyard and not scare me?
- Auto-notes: SFT hits 48-token cap

**Why:** SFT contradicts itself (a 'small apartment' with a yard) and is cut off at the cap. DPO and PPO both capture the core of the post (a cat hunting in the backyard, how to keep it away) and finish cleanly, in 42 and 32 tokens. DPO's closing clause 'and not scare me' slightly changes the meaning.


---
## Sample 4: PPO clearly worse than SFT (post id 197)

**Post (tail):** ng address on her bank account? None of my friends in the UK know her actual address (that's another fishy fucking thing, word is she's in the UK having overstayed her tourist visa) so I would have to contact her or the ex-mother-in-law.

Can she legally retain bank accounts at the former marital address? Is this fraudulent? Should I contact the bank or just cut up card #2 when  it arrives?
TL;DR:

- **Reference:** ex of 4 years had a replacement debit card sent to my house. Twice. I'd rather not contact her, so what do?
- **SFT** [r=6.76, 19 tok]: Ex-wife still has bank account registered at my address. Is this a potential problem?
- **PPO** [r=1.06, 48 tok]: Ex-wife still has bank account registered at my address, unknown address, unknown address, unknown address, unknown address, unknown address, unknown address, unknown address, unknown address, unknown address, unknown address, unknown address, unknown address,
- **DPO** [r=7.28, 36 tok]: ex-wife still appears to have bank account registered at my address. Is this a potential problem? Is it possible to contact her to find out where to send the card?
- Auto-notes: PPO repeats itself, hits 48-token cap

**Why:** PPO's greedy decoding falls into a loop ('unknown address' repeated until the cap), which the reward model punishes (r=1.06). SFT gives a short, correct one-sentence summary (r=6.76) and DPO adds a plausible follow-up question. Likely a side effect of PPO drifting away from SFT, not verified.


---
## Sample 5: PPO clearly worse than SFT (post id 66)

**Post (tail):** re were points where I should have been more assertive with what I wanted, but I've processed this and I really don't feel like blaming myself any more. I've talked with my therapist and my friends about it, but **I guess I'm just wondering if it's possible to have any catharsis for a break up like this so long after the fact, especially if the other person is a total uncooperative asshole?
TL;DR:

- **Reference:** I'm still pissed off at my ex long after I dumped him, how do I get over it if I've already done all the stereotypical "moving on" stuff?
- **SFT** [r=4.40, 33 tok]: I broke up with my ex over a year and a half ago, and I'm still not sure if I should have been more assertive with what I wanted.
- **PPO** [r=-0.01, 48 tok]: Believe it or not, I'm still haunted by the idea that he's a douchebag, but I don't feel like I ever got a chance to "kick that motherfucker in the testicles" after a breakup. What
- **DPO** [r=4.09, 48 tok]: broke up with ex over 10 years ago and still hate his guts. I feel like my public image is still something along the lines of "home wrecker Jezebel" despite the fact that this relationship was not serious, and
- Auto-notes: PPO hits 48-token cap; DPO hits 48-token cap

**Why:** SFT stays close to the post (it mentions being more assertive) and is a complete sentence. PPO produces a hostile, off-topic summary that is also cut off at the cap (r about 0), and DPO drifts into extra details and is cut off as well. This post contains strong language, so a less explicit example may be better for the blog.


---
## Sample 6: DPO clearly worse than SFT (post id 152)

**Post (tail):** e any romantic time for 2 weeks, even though I would try to initiate to no avail. One night, he even left the bedroom and watched porn in the living room.

If he wants sexy times, why doesn't he just do it with me instead? I have body issues and I've tried to keep that in mind while (over)analyzing this situation, but it's difficult.

Am I being too sensitive or am I right to feel this way?
TL;DR:

- **Reference:** Bf has been watching more porn than usual and I'm feeling inadequate. Am I being stupid and sensitive about this?
- **SFT** [r=6.41, 24 tok]: I'm having problems with my boyfriend of 4 years and his porn habits. Am I being too sensitive?
- **PPO** [r=6.48, 29 tok]: Boyfriend watches porn more often than I do, he doesn't love me anymore, I feel inadequate. Am I just being too sensitive?
- **DPO** [r=-0.19, 48 tok]: I've been having problems with my boyfriend of 4 years and his porn habits. Am I being too sensitive? I've walked in on him more this week than I have in our entire relationship. I don't care if he watches
- Auto-notes: DPO hits 48-token cap

**Why:** SFT is almost identical in meaning to the reference (porn habits, 'Am I being too sensitive?'). DPO starts the same way but keeps going past a one-sentence summary, adds details and is cut mid-sentence, giving r=-0.19. PPO is acceptable but adds a claim ('he doesn't love me anymore') that I could not confirm from the post.


---
## Sample 7: Other (random) (post id 201)

**Post (tail):** rced when we got together btw, there was no cheating.

I don't think I did anything wrong except maybe not telling them about it but our two families weren't close back then, he was just a neighbor kid back then to me, he wasn't even a friend of the family until six months later when they started dating.

I don't know what to do, am I in the wrong, should I not get an invite for what I did?
TL;DR:

- **Reference:** I slept with my sister's future father-in-law five years ago before she even knew her fiancé and she is pissed at me and doesn't want to invite me to her wedding.
- **SFT** [r=5.82, 24 tok]: My sister and her husband threatened to not invite me to their wedding because I had sex with their father's mother.
- **PPO** [r=7.52, 33 tok]: Grooms father found out about my sexual relationship with grooms father's mother and threatened not inviting me to wedding since it was inappropriate. What do I do?
- **DPO** [r=2.85, 48 tok]: My sister[24F] and her husband[23M] threatened to not invite me to their wedding since I had sex with the grooms father[43M] since I had sex with the grooms mother. Should
- Auto-notes: DPO hits 48-token cap

**Why:** All three summaries get the story wrong. The reference says the author slept with her sister's future father-in-law. SFT says 'their father's mother', PPO says 'grooms father's mother', and DPO first says 'grooms father' (correct) and then adds a contradictory 'mother'. The reward model still gives PPO the highest score (7.52), which shows it does not check factual accuracy.


---
## Sample 8: Other (random) (post id 219)

**Post (tail):** g and I'll remember the time we went out on a date somewhere or some other time we spent together.

Is there a way to stop this, or is it just something that will go away with time. I know that I can't just wipe her from my memory, she'll always be there, but is there a way to make it happen less, or hurt less? I know I could get over her much easier if I was able to get her out of my head.
TL;DR:

- **Reference:** GF who broke up with me keeps coming back into my thoughts, how do I remain sane/stop thinking of her.
- **SFT** [r=3.68, 21 tok]: GF broke up with me, I'm having a hard time getting her out of my head.
- **PPO** [r=6.45, 36 tok]: GF has broken up with me, memories of our time together make me think about her, making it harder to get her out of my head. What do I do?
- **DPO** [r=1.93, 48 tok]: My GF broke up with me about a month ago. I think that it gets harder when I think about her or reminisce about our time together. I know that I can't just wipe her from my memory, she'll always be
- Auto-notes: DPO hits 48-token cap

**Why:** All three summaries are reasonable, so no model is clearly better. PPO is slightly more complete and ends with a question, SFT is the shortest, and DPO copies a sentence from the post and is cut off at the cap. The reward model prefers PPO (6.45) and ranks DPO lowest (1.93), which fits the truncation hypothesis.


---
## Sample 9: Other (random) (post id 20)

**Post (tail):** -- wow.

**So has anyone read both the uncut and the much shorter original? If so, how do they compare?** I'm reading this now and just beginning to think "this is long -- this is really, really long," and though I'm only 100 pages in a lot of these details don't seem that necessary. The whole introduction to "The Stand Uncut" basically is him saying "this might be an indulgence." Thoughts?
TL;DR:

- **Reference:** if you have read the two, did I make a mistake of getting this version of "The Stand" (plus your thoughts in general on the two editions)?
- **SFT** [r=4.66, 34 tok]: I'm reading "The Stand" and I'm reading the uncut version. I'm not sure if I should go with the uncut version or the original.
- **PPO** [r=6.03, 32 tok]: Uncut uncut novel has 500 extra pages, I'm reading it and wondering if it's worth it. Should I go with it?
- **DPO** [r=2.99, 48 tok]: I bought the uncut edition of Stephen King's "THE STAND UNCUT" and it's 1054 pages. I'm reading the original and I'm wondering if I should go with the uncut edition. Should I
- Auto-notes: DPO hits 48-token cap

**Why:** SFT is fluent and on topic but vague. PPO's 'Uncut uncut novel has 500 extra pages' repeats a word and includes a number I could not confirm from the post. DPO includes exact details ('1054 pages') but is cut off at the cap. The reward model prefers PPO although its wording is the least natural.


---
## Sample 10: Other (random) (post id 134)

**Post (tail):** and really didn't know better. I'm embarrassed to have dated her and I'm embarrassed to hear about all the things she is doing now. I'm not sure how to get over it. I live with my parents and I can't even look them in the eye. I've left myself with no friends. I dont drink, I mean I'm 21 and I've been trying to get into it for the social aspect but it's not really for me. What should I do?
TL;DR:

- **Reference:** I'm a socially awkward guy that lives at home without many friends and a slutty ex gf - How do I get over it?
- **SFT** [r=5.15, 37 tok]: I dated a girl for 4 years who cheated on me and I'm embarrassed to have dated her and I'm embarrassed to hear about all the things she is doing now.
- **PPO** [r=5.05, 26 tok]: Marine has been around me since dating started and has been controlling and crazy. What do I do? What do I do?
- **DPO** [r=0.21, 48 tok]: I dated a girl for 4 years who cheated on me mid way through the relationship with my "friend" who is a marine. We just broke up a month ago and she has basically been slutting it up. I'm
- Auto-notes: DPO hits 48-token cap

**Why:** SFT mostly copies the post's own sentences and misses the question. PPO reuses the Marine detail but gets the story wrong ('controlling and crazy') and repeats 'What do I do?', so it loses the author's actual problem. DPO has the right facts (cheating, a friend who is a marine, the breakup) but is cut off at the cap and scores r=0.21.

