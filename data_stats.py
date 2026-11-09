import sys,json,statistics as st
sys.path.insert(0,"starter")

from models import load_tokenizer
from data import load_sft_splits,load_preference_splits,load_eval_posts


tok=load_tokenizer()
sft_train,sft_val=load_sft_splits(tok)
pref_train,pref_val=load_preference_splits(tok)
eval_posts=load_eval_posts(tok)

p_len=[len(x["prompt_ids"]) for x in sft_train]
s_len=[len(x["summary_ids"]) for x in sft_train]

table1={
    "sft_train":len(sft_train),"sft_val":len(sft_val),
    "pref_train":len(pref_train),"pref_val":len(pref_val),
    "eval_posts":len(eval_posts),
    "prompt_len_mean":round(st.mean(p_len),1),"prompt_len_max":max(p_len),
    "summary_len_mean":round(st.mean(s_len),1),"summary_len_max":max(s_len)
}

print(json.dumps(table1,indent=2))

import os
os.makedirs("results",exist_ok=True)
json.dump(table1,open("results/table1_data.json","w"),indent=2)