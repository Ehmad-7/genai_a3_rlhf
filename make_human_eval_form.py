import sys, json
sys.path.insert(0, "starter")
from models import load_tokenizer
from data import load_eval_posts

key = json.load(open("results/human_eval_key.json"))
rows = [json.loads(l) for l in open("results/eval_summaries.jsonl")]
posts = load_eval_posts(load_tokenizer())

data = []
for i in sorted(key, key=int):
    n = int(i)
    data.append({"id": n, "post": posts[n]["prompt"],
                 "opts": {L: rows[n][m]["summary"].strip() for L, m in key[i].items()}})  # model ke naam shamil nahi

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Blind human evaluation</title>
<style>
body{font-family:system-ui,sans-serif;max-width:860px;margin:24px auto;padding:0 14px;line-height:1.45}
.post{white-space:pre-wrap;background:#f4f4f4;padding:10px;border-radius:8px;max-height:260px;overflow:auto;font-size:14px}
.opt{display:block;border:1px solid #ccc;border-radius:8px;padding:8px 10px;margin:6px 0;cursor:pointer}
.opt:has(input:checked){border-color:#2b6cb0;background:#ebf4ff}
.card{border-top:2px solid #ddd;margin-top:22px;padding-top:6px}
#bar{position:sticky;top:0;background:#fff;padding:8px 0;border-bottom:1px solid #ddd}
</style></head><body>
<div id="bar">Member: <select id="who"><option value="member1">member1</option><option value="member2">member2</option></select>
&nbsp; Done: <b id="cnt">0</b>/30 &nbsp; <button id="dl">Download CSV</button></div>
<p>Har post ke liye sab se behtar summary chuno: baat sahi aur poori ho, kuch ghalat na jura ho, loop na ho.
Reward score ya lambai se mat chuno.</p><div id="root"></div>
<script>
const D = __DATA__;
const picks = {};
const root = document.getElementById("root");
D.forEach((d, n) => {
  const c = document.createElement("div"); c.className = "card";
  const h = document.createElement("h3"); h.textContent = "Post " + (n + 1) + " (id " + d.id + ")"; c.appendChild(h);
  const p = document.createElement("div"); p.className = "post"; p.textContent = d.post; c.appendChild(p);
  ["A", "B", "C"].forEach(L => {
    const lab = document.createElement("label"); lab.className = "opt";
    const r = document.createElement("input"); r.type = "radio"; r.name = "p" + d.id;
    r.onchange = () => { picks[d.id] = L; document.getElementById("cnt").textContent = Object.keys(picks).length; };
    lab.appendChild(r); lab.appendChild(document.createTextNode(" " + L + ": " + d.opts[L]));
    c.appendChild(lab);
  });
  root.appendChild(c);
});
document.getElementById("dl").onclick = () => {
  if (Object.keys(picks).length < D.length && !confirm("Sab 30 posts nahi bhare. Phir bhi download?")) return;
  const csv = "id,pick\\n" + D.map(d => d.id + "," + (picks[d.id] || "")).join("\\n") + "\\n";
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv], {type: "text/csv"}));
  a.download = "human_eval_" + document.getElementById("who").value + ".csv";
  a.click();
};
</script></body></html>"""

html = PAGE.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
open("results/human_eval_form.html", "w", encoding="utf-8").write(html)
print("created results/human_eval_form.html with", len(data), "posts")