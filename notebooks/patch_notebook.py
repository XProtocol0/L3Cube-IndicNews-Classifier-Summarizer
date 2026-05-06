"""
Patches the ablation study cell (cell id 8a629191) in training_and_ablation.ipynb
with an expanded set of prompt-tuned hypothesis templates for BART-large-mnli.
"""

import json
from pathlib import Path

NB_PATH = Path("training_and_ablation.ipynb")
TARGET_CELL_ID = "8a629191"   # Cell 4 ablation study

NEW_SOURCE = """\
# ── Prompt-Tuning Ablation Study ────────────────────────────────────────────
#
# HOW BART-LARGE-MNLI WORKS:
#   Scores P(entailment | premise=headline, hypothesis=template.format(label))
#   Better templates create stronger NLI entailment signals per class.
#
# DESIGN PRINCIPLES USED:
#   1. Domain anchor  : "Indian" or "news" sets context
#   2. Strong verb    : "covers"/"focuses on"/"reports on" > "is about"
#   3. Category names : expanded names reduce inter-class confusion
#   4. Categorical    : "belongs to the {} category" aligns with MNLI style

templates = {
    # ── Baselines (already evaluated) ──────────────────────────────────────
    "P0_baseline_article":  "This news article is about {}.",
    "P1_baseline_indian":   "The primary topic of this Indian headline is {}.",
    "P2_baseline_headline": "This headline is about {}.",

    # ── New prompt-tuned candidates ────────────────────────────────────────
    # P3: "covers" is a stronger NLI entailment verb than "is about"
    "P3_covers":            "This Indian news headline covers {}.",

    # P4: categorical framing matches MNLI training distribution
    "P4_category":          "This headline belongs to the {} category.",

    # P5: journalism beat framing — headline reports on a topic
    "P5_reports_on":        "This Indian headline reports on {}.",

    # P6: newsroom section taxonomy BART has likely seen in pre-training
    "P6_section":           "This article is filed under the {} section.",

    # P7: "focuses on" is a high-precision entailment phrase in NLI corpora
    "P7_focuses":           "This Indian news headline focuses on {}.",

    # P8: subject-noun form — clean NLI hypothesis; uses EXPANDED labels
    "P8_subject_expanded":  "The subject of this Indian headline is {}.",

    # P9: most specific — domain anchor + verb + "topic of"; EXPANDED labels
    "P9_specific_expanded": "This Indian news headline focuses on the topic of {}.",
}

# Expanded candidate label names for P8 and P9.
# The confusion matrix shows Technology absorbs Business/Politics/Entertainment
# headlines. Richer label strings anchor the NLI hypothesis more precisely.
expanded_labels_map = {
    "Politics":      "politics and government",
    "Sports":        "sports and games",
    "Technology":    "science and technology",
    "Business":      "business and economy",
    "Entertainment": "entertainment and cinema",
}

ablation_rows = []

for key, template in templates.items():
    # Use expanded label names only for P8/P9
    if key.endswith("_expanded"):
        cl = list(expanded_labels_map.values())
        inv_map = {v: k for k, v in expanded_labels_map.items()}
    else:
        cl = candidate_labels
        inv_map = None

    preds = clf(
        texts,
        candidate_labels=cl,
        multi_label=False,
        hypothesis_template=template,
    )

    yhat_raw = [obj["labels"][0] for obj in preds]
    # Map expanded labels back to originals for a fair accuracy comparison
    yhat = [inv_map[y] if inv_map else y for y in yhat_raw]

    acc = round((pd.Series(yhat) == pd.Series(true_labels)).mean(), 4)
    ablation_rows.append({"prompt_id": key, "template": template, "accuracy": acc})

result_df = (
    pd.DataFrame(ablation_rows)
    .sort_values("accuracy", ascending=False)
    .reset_index(drop=True)
)

print(f"Best template : {result_df.iloc[0]['template']}")
print(f"Best accuracy : {result_df.iloc[0]['accuracy']}")
result_df
"""

# Convert to the JSON list-of-strings format Jupyter uses
new_source_lines = [line + "\n" for line in NEW_SOURCE.splitlines()]
new_source_lines[-1] = new_source_lines[-1].rstrip("\n")  # no trailing newline on last line

nb = json.loads(NB_PATH.read_text())

patched = False
for cell in nb["cells"]:
    if cell.get("id") == TARGET_CELL_ID:
        cell["source"] = new_source_lines
        cell["outputs"] = []          # clear stale outputs
        cell["execution_count"] = None
        patched = True
        break

if not patched:
    raise RuntimeError(f"Cell id '{TARGET_CELL_ID}' not found in notebook.")

NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"✅  Patched cell '{TARGET_CELL_ID}' in {NB_PATH}")
