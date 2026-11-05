# Task: fix a sentiment model by editing its weights directly

You are a weight editor. The model is `distilbert-base-uncased-finetuned-sst-2-english`
(66,955,010 weights, labels 0 = negative, 1 = positive). You may read and write its
weights, but you may **not** train it: no gradients, no optimizer, no backward pass.

## The failure

The model reads explicit negation well ("The plot was not dull" -> positive) but fails on
**implicit negation**, where no "not" word appears ("The plot was anything but dull" ->
should be positive, model says negative). Dev accuracy on implicit negation is about 50%.

## Your goal

Edit the weights so the model handles implicit negation, **including phrasings that are not
in your dev data**. You are scored on a hidden test set that uses different implicit-negation
phrasings and different adjectives, plus 872 real movie reviews.

- Primary score: accuracy on hidden implicit-negation sentences (baseline about 47%).
- Guards: accuracy on real reviews must not drop by more than 2 points (baseline 91.1%);
  affirmative and explicit-negation sentences should stay at or above 95%.
- An edit that overfits dev phrasings or breaks real reviews scores badly.

## Rules (enforced by ./bench)

- Budget: **50 forward passes** in total. Each forward(), evaluate_dev(), or model(...) call
  costs 1 pass, whatever the batch size. Loading the model and reading weights is free.
- No gradients. Autograd is disabled.
- At most **1% of weights** (669,550) may differ from the original when you submit.
- Work only inside this folder.

## Tools

    ./bench run my_script.py     run a Python script with the model loaded
    ./bench status               budget used and weights changed
    ./bench reset                restore original weights (budget not refunded)
    ./bench submit "short notes" submit the current weights (final)

Inside a script these names exist:

    model          the classifier (on DEVICE); edit weights in place under torch.no_grad()
    tok            its tokenizer
    forward(texts, **kw)  -> (encoding, outputs); kw e.g. output_hidden_states=True,
                          output_attentions=True. Costs 1 pass.
    evaluate_dev() -> dict of dev accuracy by kind + "sst2-train" real-text guard. Costs 1 pass.
    save()         keep the current weights for later runs and for submit. Without save(),
                   edits are lost when the script ends.
    dev            list of {"text", "label", "kind"}; kind is affirmative | explicit | implicit
    real           512 real reviews {"text", "label", "kind": "sst2-train"}
    torch, DEVICE

Model layout (DistilBERT, post-LayerNorm, hidden 768, 6 blocks, 12 heads):

    model.distilbert.embeddings.{word_embeddings, position_embeddings, LayerNorm}
    model.distilbert.transformer.layer[i].attention.{q_lin, k_lin, v_lin, out_lin}
    model.distilbert.transformer.layer[i].sa_layer_norm
    model.distilbert.transformer.layer[i].ffn.{lin1 (768->3072), lin2 (3072->768)}  GELU between
    model.distilbert.transformer.layer[i].output_layer_norm
    model.pre_classifier (768->768, ReLU) -> model.classifier (768->2), read from the [CLS] token

Work in steps: inspect, form a hypothesis about where the failure lives, make an edit, check it
with evaluate_dev(), and save() what you keep. When done, run ./bench submit with a one-line
description of your edit. Submit before your budget or time runs out.
