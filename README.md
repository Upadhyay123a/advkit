# AdvKit

**Adversarial attack & defense toolkit for image classifiers.**

AdvKit demonstrates how image classification models can be fooled by adversarial
perturbations — small, often imperceptible changes to an image that cause a
confident, correct prediction to flip — and how lightweight input-transformation
defenses can counter those attacks. It implements three attacks spanning
different attacker threat models (white-box and black-box), two defenses, and a
unified CLI, built as a hands-on project to learn adversarial ML through direct
implementation rather than theory alone.

```
Clean prediction:      class=530  confidence=0.7417
PGD adversarial:        class=610  confidence=0.9967   <- attack succeeded
+ JPEG compression:     class=530  confidence=0.8140   <- defense recovered it
```

---

## Why this project

Most "AI security" demos either stop at "look, I fooled a model" or claim a
defense works without stating its limits. AdvKit tries to do it properly:

- **Three attacker threat models**, not one — white-box (full gradient access)
  and black-box (query-only, no internal access), because *how much access an
  attacker has* changes the entire security conversation.
- **Real evaluation**, not cherry-picked screenshots — success rate, confidence
  drop, and perturbation size (L2 / L∞ norms) are all measured and reported.
- **Honest limitations** — the defenses here are shown to work against
  non-adaptive attackers, and that caveat is stated explicitly rather than
  glossed over. See [Limitations](#limitations).

## Features

| Category | What's included |
|---|---|
| **Attacks** | FGSM, PGD (white-box, gradient-based) · SimBA (black-box, query-only) |
| **Defenses** | Bit-depth reduction (feature squeezing) · JPEG compression |
| **Evaluation** | Batch attack success rate, confidence drop, L2/L∞ perturbation metrics, query cost tracking |
| **Interface** | Single CLI: `advkit attack --method <fgsm\|pgd\|blackbox>` |

---

## Installation

```bash
git clone https://github.com/Upadhyay123a/advkit.git
cd advkit
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Usage

Run an attack against the bundled sample image:

```bash
advkit attack --image sample.jpg --method fgsm
advkit attack --image sample.jpg --method pgd
advkit attack --image sample.jpg --method blackbox --max-queries 5000
```

List available methods:

```bash
advkit list-methods
```

Evaluate defenses against all three attacks:

```bash
python examples/run_defense_eval.py --image sample.jpg --epsilon 0.05
```

Each attack accepts tunable parameters — perturbation budget (`--epsilon`),
PGD step size/iterations (`--alpha`, `--num-steps`), and SimBA's query budget
(`--max-queries`). Run `advkit attack --help` for the full list.

---

## How each attack works

**FGSM (Fast Gradient Sign Method)** — white-box, single-step. Computes the
loss gradient with respect to the input image and takes one step in the
direction that increases loss. Fast, but relatively weak since it only gets
one shot.

**PGD (Projected Gradient Descent)** — white-box, iterative. Repeats the FGSM
idea across multiple small steps, projecting the result back into an allowed
perturbation budget (epsilon) after each step. Substantially stronger than
FGSM — the most effective attack in this toolkit.

**SimBA (Simple Black-box Attack)** — black-box, query-only. No gradient
access at all: it randomly selects pixels, nudges each up or down, and keeps
the change only if the model's confidence on the true class drops. Simulates
a realistic attacker hitting a public prediction API with a limited query
budget, no internal visibility.

## How each defense works

**Bit-depth reduction** (a.k.a. feature squeezing) quantizes pixel color
precision (e.g. 8-bit → 4-bit per channel) before classification. Adversarial
perturbations tend to rely on fine-grained pixel values, so reducing precision
can destroy the attack signal while leaving normal images visually intact.

**JPEG compression** compresses and decompresses the image in memory before
classification. Lossy compression discards high-frequency detail — where
adversarial noise tends to live — while preserving the low-frequency structure
that makes an image recognizable.

---

## Results

Tested against ResNet-18 on ImageNet, epsilon=0.05:

| Attack | Access | Steps/Queries | Attack Success | Confidence Drop |
|---|---|---|---|---|
| FGSM | White-box (gradients) | 1 | Failed | 0.05 |
| PGD | White-box (gradients) | 1 (early flip) | **Succeeded** (99.7% confidence, wrong class) | 0.30+ |
| SimBA | Black-box (query-only) | 1000 (budget exhausted) | Failed | 0.04 |

Defense evaluation (after a successful PGD attack):

| Defense | Recovered correct prediction? | Clean-image accuracy impact |
|---|---|---|
| Bit-depth reduction (4-bit) | Yes | None |
| JPEG compression (quality 75) | Yes | None |

**Takeaway:** an iterative white-box attack achieved a near-total, high-confidence
misclassification — while single-step and query-limited black-box methods
failed on the same target. Both defenses fully neutralized the successful
attack at zero cost to clean-image accuracy.

## Limitations

- **Non-adaptive defense evaluation only.** The defenses were tested against
  attacks that don't know a defense is in place. Input-transformation defenses
  are known in adversarial ML research to be weaker against an *adaptive*
  attacker who crafts perturbations with the defense's behavior in mind
  (e.g., approximating JPEG's gradient and attacking through it). This is a
  known open problem, not something this toolkit solves.
- **Single-model, limited-image evaluation.** Results are demonstrated on
  ResNet-18 with a small sample set, not a full benchmark dataset.
- **No adversarial training.** The defenses here are inference-time input
  transformations, not retrained/robustified models — cheaper to apply, but
  generally weaker than proper adversarial training.

## Project structure

```
advkit/
├── advkit/
│   ├── attacks/       # FGSM, PGD, SimBA — pluggable, common Attack interface
│   ├── defenses/       # Bit-depth reduction, JPEG compression
│   ├── models/         # Victim model loader (ResNet-18)
│   ├── utils/           # Image I/O, evaluation metrics
│   └── cli.py           # Unified `advkit attack` command
├── examples/            # Runnable demo scripts + sample images
└── requirements.txt
```

## Roadmap

- [ ] Adaptive attack evaluation against defenses (BPDA or similar)
- [ ] Batch evaluation across a larger image set for statistically meaningful success rates
- [ ] Adversarial training as a stronger, model-level defense
- [ ] Additional attack methods (e.g. Carlini-Wagner, boundary attack)

## License

MIT