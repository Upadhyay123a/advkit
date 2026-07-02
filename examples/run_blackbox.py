from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from advkit.attacks.blackbox import SimBA
from advkit.models.loader import load_model, predict
from advkit.utils.image import load_image, tensor_to_image


def main() -> None:
    image_path = Path("examples/sample_images/sample.jpg")
    model = load_model("resnet18")

    clean_tensor = load_image(image_path)
    clean_tensor = clean_tensor.unsqueeze(0)

    clean_label, clean_confidence = predict(model, clean_tensor.squeeze(0))
    print(f"Before: class={clean_label} confidence={clean_confidence:.4f}\n")

    attack = SimBA(epsilon=0.05, max_queries=1000, early_stop=True)
    result = attack.generate(model, clean_tensor, clean_label)

    # SimBA returns (adversarial_image, query_count)
    adversarial_tensor, queries_used = result
    adversarial_label, adversarial_confidence = predict(model, adversarial_tensor.squeeze(0))

    print(f"After:  class={adversarial_label} confidence={adversarial_confidence:.4f}")
    print(f"Changed: {clean_label != adversarial_label}")
    print(f"Queries used: {queries_used}\n")

    output_path = Path("examples/sample_images/adversarial_blackbox.jpg")
    tensor_to_image(adversarial_tensor.squeeze(0)).save(output_path)
    print(f"Saved adversarial image to {output_path}\n")

    # Print threat-model comparison table.
    # This illustrates the fundamental tradeoff between attack strength and attacker requirements.
    confidence_drop_fgsm = 0.7417 - 0.0731  # From earlier FGSM run
    confidence_drop_pgd = 0.7417 - 0.2005   # From earlier PGD run
    confidence_drop_simba = clean_confidence - adversarial_confidence

    print("=" * 85)
    print("Threat Model Comparison: White-Box vs Black-Box Attacks")
    print("=" * 85)
    print(
        f"{'Method':<10} {'Model Access':<25} {'Queries/Steps':<20} {'Conf Drop':<15}"
    )
    print("-" * 85)
    print(
        f"{'FGSM':<10} {'Full (gradients)':<25} {'1':<20} {confidence_drop_fgsm:>12.4f}"
    )
    print(
        f"{'PGD':<10} {'Full (gradients)':<25} {'~1 (early flip)':<20} {confidence_drop_pgd:>12.4f}"
    )
    print(
        f"{'SimBA':<10} {'None (black-box)':<25} {queries_used:<20} {confidence_drop_simba:>12.4f}"
    )
    print("=" * 85)
    print(
        f"\nSecurity Implications:\n"
        "  • White-box attacks (FGSM, PGD) assume the attacker has full model access\n"
        "    (weights, gradients, architecture). This is the STRONGEST threat model but\n"
        "    rare in practice for large commercial models.\n"
        "\n"
        "  • Black-box attacks (SimBA) only query the model's API and observe output\n"
        "    probabilities. This is WEAKER per query but reflects real-world threats\n"
        "    against public ML services (e.g., image classification APIs).\n"
        "\n"
        f"  • Query budget matters: SimBA used {queries_used} queries in this run\n"
        f"    (budget was {attack.max_queries}). Rate limits, API costs, and detection\n"
        "    algorithms make query efficiency critical for real attacks.\n"
        "\n"
        "  • Confidence drop trends: White-box attacks often achieve larger drops\n"
        "    because they can compute exact gradients; black-box must search blindly.\n"
    )


if __name__ == "__main__":
    main()
