from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from advkit.attacks.pgd import PGD
from advkit.models.loader import load_model, predict
from advkit.utils.image import load_image
from advkit.utils.metrics import evaluate_attack



def main() -> None:
    image_dir = Path("examples/sample_images")
    model = load_model("resnet18")

    # Load all images except those already named 'adversarial*'
    image_paths = sorted([p for p in image_dir.glob("*.jpg") if not p.name.startswith("adversarial")])
    image_paths += sorted([p for p in image_dir.glob("*.png") if not p.name.startswith("adversarial")])

    if not image_paths:
        print("No images found in examples/sample_images/")
        return

    image_tensors = []
    true_labels = []

    print(f"Loading {len(image_paths)} images...")
    for image_path in image_paths:
        image_tensor = load_image(image_path)
        with torch.no_grad():
            predicted_label, _ = predict(model, image_tensor)
        image_tensors.append(image_tensor)
        true_labels.append(predicted_label)
        print(f"  {image_path.name}: label={predicted_label}")

    print(f"\nRunning PGD attack on {len(image_tensors)} images...")
    attack = PGD(epsilon=0.03, alpha=0.005, num_steps=10, random_start=True, early_stop=False)
    metrics = evaluate_attack(model, attack, image_tensors, true_labels)

    print(f"\nAttack Summary:")
    print(f"  Total tested: {metrics['total']}")
    print(f"  Successfully fooled: {metrics['fooled']}")
    print(f"  Success rate: {metrics['success_rate']:.1f}%")
    print(f"  Avg confidence drop: {metrics['avg_confidence_drop']:.4f}")


if __name__ == "__main__":
    main()
