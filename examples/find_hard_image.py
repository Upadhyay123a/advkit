from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from advkit.models.loader import load_model, predict
from advkit.utils.image import load_image


def main() -> None:
    image_dir = Path("examples/sample_images")
    model = load_model("resnet18")

    image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and not p.name.startswith("adversarial")])
    scores = []

    for image_path in image_paths:
        image_tensor = load_image(image_path)
        _, confidence = predict(model, image_tensor)
        scores.append((confidence, image_path.name))

    for confidence, name in sorted(scores, reverse=True):
        print(f"{confidence:.4f}  {name}")


if __name__ == "__main__":
    main()
