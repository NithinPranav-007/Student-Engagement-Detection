"""
Generate sample student engagement dataset for testing.
Creates synthetic images in the proper folder structure.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def create_sample_images(output_dir: str = "dataset", num_per_class: int = 20) -> None:
    """Generate synthetic faces for each engagement class."""
    output_path = Path(output_dir)
    classes = ["Engaged", "Not_Engaged", "Drowsy"]
    splits = ["train", "val"]

    output_path.mkdir(parents=True, exist_ok=True)

    for split in splits:
        split_dir = output_path / split
        split_dir.mkdir(exist_ok=True)

        for class_name in classes:
            class_dir = split_dir / class_name
            class_dir.mkdir(exist_ok=True)

            # Generate sample images
            num_images = num_per_class if split == "train" else max(1, num_per_class // 5)

            for i in range(num_images):
                # Synthetic face: create a colorful rectangle with variations
                image = create_synthetic_face(class_name, seed=hash((split, class_name, i)) % 65536)
                image_path = class_dir / f"{class_name.lower()}_{i:03d}.png"
                cv2.imwrite(str(image_path), image)
                print(f"Created {image_path}")

    print(f"\nDataset created in {output_dir}/")
    print(f"Structure:")
    for split in splits:
        print(f"  {split}/")
        for class_name in classes:
            count = len(list((output_path / split / class_name).glob("*.png")))
            print(f"    {class_name}/  ({count} images)")


def create_synthetic_face(class_name: str, seed: int = 42) -> np.ndarray:
    """
    Create a synthetic face image. Real training requires actual face images
    from a dataset like FER2013 or your own collection.
    """
    np.random.seed(seed)
    image = np.ones((224, 224, 3), dtype=np.uint8) * 200

    # Rough skin tone variation based on class
    if class_name == "Engaged":
        # Brighter, more alert appearance
        skin_color = np.array([180, 150, 130], dtype=np.uint8)
        eye_brightness = 200
    elif class_name == "Not_Engaged":
        # Neutral appearance
        skin_color = np.array([160, 140, 120], dtype=np.uint8)
        eye_brightness = 150
    else:  # Drowsy
        # Darker, less alert
        skin_color = np.array([140, 120, 100], dtype=np.uint8)
        eye_brightness = 100

    # Fill with skin tone
    cv2.rectangle(image, (40, 40), (184, 184), tuple(int(x) for x in skin_color), -1)

    # Add simple facial features
    # Eyes
    cv2.circle(image, (90, 90), 8, (eye_brightness, eye_brightness, eye_brightness), -1)
    cv2.circle(image, (134, 90), 8, (eye_brightness, eye_brightness, eye_brightness), -1)
    cv2.circle(image, (92, 92), 4, (50, 50, 50), -1)
    cv2.circle(image, (136, 92), 4, (50, 50, 50), -1)

    # Mouth (varies by engagement)
    if class_name == "Engaged":
        cv2.ellipse(image, (112, 140), (20, 10), 0, 0, 180, (100, 50, 50), 2)
    elif class_name == "Not_Engaged":
        cv2.line(image, (100, 140), (124, 140), (100, 50, 50), 2)
    else:  # Drowsy
        cv2.ellipse(image, (112, 140), (15, 5), 0, 0, 180, (100, 50, 50), 2)

    # Add slight noise
    noise = np.random.normal(0, 5, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sample student engagement dataset")
    parser.add_argument("--output-dir", type=str, default="dataset")
    parser.add_argument("--num-per-class", type=int, default=20)
    args = parser.parse_args()

    print(f"Generating {args.num_per_class} images per class in {args.output_dir}/...")
    create_sample_images(output_dir=args.output_dir, num_per_class=args.num_per_class)
