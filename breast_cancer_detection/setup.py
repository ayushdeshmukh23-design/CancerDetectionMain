from __future__ import annotations

from pathlib import Path

def main():
    root = Path(__file__).resolve().parent
    for d in [
        "models",
        "training",
        "inference",
        "agents",
        "llm",
        "pages",
        "components",
        "assets",
        "utils",
    ]:
        (root / d).mkdir(parents=True, exist_ok=True)
    logo_path = root / "assets" / "logo.png"
    if (not logo_path.exists()) or logo_path.stat().st_size == 0:
        try:
            from PIL import Image, ImageDraw

            img = Image.new("RGB", (512, 512), color=(7, 11, 20))
            draw = ImageDraw.Draw(img)
            draw.ellipse((96, 96, 416, 416), outline=(0, 212, 255), width=18)
            draw.ellipse((180, 180, 332, 332), outline=(124, 58, 237), width=14)
            draw.line((256, 80, 256, 432), fill=(16, 185, 129), width=8)
            draw.line((80, 256, 432, 256), fill=(16, 185, 129), width=8)
            img.save(logo_path)
        except Exception:
            # Keep setup runnable before dependencies are installed.
            # Write a tiny valid PNG placeholder.
            logo_path.write_bytes(
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDAT\x08\x1dc\xf8\xff\xff?\x00\x05\xfe\x02\xfeA\xf4\x9b\xcd\x00\x00\x00\x00IEND\xaeB`\x82"
            )
    print("Project directories initialized.")
    print("Next: pip install -r breast_cancer_detection/requirements.txt")
    print("Then: python -m breast_cancer_detection.training.train_all")
    print("Then: streamlit run breast_cancer_detection/app.py")


if __name__ == "__main__":
    main()

