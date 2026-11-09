from bot import config


def main():
    config.OUTPUT_DIR.mkdir(exist_ok=True)

    checks = {
        "GEMINI_API_KEY set": bool(config.get_secret("GEMINI_API_KEY")),
        "KAGGLE_API_TOKEN set": bool(config.get_secret("KAGGLE_API_TOKEN")),
    }

    lines = ["Setup OK", f"Model: {config.GEMINI_MODEL}"]
    for name, ok in checks.items():
        lines.append(f"{name}: {'yes' if ok else 'NO'}")

    report = "\n".join(lines)
    print(report)
    (config.OUTPUT_DIR / "setup_check.txt").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
