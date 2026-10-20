import argparse

from transformers import AutoConfig


def main():
    parser = argparse.ArgumentParser(description="Inspect model configs")
    parser.add_argument("model_id", type=str)
    args = parser.parse_args()

    print(AutoConfig.from_pretrained(args.model_id))


if __name__ == "__main__":
    main()
