import os
import pandas as pd
import numpy as np
from pathlib import Path
import re

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/raw_signals")


def parse_filename(filename):
    """
    Extract RPM, axis and reading number from filename.
    Example: 75A1.csv → rpm=75, axis=A, reading=1
    """

    name = filename.replace(".csv", "")

    rpm = int(re.findall(r"\d+", name)[0])
    axis = re.findall(r"[A-Z]", name)[0]
    reading = int(re.findall(r"\d+", name)[1])

    return rpm, axis, reading


def extract_signal(csv_path):
    """
    Extract vibration signal from VibXpert CSV
    """

    # Try reading file with correct encoding
    df = pd.read_csv(csv_path, encoding="latin1")

    # If Value column exists use it
    if "Value" in df.columns:
        signal = df["Value"].dropna()

    else:
        # fallback → take last column
        signal = df.iloc[:, -1].dropna()

    return signal


def process_file(file_path):

    rpm, axis, reading = parse_filename(file_path.name)

    signal = extract_signal(file_path)

    rpm_folder = OUTPUT_DIR / str(rpm)
    rpm_folder.mkdir(parents=True, exist_ok=True)

    save_path = rpm_folder / f"{axis}_{reading}.csv"

    signal.to_csv(save_path, index=False)

    print(f"Saved: {save_path}")


def main():

    OUTPUT_DIR.mkdir(exist_ok=True)

    files = list(RAW_DIR.glob("*.csv"))

    print(f"Found {len(files)} files")

    for file in files:

        try:
            process_file(file)

        except Exception as e:
            print(f"Error processing {file.name}: {e}")


if __name__ == "__main__":
    main()