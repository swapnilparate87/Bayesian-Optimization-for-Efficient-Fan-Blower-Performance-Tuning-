import pandas as pd
from pathlib import Path

from time_features import compute_time_features
from fft_features import compute_fft_features


RAW_SIGNAL_DIR = Path("data/raw_signals")
FLOWRATE_FILE = Path("data/flowrate/RPM vs Flow rate Data.csv")

OUTPUT_FILE = Path("data/processed/final_dataset.csv")


def load_signal(file_path):

    # Skip first row because it contains "Unnamed: 3"
    df = pd.read_csv(file_path, skiprows=1)

    signal = df["Value"].dropna().values

    return signal

    
def extract_features(signal):

    features = {}

    features.update(compute_time_features(signal))
    features.update(compute_fft_features(signal))

    return features


def process_rpm_folder(rpm_folder):

    rpm = int(rpm_folder.name)

    rows = []

    for reading in [1, 2]:

        row = {"RPM": rpm}

        for axis in ["A", "H", "V"]:

            file_path = rpm_folder / f"{axis}_{reading}.csv"

            signal = load_signal(file_path)

            features = extract_features(signal)

            for key, value in features.items():
                row[f"{axis}_{key}"] = value

        rows.append(row)

    return rows


def main():

    dataset = []

    for rpm_folder in RAW_SIGNAL_DIR.iterdir():

        if rpm_folder.is_dir():

            dataset.extend(process_rpm_folder(rpm_folder))

    df = pd.DataFrame(dataset)

    flow_df = pd.read_csv(FLOWRATE_FILE)

    df = df.merge(flow_df, on="RPM")

    df.to_csv(OUTPUT_FILE, index=False)

    print("Dataset saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()