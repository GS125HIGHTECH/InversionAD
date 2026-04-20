import pandas as pd
import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt


class OilDataset(Dataset):
    def __init__(self, data_root, img_size, train=True, transform_type='none', **kwargs):
        self.train = train
        self.data_root = data_root
        self.img_size = img_size
        self.window_size = img_size * img_size
        self.contamination_rate = 0.006

        self._ensure_data_exists()

        file_name = "train_oil.csv" if train else "test_oil.csv"
        csv_path = os.path.join(data_root, file_name)
        df = pd.read_csv(csv_path)
        returns = df['log_returns'].values

        # Pobieramy flagi filtrowania z kwargs
        normal_only = kwargs.get('normal_only', False)
        anom_only = kwargs.get('anom_only', False)

        if not train and 'is_anomaly' in df.columns:
            raw_labels = df['is_anomaly'].values
        else:
            raw_labels = np.zeros(len(returns))

        if train:
            self.mean = float(returns.mean())
            self.std = float(returns.std())
            print(f"--- Statystyki TRENINGOWE obliczone: mean={self.mean:.6f}, std={self.std:.6f} ---")
        else:
            raw_mean = kwargs.get('train_mean')
            raw_std = kwargs.get('train_std')

            if raw_mean is None or raw_std is None:
                print("WARNING: Brak train_mean/std w kwargs! Używam fallback 0.0/1.0 (możliwy błąd!)")
                self.mean = 0.0
                self.std = 1.0
            else:
                self.mean = float(raw_mean)
                self.std = float(raw_std)
                print(f"--- Statystyki TESTOWE wczytane z configu: mean={self.mean:.6f}, std={self.std:.6f} ---")

        normalized_data = (returns - self.mean) / (self.std + 1e-8)

        self.samples = []
        self.final_labels = []
        step = self.window_size

        for i in range(0, len(normalized_data) - self.window_size + 1, step):
            window = normalized_data[i: i + self.window_size]
            label = 1 if np.any(raw_labels[i : i + self.window_size] == 1) else 0

            # --- KLUCZOWY DODATEK: FILTROWANIE ---
            if not train:
                if normal_only and label == 1: continue
                if anom_only and label == 0: continue
            # --------------------------------------

            self.samples.append(window.reshape(1, self.img_size, self.img_size))
            self.final_labels.append(label)

        print(f"Załadowano {'treningowy' if train else 'testowy'} zbiór: {len(self.samples)} okien.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {
            'samples': torch.from_numpy(self.samples[idx]).float(),
            'clslabels': 0,
            'labels': self.final_labels[idx]
        }

    def _ensure_data_exists(self):
        train_path = os.path.join(self.data_root, "train_oil.csv")
        test_path = os.path.join(self.data_root, "test_oil.csv")
        raw_path = os.path.join(self.data_root, "DCOILBRENTEU.csv")

        if os.path.exists(train_path) and os.path.exists(test_path):
            return

        print(f"--- Przygotowuję dane (Isolation Forest) z {raw_path} ---")

        df = pd.read_csv(raw_path)
        df["oil_price"] = pd.to_numeric(df["DCOILBRENTEU"], errors="coerce")
        df = df.drop(columns=["DCOILBRENTEU"])

        df["observation_date"] = pd.to_datetime(df["observation_date"])
        df["oil_price"] = pd.to_numeric(df["oil_price"], errors="coerce")

        # Sprawdzanie ile niepoprawnych danych
        missing_mask = df['oil_price'].isna()
        outlier_mask = (df['oil_price'] <= 0)
        problematic_df = df[missing_mask | outlier_mask].copy()

        if not problematic_df.empty:
            print(f"Znaleziono {len(problematic_df)} błędnych danych.")
            # print(problematic_df[['observation_date', 'oil_price']].to_string(index=False))
        else:
            print("Brak błędów w danych.")

        # Preprocessing (Interpolacja liniowa)
        df_clean = df.copy()
        df_clean['oil_price'] = df_clean['oil_price'].interpolate(method='linear')

        # Sprawdzenie po czyszczeniu
        print(f"\nLiczba braków po czyszczeniu: {df_clean['oil_price'].isna().sum()}")

        # Logarytmiczne stopy zwrotu
        df_clean['log_returns'] = np.log(df_clean['oil_price'] / df_clean['oil_price'].shift(1))

        # Dla pierwszej daty nie ma zmiany, więc wyjdzie NaN)
        df_clean = df_clean.dropna(subset=['log_returns'])

        X = df_clean[['log_returns']].values
        model = IsolationForest(n_estimators=1000, contamination=self.contamination_rate, random_state=42)
        df_clean['anomaly'] = model.fit_predict(X)

        anomalies = df_clean[df_clean['anomaly'] == -1]

        print(f"Liczba anomalii wykrytych przez Isolation Forest: {len(anomalies)}\n")
        # print(anomalies[['observation_date', 'oil_price', 'log_returns']])

        df_clean['is_anomaly'] = (df_clean['anomaly'] == -1).astype(int)
        df_clean['is_anomaly'] = df_clean['is_anomaly'].rolling(window=3, center=True, min_periods=1).max().astype(int)

        window_size = self.window_size
        blocks = []
        block_labels = []

        # Tniemy dane na nienachodzące na siebie bloki, żeby uniknąć wycieku danych
        for i in range(0, len(df_clean) - window_size, window_size):
            block = df_clean.iloc[i: i + window_size].copy()
            # Jeśli w bloku jest choć jedna anomalia, cały blok uznajemy za anomalny
            is_anom = 1 if block['is_anomaly'].any() else 0
            blocks.append(block)
            block_labels.append(is_anom)

        # Rozdzielenie bloków na normalne i anomalne
        normal_blocks = [blocks[i] for i, label in enumerate(block_labels) if label == 0]
        anom_blocks = [blocks[i] for i, label in enumerate(block_labels) if label == 1]

        print(f"Razem bloków: {len(blocks)}")
        print(f"Bloki normalne: {len(normal_blocks)}, Bloki z anomaliami: {len(anom_blocks)}")

        # Tworzenie zbalansowanego zbioru TESTOWEGO (50% normalnych, 50% anomalnych)
        np.random.seed(42)
        n_anom_test = len(anom_blocks)  # bierzemy wszystkie anomalie do testu, żeby było ich jak najwięcej

        test_normal_indices = np.random.choice(len(normal_blocks), size=n_anom_test, replace=False)
        test_normal_blocks = [normal_blocks[i] for i in test_normal_indices]

        train_indices = list(set(range(len(normal_blocks))) - set(test_normal_indices))
        train_blocks = [normal_blocks[i] for i in train_indices]

        df_train = pd.concat(train_blocks).reset_index(drop=True)
        df_test = pd.concat(anom_blocks + test_normal_blocks).reset_index(drop=True)

        df_train.to_csv(train_path, index=False)
        df_test.to_csv(test_path, index=False)
        print("--- Dane przygotowane i zapisane! ---")

        plt.figure(figsize=(18, 8))

        plt.scatter(df_train['observation_date'], df_train['log_returns'],
                    color='royalblue', s=2, label='Training Data (Normal)', alpha=0.5)

        df_test_normal = df_test[df_test['is_anomaly'] == 0]
        plt.scatter(df_test_normal['observation_date'], df_test_normal['log_returns'],
                    color='lightgreen', s=5, label='Test Data (Normal)', alpha=0.7)

        df_test_anom = df_test[df_test['is_anomaly'] == 1]
        plt.scatter(df_test_anom['observation_date'], df_test_anom['log_returns'],
                    color='red', s=10, label='Test Anomalies', zorder=5)

        plt.xlim(df_train['observation_date'].min() - pd.Timedelta(days=100),
                 df_train['observation_date'].max() + pd.Timedelta(days=100))

        plt.title(
            f"Data Distribution: Train ({len(train_blocks)} blocks) vs Test ({len(anom_blocks) + len(test_normal_blocks)} blocks)")
        plt.xlabel("Date")
        plt.ylabel("Log Returns")

        lgnd = plt.legend(loc='upper left')

        for handle in lgnd.legend_handles:
            handle.set_sizes([30.0])
        plt.grid(True, alpha=0.2)
        plt.show()