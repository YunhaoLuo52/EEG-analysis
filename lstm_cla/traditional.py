import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# Update the script to include the TraditionalEEGModel class and run_traditional_ml_model function

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from scipy.signal import welch
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler

class TraditionalEEGModel:
    def __init__(self, k=3):
        self.k = k
        self.model = KNeighborsClassifier(n_neighbors=self.k)
        self.scaler = StandardScaler()

    def extract_features(self, signal, fs=256):
        # Use Welch’s method to calculate the power spectral density
        f, psd = welch(signal, fs=fs, nperseg=128)
        total_power = np.sum(psd)

        mean_freq = np.sum(f * psd) / total_power
        median_freq = f[np.argsort(np.cumsum(psd))[len(psd) // 2]]
        spectral_entropy = entropy(psd / total_power)
        spectral_centroid = np.sum(f * psd) / np.sum(psd)

        return [mean_freq, median_freq, spectral_entropy, spectral_centroid]

    def prepare_dataset(self, data, labels):
        feature_set = []
        for sample in data:
            trimmed = sample[934:]
            features = self.extract_features(trimmed)
            feature_set.append(features)
        feature_set = np.array(feature_set)
        return feature_set, np.array(labels)

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)



# updated_code_svm = updated_code.replace(
#     "from sklearn.neighbors import KNeighborsClassifier",
#     "from sklearn.svm import SVC"
# ).replace(
#     "self.model = KNeighborsClassifier(n_neighbors=self.k)",
#     "self.model = SVC(kernel='linear', probability=True)"
# )


# Define paths
data_path = '/Users/yunhaoluo/Desktop/EEG_analysis/processed_data/'
train_features_path = data_path + "train.csv"
train_labels_path = data_path + "train_labels.csv"
test_features_path = data_path + "test.csv"
test_labels_path = data_path + "test_labels.csv"

X_train = pd.read_csv(train_features_path) 
X_test = pd.read_csv(train_labels_path) 
y_train = pd.read_csv(test_features_path) 
y_test = pd.read_csv(test_labels_path) 

model = TraditionalEEGModel(k=3)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))



