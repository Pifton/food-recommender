import numpy as np
import pandas as pd
import warnings
import sys
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA, KernelPCA

def prepare_data(file, preprocesser, decomposer, params=None):
    np.set_printoptions(threshold=sys.maxsize)
    warnings.filterwarnings('ignore')
    data = pd.read_csv(file, sep=',', quotechar='"')
    values = data.iloc[:, 4:].values
    # Test pour differents scalers
    if preprocesser == "StandardScaler":
        values_scaled = StandardScaler().fit_transform(values)
    elif preprocesser == "MinMaxScaler":
        values_scaled = MinMaxScaler().fit_transform(values)
    elif preprocesser == "RobustScaler":
        values_scaled = RobustScaler().fit_transform(values)
    # Test pour differents decomposers
    # print(values_scaled)
    if decomposer == "PCA":
        pca = PCA(**params)
        values_pca = pca.fit_transform(values_scaled)
        # print(values_pca)
    elif decomposer == "KernelPCA":
        pca = KernelPCA(**params)
        values_pca = pca.fit_transform(values_scaled)
        # print(values_pca)

    labels = data.iloc[:, :3].values

    return data, values_pca, labels