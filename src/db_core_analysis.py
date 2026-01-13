"""
db_core_analysis.py

Implements the core data-processing pipeline, including transformation of the raw dataset into the working dataset,
and defines the primary functions used for database analysis.
"""

# ============================
# Imports
# ============================
import os
import shutil

import numpy as np
import pandas as pd
import operator
from functools import reduce
import json
import itertools

from tinydb import TinyDB, Query

from pymatgen.core import Composition
from matminer.featurizers.composition import Miedema
from matminer.featurizers.composition import WenAlloys

from scipy.signal import savgol_filter

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, ShuffleSplit, KFold, cross_val_predict
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RationalQuadratic, RBF, WhiteKernel
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import classification_report, mean_absolute_error, confusion_matrix
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr
import shap

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import networkx as nx

import pickle

import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
from matplotlib.colors import to_hex, to_rgb
from seaborn import heatmap


# ============================
# Constants / Configuration
# ============================
ATOMIC_MASSES = {
    "H": 1.00794,
    "He": 4.002602,
    "Li": 6.941,
    "Be": 9.0122,
    "B": 10.811,
    "C": 12.0107,
    "N": 14.0067,
    "O": 15.9994,
    "F": 18.9984032,
    "Ne": 20.1797,
    "Na": 22.9898,
    "Mg": 24.305,
    "Al": 26.9815386,
    "Si": 28.0855,
    "P": 30.9738,
    "S": 32.065,
    "Cl": 35.453,
    "K": 39.0983,
    "Ar": 39.948,
    "Ca": 40.078,
    "Sc": 44.9559,
    "Ti": 47.867,
    "V": 50.9415,
    "Cr": 51.9961,
    "Mn": 54.9380,
    "Fe": 55.845,
    "Ni": 58.6934,
    "Co": 58.9332,
    "Cu": 63.546,
    "Zn": 65.409,
    "Ga": 69.723,
    "Ge": 72.630,
    "As": 74.9216,
    "Se": 78.96,
    "Br": 79.904,
    "Kr": 83.798,
    "Rb": 85.4678,
    "Sr": 87.62,
    "Y": 88.9058,
    "Zr": 91.224,
    "Nb": 92.9064,
    "Mo": 95.94,
    "Tc": 98.00,
    "Ru": 101.07,
    "Rh": 102.905,
    "Pd": 106.42,
    "Ag": 107.8682,
    "Cd": 112.411,
    "In": 114.818,
    "Sn": 118.710,
    "Sb": 121.76,
    "I": 126.9045,
    "Te": 127.60,
    "Xe": 131.293,
    "Cs": 132.9054,
    "Ba": 137.327,
    "La": 138.9055,
    "Ce": 140.116,
    "Pr": 140.9077,
    "Nd": 144.242,
    "Pm": 145.00,
    "Sm": 150.36,
    "Eu": 151.965,
    "Gd": 157.25,
    "Tb": 158.9253,
    "Dy": 162.500,
    "Ho": 164.930,
    "Er": 167.259,
    "Tm": 168.9342,
    "Yb": 173.045,
    "Lu": 174.9668,
    "Hf": 178.49,
    "Ta": 180.9479,
    "W": 183.84,
    "Re": 186.207,
    "Os": 190.23,
    "Ir": 192.217,
    "Pt": 195.084,
    "Au": 196.9666,
    "Hg": 200.592,
    "Tl": 204.38,
    "Pb": 207.2,
    "Bi": 208.9804,
    "Th": 232.0377,
    "Pa": 231.0359,
    "U": 238.0289,
    "Np": 237.0482,
    "Pu": 244.0642,
    "Am": 243.0614,
    "Cm": 247.0703,
    "Bk": 247.0703,
    "Cf": 251.0796,
    "Es": 252.0829,
    "Fm": 257.0951,
    "Md": 258.0951,
    "No": 259.1009,
    "Lr": 262.110,
    "Rf": 267.122,
    "Db": 270.133,
    "Sg": 271.133,
    "Bh": 270.133,
    "Hs": 277.15,
    "Mt": 276.15,
    "Ds": 281.16,
    "Rg": 280.16,
    "Cn": 285.17,
    "Nh": 284.18,
    "Fl": 289.19,
    "Mc": 288.19,
    "Lv": 293.20,
    "Ts": 294.21,
    "Og": 294.21
}

IND_ELEMENTS = {
    "Fe": 0, "Mn": 1, "Al": 2, "Si": 3, "Cr": 4,
    "Ni": 5, "C": 6, "Co": 7, "N": 8, "Mo": 9,
    "Cu": 10, "Ti": 11, "V": 12, "Pd": 13, "Zn": 14
}

ELEMENT_LIST = ['Fe', 'Mn', 'Al', 'Si', 'Cr', 'Ni', 'C', 'Co', 'N', 'Mo', 'Cu', 'Ti', 'V', 'Pd', 'Zn']

MATMINER_FEATURES = [
    'Miedema_deltaH_ss_fcc',
    'Miedema_deltaH_ss_bcc',
    'Miedema_deltaH_ss_hcp',
    'Miedema_deltaH_ss_no_latt',
    'Yang delta',
    'Yang omega',
    'APE mean',
    'Radii local mismatch',
    'Radii gamma',
    'Configuration entropy',
    'Lambda entropy',
    'Electronegativity delta',
    'Electronegativity local mismatch',
    'VEC mean',
    'Mixing enthalpy',
    'Mean cohesive energy',
    'Interant electrons',
    'Shear modulus mean',
    'Shear modulus delta',
    'Shear modulus local mismatch',
    'Shear modulus strength model'
]

FCC_LATTICE_PARAMETER = {
    "C": 3.56,  # cubic Fd-3m
    "N": 3.11,
    "Al": 4.04,
    "Si": 3.82,
    "Ti": 4.11,
    "V": 3.79,
    "Cr": 3.58,
    "Mn": 3.47,
    "Fe": 3.66,
    "Co": 3.51,
    "Ni": 3.48,
    "Cu": 3.58,
    "Zn": 3.93,
    "Mo": 3.98,
    "Pd": 3.92
}

HCP_LATTICE_PARAMETER = {
    "C": (2.50, 4.17),  # hex P63/mmc but 4 at/cell
    "N": (3.76, 6.09),
    "Al": (2.81, 4.87),
    "Si": (2.67, 4.51),
    "Ti": (2.94, 4.64),
    "V": (2.60, 4.72),
    "Cr": (2.44, 4.48),
    "Mn": (2.48, 4.03),
    "Fe": (2.43, 3.84),
    "Co": (2.47, 4.02),
    "Ni": (2.45, 4.07),
    "Cu": (2.53, 4.11),
    "Zn": (2.61, 4.87),
    "Mo": (2.77, 4.88),
    "Pd": (2.78, 4.63),
}

CLUSTER_COLORS = [
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#7f7f7f',
    '#bcbd22',
    '#17becf',
    '#ff00ff',
    '#2e8b57'
]

MECHANISM_COLORS = {
    'TWIP': '#66c2a5',
    'TRIPe': '#fc8d62',
    'TRIPa': '#8da0cb',
    'TWIP/TRIPe': '#e78ac3',
    'TWIP/TRIPa': '#a6d854',
    'TRIPe/TRIPa': '#ffd92f',
    'TWIP/TRIPe/TRIPa': '#e5c494',
    'Other': '#b3b3b3'
}

CLUSTER11_LABELS = {
        0: '0: Fe65Cr18Ni12',
        1: '1: Fe77Mn17',
        2: '2: Fe23Co22Cr20Ni20Mn12',
        3: '3: Fe53Mn27Cr6',
        4: '4: Cu93Al6',
        5: '5: Co43Ni35Cr18',
        6: '6: Co89Ni5',
        7: '7: Ni98',
        8: '8: Cu75Zn21',
        9: '9: Ni53Cr18Co9',
        10: '10: Al100'
    }

SAX_LABELS = ['a) ', 'b) ', 'c) ', 'd) ', 'e) ', 'f) ', 'g) ', 'h) ']

MECHANISM_LABELS = [
        'TWIP',
        'TRIPe',
        'TRIPa',
        'TWIP/TRIPe',
        'TWIP/TRIPa',
        'TRIPe/TRIPa',
        'TWIP/TRIPe/TRIPa',
        'Other'
    ]

features_figure_labels = {
    'is_twip': 'TWIP',
    'is_trip_epsilon': r'TRIP_$\epsilon$',
    'is_trip_alpha': r'TRIP_$\alpha$',
    'iSFE_GPR': 'iSFE GPR',
    'gFCC2BCC': r'$\Delta$G$^{fcc-bcc}$',
    'gFCC2HCP': r'$\Delta$G$^{fcc-hcp}$',
    'interface_e_fcc_hcp': r'$\sigma^{fcc-hcp}$',
    'a_fcc_pred': r'a$_{fcc}$',
    'a_hcp_pred': r'a$_{hcp}$',
    'c_hcp_pred': r'c$_{hcp}$',
    'Yang delta': 'Yang delta',
    'Yang omega': 'Yang omega',
    'Mean cohesive energy': 'Mean cohesive energy',
    'Radii local mismatch': 'Radii local mismatch',
    'Radii gamma': 'Radii gamma',
    'Electronegativity delta': 'Electronegativity delta',
    'VEC mean': 'VEC mean',
    'Shear modulus mean': 'Shear modulus mean',
    'Mixing enthalpy': 'Mixing enthalpy'
}


# ============================
# Classes
# ============================
class BaseModel:
    def __init__(self, name, regr, param_grid, scaler=StandardScaler(), scores=None):
        self.name = name
        self.regr = regr
        self.param_grid = param_grid
        self.scaler = scaler
        self.scores = scores

    def set_regr(self, param: dict | GridSearchCV):
        if isinstance(param, dict):
            self.regr = self.regr.__class__(**param)
        else:
            self.regr = param

    def set_scaler(self, scaler=StandardScaler()):
        self.scaler = scaler

    def set_scores(self, scores=None):
        self.scores = scores


class ModelGPR(BaseModel):
    def __init__(self):
        param_grid = {
            'kernel': [
                ConstantKernel(constant_value=1.0) * Matern(nu=0.5) + WhiteKernel(noise_level=1e-3),
                ConstantKernel(constant_value=1.0) * Matern(nu=1.5) + WhiteKernel(noise_level=1e-3),
                ConstantKernel(constant_value=1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-3),
                ConstantKernel(constant_value=1.0) * RationalQuadratic() + WhiteKernel(noise_level=1e-3),
                ConstantKernel(constant_value=1.0) * RBF() + WhiteKernel(noise_level=1e-3)
            ],
            'alpha': [1e-10],  # Regularization parameter
            'normalize_y': [True],
            'n_restarts_optimizer': [10],  # Number of restarts for optimizer
        }
        super().__init__('GPR', GaussianProcessRegressor(), param_grid)


class ModelHGBR(BaseModel):
    def __init__(self):
        param_grid = {
            'max_iter': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_leaf': [10, 20],
            'learning_rate': [0.1],
            'l2_regularization': [0, 0.1, 0.5]
        }
        super().__init__('HGBR', HistGradientBoostingRegressor(), param_grid)


class ModelRFR(BaseModel):
    def __init__(self):
        param_grid = {
            # 'n_estimators': [50, 100, 200],
            'n_estimators': [50],
            'criterion': ['squared_error', 'friedman_mse'],
            'max_depth': [None, 10, 20],
            # 'min_samples_split': [2, 5, 10],
            'min_samples_split': [2],
            # 'min_samples_leaf': [1, 2, 4, 10, 20],
            'min_samples_leaf': [1],
            'max_features': ['sqrt', 1, 2]
        }
        super().__init__('RFR', RandomForestRegressor(), param_grid)


class ModelADAR(BaseModel):
    def __init__(self):
        param_grid = {
            'estimator': [DecisionTreeRegressor(max_depth=8),
                          DecisionTreeRegressor(max_depth=12),
                          DecisionTreeRegressor(max_depth=8, min_samples_leaf=2)],
            'n_estimators': [100, 200],
            'learning_rate': [0.1, 1.0, 2.0],
            'loss': ['linear', 'square', 'exponential']
        }
        super().__init__('ADAR', AdaBoostRegressor(), param_grid)


class ModelKRR(BaseModel):
    def __init__(self):
        param_grid = {
            'alpha': [1e0, 1e1, 1e2, 1e3],
            'kernel': ['poly'],
            'degree': [2, 3, 4, 5],
            'coef0': [0, 1, 5, 20, 30],
            'gamma': [0.1, 0.5, 0.75, 1]
        }
        super().__init__('KRR', KernelRidge(), param_grid)


class ModelSVM(BaseModel):
    def __init__(self):
        param_grid = {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [3, 4, 5]
        }
        super().__init__('SVM', SVR(), param_grid)


class ModelMLPR(BaseModel):
    def __init__(self):
        param_grid = {
            'hidden_layer_sizes': [(150,)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd'],
            'alpha': [1e-1],
            'max_iter': [10000],
            'learning_rate': ['adaptive']
        }
        super().__init__('MLPR', MLPRegressor(), param_grid)


# ============================
# Utility scripts
# ============================
def lattice_param_fcc(comp):
    """
    Computes the FCC lattice parameter using a rule-of-mixture based on elemental FCC lattice parameters

    """
    lattice_param = 0.0
    for elt in comp:
        if (comp[elt] is not None) and (not np.isnan(comp[elt])):
            lattice_param += comp[elt] * FCC_LATTICE_PARAMETER[elt]

    return lattice_param


def lattice_param_hcp(comp, perfect=True):
    """
    Computes the HCP lattice parameter using a rule-of-mixture based on elemental HCP lattice parameters

    """
    lattice_param_a = 0.0
    lattice_param_c = 0.0
    for elt in comp:
        if (comp[elt] is not None) and (not np.isnan(comp[elt])):
            lattice_param_a += comp[elt] * HCP_LATTICE_PARAMETER[elt][0]
            if perfect:
                lattice_param_c += comp[elt] * HCP_LATTICE_PARAMETER[elt][0] * (5 / 3)  # assumed compact hcp packing
            else:
                lattice_param_c += comp[elt] * HCP_LATTICE_PARAMETER[elt][1]

    return lattice_param_a, lattice_param_c


def b_p(row):
    return 0.1 * np.sqrt(6) * lattice_param_fcc(row[ELEMENT_LIST].to_dict()) / 6


def composition_conversion(composition: dict, to_at: bool):
    """
    Converts composition from at.T to wt.% or inversely

    """
    if to_at:
        # Weight fraction to atomic fraction
        sum_am_wt = 0
        for element in composition:
            sum_am_wt += np.array(composition[element]) / ATOMIC_MASSES[element]
        composition_out = {k: np.array(composition[k]) / ATOMIC_MASSES[k] / sum_am_wt
                           for k in composition}
    else:
        # Atomic fraction to weight fraction
        sum_am_at = 0
        for element in composition:
            sum_am_at += np.array(composition[element]) * ATOMIC_MASSES[element]
        composition_out = {k: np.array(composition[k]) * ATOMIC_MASSES[k] / sum_am_at
                           for k in composition}

    return composition_out


def ordered_features(df):
    """
    Orders the dataframe columns based on ELEMENT_LIST and MATMINER_FEATURES

    """
    if all([mf in df for mf in MATMINER_FEATURES]):
        return df[[*ELEMENT_LIST, *MATMINER_FEATURES]]
    else:
        return df[ELEMENT_LIST]


def moving_averaged_array(input_list):
    """
    Computes a moving average

    """
    input_array = np.array(input_list)
    return (input_array[:-1] + input_array[1:]) / 2


def sort_xy_lists(array1, array2):
    if array1.size > 1:
        comb = sorted(zip(array1, array2))
        sorted_list1, sorted_list2 = zip(*comb)

        return np.array(sorted_list1), np.array(sorted_list2)
    else:
        return array1, array2


def substract_series(s1, s2):
    if isinstance(s1, pd.Series):
        result = s1.combine(s2, lambda lst, val: (np.array(lst) - val).tolist())
    elif isinstance(s1, list):
        result = [s - val for s, val in zip(s1, s2)]
    else:
        return TypeError
    return result


def df_to_latex_stats(df, row_columns, cluster_col, decimals=0):
    """
    Creates a easy-to-print output to represent a dataframe as a Latex table

    """
    # Build aggregation dict
    agg_dict = {
        col: ['mean', 'std', 'count']
        for col in df.select_dtypes(include='number')
        if col != cluster_col
    }

    # Aggregate by cluster
    stats = df.groupby(cluster_col).agg(agg_dict)
    stats.columns = [f"{col}_{stat}" for col, stat in stats.columns]

    # Format mean and std columns
    for col in stats.columns:
        if col.endswith('_mean') or col.endswith('_std'):
            if any(elt in col for elt in ELEMENT_LIST):
                stats[col] = stats[col].apply(lambda x: f"{100 * x:.{decimals}f}")
            else:
                stats[col] = stats[col].apply(lambda x: f"{1 * x:.{decimals}f}")

    # Combine mean ± std into single columns
    mean_std_cols = []
    new_columns = {}
    for col in df.select_dtypes(include='number'):
        if col == cluster_col:
            continue
        mean_col = f"{col}_mean"
        std_col = f"{col}_std"
        if mean_col in stats and std_col in stats:
            new_col = f"{col}"
            new_columns[new_col] = stats[mean_col].astype(str) + " ± " + stats[std_col].astype(str)
            mean_std_cols.append(new_col)
    stats = stats.assign(**new_columns)

    # Get count column (from any numeric column)
    first_numeric = next(c for c in df.select_dtypes(include='number') if c != cluster_col)
    count_col = f"{first_numeric}_count"

    # Build final table
    final_stats = stats[[count_col] + mean_std_cols].rename(columns={count_col: 'count'})

    # Restrict to requested columns as rows (transpose)
    final_stats_T = final_stats.loc[:, ['count'] + row_columns].T
    final_stats_T.index.name = "Parameter"

    # Output LaTeX
    latex_table = final_stats_T.to_latex(index=True, escape=False)
    return latex_table


def copy_database_if_not_exists(source, destination):
    """
    Copies the source database to the destination database if the second does not currently exist

    """
    if not os.path.exists(destination):
        shutil.copy(source, destination)


def e_strain_pierce(afcc, ahcp, chcp, nu, shear_modulus):
    n_avogadro = 6.022e23

    vmfcc = afcc ** 3 * n_avogadro / 4
    vmhcp = np.sqrt(3) * ahcp ** 2 * chcp * n_avogadro / 4
    vs = (vmhcp - vmfcc) / vmfcc

    cfcc = 2 * afcc / np.sqrt(3)
    e33 = (chcp - cfcc) / cfcc
    e11 = (vs - e33) / 2

    eta = (7 - 5 * nu) / 15 / (1 - nu)

    e_dil = 2 * (1 + nu) * shear_modulus * vmfcc * vs ** 2 / 9 / (1 - nu)
    e_sh = eta * vmfcc * 2 * shear_modulus * (2 * (e11 - e33) ** 2) / 6

    return e_dil + e_sh


def perform_cross_validation(dataframe, target, model=ModelGPR(), scale=True, scoring='neg_mean_absolute_error',
                             plot_splits=True, plot_shap=False, plot_confusion_matrix=False,
                             save_model=True, save_path='data/models/', model_description=''):
    """
    Performs cross validation to find the best hyperparameters for a given model

    """
    x_array = dataframe.drop(target, axis=1)
    y = dataframe[target]

    if scale:
        scaler = StandardScaler()
        x_array = pd.DataFrame(scaler.fit_transform(x_array), columns=x_array.columns)
        model.set_scaler(scaler)

    # split strategy
    split_strategy = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

    # hyperparam search
    grid_search = GridSearchCV(model.regr, model.param_grid, cv=split_strategy, scoring=scoring, n_jobs=-1)
    grid_search.fit(x_array, y)

    best_params = grid_search.best_params_
    print(f'Best parameters for {model.name}: {best_params}')
    print(f'Best CV score for {model.name}: {grid_search.best_score_}')

    metrics = {
        'MAPE': [],
        'MAE': [],
        'RMSE': [],
        'Spearman': [],
        'R2': []
    }
    train_sets = []

    if plot_splits:
        fig, ax = plt.subplots()
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')

    confusion_matrices = []
    for train_index, test_index in split_strategy.split(x_array):
        x_train, x_test = x_array.iloc[train_index], x_array.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.set_regr(best_params)
        model.regr.fit(x_train, y_train)
        y_pred = model.regr.predict(x_test)

        if plot_splits:
            ax.scatter(y_test, y_pred, alpha=0.7)

        metrics['MAPE'].append(mean_absolute_percentage_error(y_test, y_pred))
        metrics['MAE'].append(mean_absolute_error(y_test, y_pred))
        metrics['RMSE'].append(root_mean_squared_error(y_test, y_pred))
        metrics['Spearman'].append(spearmanr(y_test, y_pred)[0])
        metrics['R2'].append(r2_score(y_test, y_pred))

        train_sets.append([x_train, y_train])

        if plot_confusion_matrix:
            threshold = 0.5
            confusion_matrices.append(confusion_matrix(y_pred >= threshold, y_test >= threshold, labels=[0, 1]))

    if plot_confusion_matrix:
        _, ax_cm = plt.subplots(1, 1, figsize=(6, 6))
        heatmap(np.sum(confusion_matrices, axis=0), annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'], ax=ax_cm)
        ax_cm.set(xlabel='Predicted Label', ylabel='True Label')

    if plot_splits:
        ax.set(xlim=[y.min(), y.max()], ylim=[y.min(), y.max()],
               xlabel='True Values', ylabel='Predictions')
        ax.set_title(f'{model.name} - Cross-Validation Splits')
        fig.tight_layout()
        plt.show()

    # Saving best model based on the best MAE score for one split (based on lowest MAE)
    best_idx = np.argmin(metrics['MAE'])
    x_train_best, y_train_best = train_sets[best_idx]

    model.set_regr(best_params)
    model.regr.fit(x_train_best, y_train_best)

    if save_model:
        model_filename = f'{save_path}model_{target}_{model.name}_{model_description}_bestSplit.pkl'
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)

    if plot_splits:
        fig_best, ax_best = plt.subplots()
        ax_best.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')

        x_test_best = x_array.drop(x_train_best.index)
        y_test_best = y.drop(y_train_best.index)

        y_train_best_pred = model.regr.predict(x_train_best)
        y_test_best_pred = model.regr.predict(x_test_best)
        ax_best.scatter(y_train_best, y_train_best_pred, alpha=0.7, label='train')
        ax_best.scatter(y_test_best, y_test_best_pred, alpha=0.7, label='test')
        ax_best.set(xlim=[y.min(), y.max()], ylim=[y.min(), y.max()])
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.legend(loc='lower right')
        plt.title(f'{model.name} - CV Best Split')
        fig_best.tight_layout()
        plt.show()

    scores = pd.DataFrame(metrics)
    print(f'Average mape across folds with best model: {np.mean(scores['MAPE']):.4f}')
    print(f'Average mae across folds with best model: {np.mean(scores['MAE']):.4f}')
    print(f'Average rmse across folds with best model: {np.mean(scores['RMSE']):.4f}')
    print(f'Average spearmanr across folds with best model: {np.mean(scores['Spearman']):.4f}')
    print(f'Average R² across folds with best model: {np.mean(scores['R2']):.4f}')

    model.set_scores(scores)

    # Fit model with grid_search best params (refit on all data)
    model.set_regr(grid_search)

    if save_model:
        model_filename = f'{save_path}model_{target}_{model.name}_{model_description}.pkl'
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)

    if plot_shap:
        my_cmap = plt.get_cmap("RdYlBu_r")
        if 'RFR' in model.name or 'ADAR' in model.name or 'HGBR' in model.name:
            # explainer = shap.TreeExplainer(model.regr.best_estimator_, x_train_best)
            explainer = shap.TreeExplainer(model.regr.best_estimator_, x_array)
            shap_values = explainer.shap_values(x_array)
            shap.summary_plot(shap_values, x_array, cmap=my_cmap)
        else:
            best_idx = np.argmin(metrics['MAE'])
            x_train_best, y_train_best = train_sets[best_idx]
            x_test_best = x_array.drop(x_train_best.index)

            x_train_summary = shap.kmeans(x_train_best, 10)

            explainer = shap.KernelExplainer(model.regr.best_estimator_.predict, x_train_summary)
            shap_values = explainer.shap_values(x_test_best)
            shap.summary_plot(shap_values, x_test_best, cmap=my_cmap)

        return shap_values, explainer


def build_query(conditions, _qry=None):
    """
    Builds complex TinyDB query

    """
    if _qry is None:
        _qry = Query()

    op_map = {
        'eq': operator.eq,  # equal to              (a == b)
        'ne': operator.ne,  # not equal to          (a != b)
        'lt': operator.lt,  # less than             (a < b)
        'le': operator.le,  # less or equal to      (a <= b)
        'gt': operator.gt,  # greater than          (a > b)
        'ge': operator.ge,  # greater or equal to   (a >= b)
    }

    query_parts = []

    for cond in conditions:
        feature_path = cond["feature"].split('.')
        # q = reduce(getattr, feature_path, Query())
        q = reduce(lambda obj, attr: getattr(obj, attr), feature_path, Query())

        if cond["op"] in op_map:
            q = op_map[cond["op"]](q, cond["value"])
        elif cond["op"] == "between":
            _low, _high = cond["value"]
            q = q.test(lambda v, low=_low, high=_high: low <= v <= high)
        elif cond["op"] == "exists":
            q = q.exists()
        elif cond["op"] == "not_nan":
            q = q.test(lambda v: v is not None and not (isinstance(v, float) and np.isnan(v)))
        elif cond["op"] == "not_elt":
            # Include if value does not exist or is less than specified
            q = ~q.exists() | q.test(lambda v, val=cond["value"]: v < val)
        else:
            raise ValueError(f"Unsupported operation: {cond['op']}")

        query_parts.append(q)

    if not query_parts:
        return _qry

    return reduce(operator.and_, query_parts)


def get_nested(item, field_path, default=None):
    """
    Safely get a nested field from a dict given a dotted path.

    """
    try:
        return reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default,
                      field_path.split('.'),
                      item)
    except Exception:
        return default


def create_df_from_feature_set(
        feature_set, source="data/data_ext.json", export_to_csv=False, export_to_pkl=False):
    """
    Creates a dataframe containing the subset based on feature conditions

    """
    db = TinyDB(source)

    qry = build_query(feature_set['conditions'])
    entries = db.search(qry)

    rows = []
    for item in entries:
        row = {}
        for feature in feature_set['features']:
            if feature == 'doc_id':
                row['doc_id'] = item.doc_id  # TinyDB specific
            else:
                feature_name = feature.split('.')
                row[feature_name[-1]] = get_nested(item, feature)
        rows.append(row)

    df = pd.DataFrame(rows)

    if export_to_csv:
        filename = input("Enter save filename:")
        df.to_csv(f'{filename}.csv')
    if export_to_pkl:
        filename = input("Enter save filename:")
        df.to_pickle(f'{filename}.pkl')

    return df


def create_sfe_df(source="data/data_dev.json", composition_key='composition_at',
                  is_featurized=False, sfe_upper_bound=None,
                  export_to_csv=False, export_to_pkl=False):
    """
    Creates a dataframe containing the subset with SFE data

    """
    db_dev = TinyDB(source)
    qry = Query()

    if sfe_upper_bound is not None:
        condition = ((qry.thermodynamic_param.iSFE > -1000) &
                     (qry.thermodynamic_param.iSFE < sfe_upper_bound))
    else:
        condition = qry.thermodynamic_param.iSFE > -1000
    entries = db_dev.search(condition)

    array = np.zeros((len(entries), len(IND_ELEMENTS) + 1))
    for i, item in enumerate(entries):
        for elt in item[composition_key]:
            array[i, IND_ELEMENTS[elt]] = item[composition_key][elt]
        array[i, -1] = item['thermodynamic_param']['iSFE']

    df = pd.DataFrame(array, columns=[*ELEMENT_LIST, 'iSFE'])

    if is_featurized:
        array_mmf = np.zeros((len(entries), len(MATMINER_FEATURES)))
        for i, item in enumerate(entries):
            for j, feature in enumerate(MATMINER_FEATURES):
                array_mmf[i, j] = item['matminer_features'][feature]

        dataframe_mmf = pd.DataFrame(array_mmf, columns=[*MATMINER_FEATURES])
        df = pd.concat([df, dataframe_mmf], axis=1)

    if export_to_csv:
        filename = input("Enter save filename:")
        df.to_csv(f'{filename}.csv')
    if export_to_pkl:
        filename = input("Enter save filename:")
        df.to_pickle(f'{filename}.pkl')

    return df


# ============================
# Data-processing
# ============================
def update_composition(source_db_ori="data/data_raw.json", source_db_dev="data/data_dev.json"):
    """
    Adds or fills in the compositions in wt.% and in at.% to the dev database

    """
    if not os.path.exists(source_db_dev):
        scr = source_db_ori
        dst = source_db_dev
        shutil.copy(scr, dst)

    db_dev = TinyDB(source_db_dev)

    entries = db_dev.all()
    for item in entries:
        if 'composition' in item:
            composition_wt_update = item['composition']

            for element in item['composition'].copy():
                if item['composition'][element] == 0:
                    del composition_wt_update[element]

            # add Fe in case it has been used as bal.
            if 'Fe' not in composition_wt_update:
                x_fe = 1
                for element in composition_wt_update:
                    x_fe = x_fe - composition_wt_update[element]
                if x_fe > 1e-4:
                    composition_wt_update["Fe"] = x_fe  # adds Fe only if w(Fe) > 0.1 wt%

            # updates composition in wt, adding Fe and removing unused elements
            db_dev.update({'composition': composition_wt_update}, doc_ids=[item.doc_id])

            if 'composition_at' not in item or not item['composition_at']:
                # wt to at
                composition_at_update = composition_conversion(composition_wt_update, to_at=True)

                db_dev.update({'composition_at': composition_at_update}, doc_ids=[item.doc_id])

        if 'composition_at' in item:
            composition_at_update = item['composition_at']

            for element in item['composition_at'].copy():
                if item['composition_at'][element] == 0:
                    del composition_at_update[element]

            # add Fe in case it has been used as balance element
            if 'Fe' not in composition_at_update:
                x_fe = 1
                for element in composition_at_update:
                    x_fe = x_fe - composition_at_update[element]
                if x_fe > 1e-4:
                    composition_at_update["Fe"] = x_fe  # adds Fe only if x(Fe) > 0.1 at%

            # updates composition in at, adding Fe and removing unused elements
            db_dev.update({'composition_at': composition_at_update}, doc_ids=[item.doc_id])

            if 'composition' not in item or not item['composition']:
                # at to wt
                composition_wt_update = composition_conversion(composition_at_update, to_at=False)

                db_dev.update({'composition': composition_wt_update}, doc_ids=[item.doc_id])


def update_matminer_features(source="data/data_dev.json"):
    """
    Adds MATMINER features to all entries of the database

    """
    db = TinyDB(source)
    entries = db.all()

    item_ids = []
    comp_pmg = []
    for i, item in enumerate(entries):
        comp_dic = item['composition_at'].copy()

        comp_pmg.append(Composition(comp_dic))
        item_ids.append(item.doc_id)

    miedema_featurizer = Miedema(struct_types='ss', ss_types='all')
    wen_featurizer = WenAlloys()

    df = pd.DataFrame()
    df['comp'] = comp_pmg
    df['id'] = item_ids
    df = miedema_featurizer.featurize_dataframe(df, col_id='comp')
    df = wen_featurizer.featurize_dataframe(df, col_id='comp')

    for index, doc_id in df['id'].items():
        features = {}
        for feature in MATMINER_FEATURES:
            features[feature] = df[feature][index]
        db.update({'matminer_features': features}, doc_ids=[doc_id])


def update_mech_param(source_db_ori="data/data_raw.json", source_db_dev="data/data_dev.json"):
    """
    Computes engineering and true tensile properties

    """
    if not os.path.exists(source_db_dev):
        scr = source_db_ori
        dst = source_db_dev
        shutil.copy(scr, dst)

    db_dev = TinyDB(source_db_dev)

    entries_to_update = db_dev.all()
    for item in entries_to_update:
        # define variables
        engineering_strain = np.array(item['mechanical_param'].get('engineering_strain', [float('nan')]))
        engineering_stress = np.array(item['mechanical_param'].get('engineering_stress', [float('nan')]))
        true_strain = np.array(item['mechanical_param'].get('true_strain', [float('nan')]))
        true_stress = np.array(item['mechanical_param'].get('true_stress', [float('nan')]))
        yield_strength = np.array(item['mechanical_param'].get('yield_strength', [float('nan')]))
        true_uniform_strain = np.array(item['mechanical_param'].get('true_uniform_strain', [float('nan')]))
        true_ultimate_stress = \
            np.array(item['mechanical_param'].get('true_ultimate_stress', [float('nan')]))

        true_strain, true_stress = sort_xy_lists(true_strain, true_stress)
        engineering_strain, engineering_stress = sort_xy_lists(engineering_strain, engineering_stress)

        # if variable = np.array(float('nan')), variable.size is 1
        if (engineering_stress.size > 1) & (not true_stress.size > 1):
            # express engineering strains as - instead of %
            if max(engineering_strain) > 5:
                engineering_strain = engineering_strain / 100

            # derive true stress and true strain from engineering stress and engineering strain
            ind_max = np.argmax(engineering_stress)
            true_strain = np.log(engineering_strain[0:ind_max] + 1)
            true_stress = engineering_stress[0:ind_max] * (engineering_strain[0:ind_max] + 1)
        elif (not engineering_stress.size > 1) & (true_stress.size > 1):
            # express true strains as - instead of %
            if max(true_strain) > 5:
                true_strain = true_strain / 100

            # derive engineering stress and engineering strain from true stress and true strain
            ind_max = np.argmax(true_stress)
            engineering_strain = np.exp(true_strain[0:ind_max]) - 1
            engineering_stress = np.divide(true_stress[0:ind_max], engineering_strain + 1)

        if np.isnan(yield_strength) & (true_stress.size > 1):
            true_strain_filt = savgol_filter(true_strain, 11, 3)
            true_stress_filt = savgol_filter(true_stress, 11, 3)

            ds = np.diff(true_stress_filt)
            de = np.diff(true_strain_filt)
            if len(ds) != len(de):
                print(item)

            true_strain_avg = moving_averaged_array(true_strain_filt)
            true_stress_avg = moving_averaged_array(true_stress_filt)
            dsde = np.divide(ds, de, out=np.zeros_like(ds), where=de != 0)
            dsde_e_s = np.divide(np.multiply(dsde, true_strain_avg), true_stress_avg)

            dsde_e_s_filt = savgol_filter(dsde_e_s, 11, 3)
            ys_ind = []
            for ind in range(10):
                tmp = np.argmin(dsde_e_s[ind:][dsde_e_s[ind:] > 0][true_stress_avg[ind:][dsde_e_s[ind:] > 0] < 750])
                if tmp > 3:
                    ys_ind.append(tmp - 1 + ind)
                else:
                    ys_ind.append(0)
            yield_strength = true_stress_avg[dsde_e_s > 0][np.max(ys_ind)]

            if (yield_strength < 200) | (yield_strength > 650):
                print(yield_strength)

                fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                ax[0].plot(engineering_strain, engineering_stress, 'o-')
                ax[0].plot([0, 0.15], [yield_strength, yield_strength])
                ax[0].set(ylim=(0, None))

                ax[1].plot(true_stress_avg, dsde_e_s, 'o-')
                ax[1].plot([yield_strength, yield_strength], [0, 1])
                ax[1].plot(true_stress_avg, dsde_e_s_filt, 'o-')

                plt.show()
                yield_strength = np.float64(input('The YS is:\n'))

        if np.isnan(true_uniform_strain):
            true_uniform_strain = np.max(true_strain)

        if np.isnan(true_ultimate_stress):
            true_ultimate_stress = np.max(true_stress)

        if (not np.isnan(yield_strength)) & (true_stress.size > 1):
            true_strain_filt = savgol_filter(true_strain, 11, 3)
            true_stress_filt = savgol_filter(true_stress, 11, 3)

            ds = np.diff(true_stress_filt)
            de = np.diff(true_strain_filt)
            dsde = np.divide(ds, de, out=np.zeros_like(ds), where=de != 0)

            true_strain_avg = moving_averaged_array(true_strain_filt)
            true_stress_avg = moving_averaged_array(true_stress_filt)

            ind_pl = true_stress_avg > yield_strength

            # incremental hardening coefficient (d(sigma)/d(epsilon))*(epsilon/sigma)
            nincr = np.divide(np.multiply(dsde[ind_pl], true_strain_avg[ind_pl]), true_stress_avg[ind_pl])

            if len(nincr) == 0:
                print(f'{nincr=}')

            # plastic work (integral of the (plastic) stress-strain curve)
            plastic_work = np.trapz(true_stress_avg[ind_pl], true_strain_avg[ind_pl])
        else:
            nincr = np.nan
            plastic_work = np.nan

        mechanical_param_update = item['mechanical_param']
        mechanical_param_update['engineering_strain'] = engineering_strain.tolist()
        mechanical_param_update['engineering_stress'] = engineering_stress.tolist()
        mechanical_param_update['true_strain'] = true_strain.tolist()
        mechanical_param_update['true_stress'] = true_stress.tolist()
        mechanical_param_update['yield_strength'] = yield_strength.tolist()
        mechanical_param_update['true_uniform_strain'] = true_uniform_strain.tolist()
        mechanical_param_update['true_ultimate_stress'] = true_ultimate_stress.tolist()
        mechanical_param_update['nincr_mean'] = np.mean(nincr[nincr > 0])
        mechanical_param_update['nincr_max'] = np.max(nincr)
        mechanical_param_update['plastic_work'] = plastic_work

        db_dev.update({'mechanical_param': mechanical_param_update}, doc_ids=[item.doc_id])


def add_serrated_yielding_bool():
    """
    Goes through each entry of the source db to manually add serrated_yielding to mechanical_param.

    """
    source_db_ori = "data/data_raw.json"
    db_ori = TinyDB(source_db_ori)

    entries_to_update = db_ori.all()
    for item in entries_to_update:
        mechanical_param_update = item['mechanical_param']

        cond = False
        if "engineering_strain" in mechanical_param_update:
            if isinstance(mechanical_param_update['engineering_strain'], list):
                cond = True
        if "true_strain" in mechanical_param_update:
            if isinstance(mechanical_param_update['true_strain'], list):
                cond = True
        if cond and ("serrated_yielding" not in mechanical_param_update):
            print(item['zotero_reference'])
            if 'composition' in item:
                print(item['composition'])
            elif 'composition_at' in item:
                print(item['composition_at'])

            serrated_yielding = input('Serrated yielding? :\n')

            mechanical_param_update['serrated_yielding'] = int(serrated_yielding)

            db_ori.update({'mechanical_param': mechanical_param_update}, doc_ids=[item.doc_id])


def update_defmech(source_db="data/data_dev.json"):
    """
    Goes through each entry of the source db to manually add is_twip, is_trip_epsilon, and is_trip_alpha to
    mechanical_param.

    1 if the mechanism is observed,
    0 if the mechanism is not observed,
    0.5 if unconfirmed but probable.

    """
    db = TinyDB(source_db)

    qry = Query()
    condition = qry.mechanical_param.true_stress.any(float)

    entries = db.search(condition)

    save_doc_id = []
    save_zotero_reference = []
    save_composition_at = []
    save_defmech = []
    for item in entries:
        print(item['zotero_reference'])
        print(item['composition_at'])

        save_doc_id.append(item.doc_id)
        save_zotero_reference.append(item['zotero_reference'])
        save_composition_at.append(item['composition_at'])

        # is_twip_tripe_tripa must be a list of 3 int
        is_twip_tripe_tripa = list(map(int, input('1: true, 0: false, 0.5: possible\ntwip | trip_e | trip_a\n').split()))
        save_defmech.append(is_twip_tripe_tripa)

        mechanical_param_update = item['mechanical_param']
        mechanical_param_update['is_twip'] = is_twip_tripe_tripa[0]
        mechanical_param_update['is_trip_epsilon'] = is_twip_tripe_tripa[1]
        mechanical_param_update['is_trip_alpha'] = is_twip_tripe_tripa[2]

        db.update({'mechanical_param': mechanical_param_update}, doc_ids=[item.doc_id])

    save_df = pd.DataFrame([save_doc_id, save_zotero_reference, save_composition_at, save_defmech])
    save_df = save_df.T
    save_df.columns = ['doc_id', 'zotero_reference', 'composition_at', 'deformation_mech']
    save_df.to_pickle('df_deformation_mech_backup.pkl')


def update_gibbs_tcpython(input_gibbs='data/df_equilibria_with_gibbs.pkl', source_db="data/data_dev.json"):
    """
    Reads the TC-python outputs to update the database with Gibbs energy of phases FCC, BCC, and HCP as well as the
    differences in energies between FCC and HCP, FCC and BCC, and HCP and BCC

    """
    df = pd.read_pickle(input_gibbs)
    db = TinyDB(source_db)

    doc_ids = df['doc_id']
    gfccs = df['gFCC']
    gbccs = df['gBCC']
    ghcps = df['gHCP']
    for doc_id, gfcc, gbcc, ghcp in zip(doc_ids, gfccs, gbccs, ghcps):
        item = db.get(doc_id=doc_id)

        thermodynamic_para_update = item['thermodynamic_param']
        thermodynamic_para_update['gFCC'] = gfcc
        thermodynamic_para_update['gBCC'] = gbcc
        thermodynamic_para_update['gHCP'] = ghcp
        thermodynamic_para_update['gFCC2HCP'] = gfcc - ghcp
        thermodynamic_para_update['gFCC2BCC'] = gfcc - gbcc
        thermodynamic_para_update['gHCP2BCC'] = ghcp - gbcc

        db.update({'thermodynamic_param': thermodynamic_para_update}, doc_ids=[doc_id])


def update_interface_energy(input_interface='data/df_interfaceE20250729_300.pkl', source_db="data/data_dev.json"):
    df_interface = pd.read_pickle(input_interface)
    db = TinyDB(source_db)

    doc_ids = df_interface['doc_id']
    interface_e_fcc_hcp = df_interface['interface_e_fcc_hcp']

    for doc_id, sig_i in zip(doc_ids, interface_e_fcc_hcp):
        item = db.get(doc_id=doc_id)

        thermodynamic_para_update = item['thermodynamic_param']
        thermodynamic_para_update['interface_e_fcc_hcp'] = sig_i

        db.update({'thermodynamic_param': thermodynamic_para_update}, doc_ids=[doc_id])


def update_lattice_parameter(source_db="data/data_dev.json"):
    db_ext = TinyDB(source_db)
    entries = db_ext.all()

    for item in entries:
        thermodynamic_para_update = item['thermodynamic_param']
        comp = item['composition_at']

        a_fcc = lattice_param_fcc(comp) * 1e-10
        a_hcp, c_hcp = lattice_param_hcp(comp, perfect=False)
        a_hcp *= 1e-10
        c_hcp *= 1e-10

        thermodynamic_para_update['a_fcc_pred'] = a_fcc
        thermodynamic_para_update['a_hcp_pred'] = a_hcp
        thermodynamic_para_update['c_hcp_pfct_pred'] = (5 / 3) * a_hcp
        thermodynamic_para_update['c_hcp_pred'] = c_hcp

        db_ext.update({'thermodynamic_param': thermodynamic_para_update}, doc_ids=[item.doc_id])


def extrapol_sfe_models(model_str_list, df_list, isfe_str_list, destination="data/data_ext.json"):
    """
    Computes the SFE based on composition using a given model

    """
    db_ext = TinyDB(destination)

    item_ids = df_list[0]['itemID']
    if len(df_list) > 1:
        if not all(item_ids.equals(df['itemID']) for df in df_list[1:]):
            raise ValueError("itemID columns do not match across all DataFrames")

    df_ext = pd.DataFrame()
    df_ext['itemID'] = item_ids
    for model_name, df, isfe_str in zip(model_str_list, df_list, isfe_str_list):
        with open(f'data/models/{model_name}.pkl', 'rb') as f:
            model = pickle.load(f)
        df.fillna(0, inplace=True)
        df = ordered_features(df)
        if model.scaler is not None:
            df = pd.DataFrame(model.scaler.transform(df), columns=df.columns)

        df_ext[isfe_str] = model.regr.predict(df)

    dict_ext = df_ext.set_index('itemID')[isfe_str_list].to_dict(orient='index')
    for e in dict_ext:
        item = db_ext.get(doc_id=e)

        thermodynamic_para_update = item['thermodynamic_param']
        for sfe in dict_ext[e]:
            thermodynamic_para_update[sfe] = dict_ext[e][sfe]

        db_ext.update({'thermodynamic_param': thermodynamic_para_update}, doc_ids=[item.doc_id])


def extrapol_sfe_gibbs(destination="data/data_ext.json"):
    """
    Computes the SFE based on a thermodynamic description

    """

    db_ext = TinyDB(destination)
    entries = db_ext.all()

    for item in entries:
        thermodynamic_para_update = item['thermodynamic_param']

        n_avogadro = 6.022e23

        comp = item['composition_at']

        a_fcc = lattice_param_fcc(comp) * 1e-10
        a_hcp, c_hcp = lattice_param_hcp(comp)
        a_hcp *= 1e-10
        c_hcp *= 1e-10

        rho = 4 / np.sqrt(3) / a_fcc ** 2 / n_avogadro

        e_str = 10 * e_strain_pierce(a_fcc, a_hcp, c_hcp, 0.3, 1e6 * item['matminer_features']['Shear modulus mean'])

        sigma_fcc2hcp = 100 * item['thermodynamic_param']['interface_e_fcc_hcp']
        if np.isnan(sigma_fcc2hcp):
            sigma_fcc2hcp = 8

        isfe_gibbs = 2000 * rho * (-item['thermodynamic_param']['gFCC2HCP'] + e_str) + 2 * sigma_fcc2hcp

        thermodynamic_para_update['iSFE_TC'] = isfe_gibbs

        # Add the 'iSFE_model' value in the destination database
        db_ext.update({'thermodynamic_param': thermodynamic_para_update}, doc_ids=[item.doc_id])


# ============================
# Data analysis
# ============================
def kmeans_clustering_comp(_df, n_clusters=None, list_of_elements=None, scale=False):
    if list_of_elements is None:
        list_of_elements = ELEMENT_LIST

    x_data = _df[list_of_elements]
    if scale:
        scaler = StandardScaler()
        x_data = pd.DataFrame(scaler.fit_transform(x_data), columns=x_data.columns)

    if n_clusters is None:
        inertias = []
        ks = range(1, 21)

        for k in ks:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(x_data)
            inertias.append(kmeans.inertia_)

        n_clusters = int(input("optimal number of clusters:\n"))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    labels = kmeans.fit_predict(x_data)

    # order labels based on number of entries
    label_counts = pd.Series(labels).value_counts().sort_values(ascending=False)
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(label_counts.index)}
    ordered_labels = pd.Series(labels).map(label_mapping)

    df_out = _df.copy()
    df_out['cluster'] = ordered_labels.to_numpy()

    return df_out


def distance_from_knn(new_x, x_data, k=5, metric='euclidean', pvalue=10, scale=False, print_neighbours=False,
                      threshold=None, neighbours=None, return_neighbours=False):
    if scale:
        scaler = StandardScaler()
        x_data = pd.DataFrame(scaler.fit_transform(x_data), columns=x_data.columns)
        new_x = pd.DataFrame(scaler.transform(new_x), columns=x_data.columns)

    if neighbours is None:
        neighbours = NearestNeighbors(n_neighbors=k + 1, metric=metric)
        neighbours.fit(x_data)

    if threshold is None:
        dists, indices = neighbours.kneighbors(x_data)
        mean_knn_distances = dists[:, 1:].mean(axis=1)

        threshold = np.percentile(mean_knn_distances, 100 - pvalue)
    else:
        indices = None

    dist_to_new, indices_to_new = neighbours.kneighbors(new_x, n_neighbors=k)
    nearest_neighbors = [x_data.iloc[indices] for indices in indices_to_new]
    if print_neighbours:
        for i, neighbors in enumerate(nearest_neighbors):
            print(f"\nNearest neighbors for sample {i}:")
            print(neighbors)

    if return_neighbours:
        return dist_to_new.mean(axis=1), threshold, neighbours, indices, nearest_neighbors
    else:
        return dist_to_new.mean(axis=1), threshold, neighbours, indices


def knn_visualization(x_data, indices, scale=False, conditions=None, to_drop=None,
                      labels=None, cmap_name='viridis'):
    if conditions is None:
        node_colors = 'steelblue'
    else:
        cmap = cm.get_cmap(cmap_name)
        colors = [cmap(i / (len(conditions) - 1)) for i in range(len(conditions))]

        ops = {
            'gt': operator.gt,
            'lt': operator.lt,
            'ge': operator.ge,
            'le': operator.le,
            'eq': operator.eq,
            'ne': operator.ne
        }

        x_data["node_colors"] = 'steelblue'

        for cond_set, color in zip(conditions, colors):
            mask = pd.Series(True, index=x_data.index)
            for col, op_str, val in cond_set:
                mask &= ops[op_str](x_data[col], val)
            x_data.loc[mask, "node_colors"] = to_hex(color)

        node_colors = x_data["node_colors"]
        x_data = x_data.drop("node_colors", axis=1)

        node_colors = [to_rgb(color) for color in node_colors]

    if to_drop is not None:
        x_data = x_data.drop(to_drop, axis=1)

    if scale:
        scaler = StandardScaler()
        x_data = pd.DataFrame(scaler.fit_transform(x_data), columns=x_data.columns)

    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_data)

    g = nx.Graph()
    for i, point in enumerate(x_pca):
        g.add_node(i, pos=point)

    for i in range(len(x_pca)):
        if i not in indices[i, 1:]:
            for j in indices[i, 1:]:
                dist = np.linalg.norm(x_data.iloc[i].values - x_data.iloc[j].values)
                g.add_edge(i, j, weight=dist)

    spring_pos = nx.spring_layout(g, weight='weight', seed=42)
    plt.figure(figsize=(10, 10))
    nx.draw(g, spring_pos, node_size=50, alpha=0.7, with_labels=False,
            edge_color='gray', width=0.5, node_color=node_colors)

    # Add text labels
    if labels is not None:
        for i, pos in spring_pos.items():
            s = ''
            for label in labels:
                val = np.round(100 * x_data.iloc[i][label])
                if val > 0:
                    s += label + str(val)
            plt.text(pos[0], pos[1], s, fontsize=8, ha='center', va='center')

    plt.title("k-NN Graph with Spring Layout")
    plt.axis('off')
    plt.gca().set_aspect('equal')
    plt.show()
