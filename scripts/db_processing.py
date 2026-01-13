"""
db_core_analysis.py

Builds the dev and ext databases from the raw datasets
"""

# ============================
# Imports
# ============================
from time import perf_counter
from src.db_core_analysis import ELEMENT_LIST, MATMINER_FEATURES
from src.db_core_analysis import ModelGPR, ModelRFR, ModelHGBR, ModelADAR, ModelKRR, ModelMLPR, ModelSVM
from src.db_core_analysis import copy_database_if_not_exists, create_df_from_feature_set
from src.db_core_analysis import create_sfe_df, perform_cross_validation
from src.db_core_analysis import update_composition, update_lattice_parameter, update_matminer_features
from src.db_core_analysis import update_mech_param, update_defmech
from src.db_core_analysis import update_gibbs_tcpython, update_interface_energy
from src.db_core_analysis import extrapol_sfe_models, extrapol_sfe_gibbs


# ============================
# Compute models and extrapolate SFE values
# ============================
def model_cv(model_list=None, is_featurized=False, plot_splits=True):
    _t0 = perf_counter()
    if model_list is None:
        model_list = [
            ModelGPR(),
            ModelRFR(),
            ModelHGBR(),
            ModelADAR(),
            ModelKRR(),
            ModelMLPR()
        ]

    _df = create_sfe_df(sfe_upper_bound=100, is_featurized=is_featurized)
    for _model in model_list:
        if is_featurized:
            perform_cross_validation(_df.copy(), 'iSFE', _model, model_description='featurized',
                                                 plot_splits=plot_splits)
        else:
            perform_cross_validation(_df.copy(), 'iSFE', _model,
                                                 plot_splits=plot_splits)

    _t1 = perf_counter()
    print('Elapsed time for hyperparameter CV: ', _t1 - _t0)


def main(
        update=False,
        thermo_modeling=False,
        extrapol_sfe_cv=False,
        extrapol_sfe_cv_featurized=False,
        extrapol_sfe_to_db=False,
        extrapol_sfe_to_db_featurized=False,
        extrapol_sfe_tc=False,
):

    if update:
        t0 = perf_counter()
        '''updates to myDatabase_dev.json'''
        update_composition()
        update_mech_param()
        update_defmech()
        if thermo_modeling:
            update_gibbs_tcpython(input_gibbs='data/df_equilibria_with_gibbs1.pkl')
            update_gibbs_tcpython(input_gibbs='data/df_equilibria_with_gibbs2.pkl')
            update_gibbs_tcpython(input_gibbs='data/df_equilibria_with_gibbs3.pkl')
            update_interface_energy(input_interface='data/df_interfaceE20250729_300.pkl')
            update_lattice_parameter()
        t1 = perf_counter()
        print('Elapsed time for updating database: ', t1 - t0)

        t0 = perf_counter()
        update_matminer_features()
        t1 = perf_counter()
        print('Elapsed time for featurizing database: ', t1 - t0)

    if extrapol_sfe_cv:
        '''extrapolates to myDatabase_ext.json'''
        copy_database_if_not_exists("data/data_dev.json", "data/data_ext.json")

        '''Model hyperparameter cross validation'''
        model_cv()
        if extrapol_sfe_cv_featurized:
            model_cv(is_featurized=True)

    '''Extrapolate SFE values to db_ext'''
    if extrapol_sfe_to_db | extrapol_sfe_to_db_featurized:
        t0 = perf_counter()

        model_str_list = []
        isfe_str_list = []
        df_list = []
        features = [
            'doc_id',
            'composition_at.Fe',
            'composition_at.Mn',
            'composition_at.Cr',
            'composition_at.Ni',
            'composition_at.Co',
            'composition_at.C',
            'composition_at.N',
            'composition_at.V',
            'composition_at.Ti',
            'composition_at.Mo',
            'composition_at.Cu',
            'composition_at.Al',
            'composition_at.Si',
            'composition_at.Zn',
            'composition_at.Pd'
        ]
        conditions = [{'feature': 'composition_at', 'op': 'exists'}]
        if extrapol_sfe_to_db:
            model_str_list = [
                *model_str_list,
                'model_iSFE_RFR_',
                'model_iSFE_GPR_',
                'model_iSFE_HGBR_',
                'model_iSFE_ADAR_',
                'model_iSFE_KRR_',
                'model_iSFE_MLPR_'
            ]

            df = create_df_from_feature_set({'features': features, 'conditions': conditions})
            df['itemID'] = df['doc_id']
            df.pop('doc_id')
            df_list = [*df_list, df.copy(), df.copy(), df.copy(), df.copy(), df.copy(), df.copy()]

            isfe_str_list = [
                *isfe_str_list,
                'iSFE_RFR',
                'iSFE_GPR',
                'iSFE_HGBR',
                'iSFE_ADAR',
                'iSFE_KRR',
                'iSFE_MLPR'
            ]

        if extrapol_sfe_to_db_featurized:
            model_str_list = [
                *model_str_list,
                'model_iSFE_RFR_featurized',
                'model_iSFE_GPR_featurized',
                'model_iSFE_HGBR_featurized',
                'model_iSFE_ADAR_featurized',
                'model_iSFE_KRR_featurized',
                'model_iSFE_MLPR_featurized'
            ]

            df_ftrzd = create_df_from_feature_set(
                {'features': [*features, *MATMINER_FEATURES], 'conditions': conditions})
            df_ftrzd['itemID'] = df_ftrzd['doc_id']
            df_ftrzd.pop('doc_id')
            df_list = [*df_list,
                       df_ftrzd.copy(), df_ftrzd.copy(), df_ftrzd.copy(),
                       df_ftrzd.copy(), df_ftrzd.copy(), df_ftrzd.copy()]

            isfe_str_list = [
                *isfe_str_list,
                'iSFE_RFR_featurized',
                'iSFE_GPR_featurized',
                'iSFE_HGBR_featurized',
                'iSFE_ADAR_featurized',
                'iSFE_KRR_featurized',
                'iSFE_MLPR_featurized'
            ]

        extrapol_sfe_models(model_str_list, df_list, isfe_str_list)
        t1 = perf_counter()
        print('Elapsed time for extrapoling SFE: ', t1 - t0)

    if extrapol_sfe_tc:
        t0 = perf_counter()
        extrapol_sfe_gibbs()
        t1 = perf_counter()
        print('Elapsed time for extrapoling TC SFE: ', t1 - t0)


if __name__ == "__main__":
    main()
