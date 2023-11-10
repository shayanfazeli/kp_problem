# - preparing the master dataframe
from typing import List, Tuple
import os
from tqdm import tqdm
import random
import numpy
import pandas
from tabulate import tabulate
from kp_problem.utilities.dataframe_utils import reward_function, append_previous_month_info


def fetch_dataset(
        dataset_url: str = 'https://figshare.com/ndownloader/files/35249488',
        ewma_histories: bool = False,
        ewma_alpha: float = 0.9,
        ewma_adjust: bool = False,
        prev_month: bool = False,
        balance_by_n_rows_per_treatment: int = 0
) -> pandas.DataFrame:
    # - getting the dataset from the web and caching it
    cache_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../resources/raw_data/dataset.pkl'))
    if os.path.isfile(cache_path):
        df = pandas.read_pickle(cache_path)
    else:
        df = pandas.read_csv(dataset_url).drop(columns=['Unnamed: 0'])
        df.to_pickle(cache_path)

    # - adding the reward
    df['reward'] = reward_function(df.VL.to_numpy(), df.CD4.to_numpy())

    # - building this metadata just for our use.
    feature_groups = dict(
        static=['Gender', 'Ethnic'],
        categorical=['Comp. NNRTI', 'Comp. INI', 'Base Drug Combo', 'Drug (M)', 'Extra PI','Extra pk-En',],
        numerical=['VL','CD4','Rel CD4', 'reward'],
        target=['VL', 'CD4', 'Rel CD4', 'VL (M)', 'CD4 (M)', 'reward', 'reward_advantage']
    )

    # - one-hot conversion
    df = pandas.get_dummies(df, columns=feature_groups['categorical']+feature_groups['static'], dtype=float)
    df.sort_values(['PatientID', 'Timepoints'], inplace=True)

    # - computing ewma features (up until and including t)
    if ewma_histories:
        chosen_features = [e for e in df.columns.tolist() if e not in ['PatientID', 'Timepoints'] and 'Gender' not in e and 'Ethnic' not in e]# and 'Comp. NNRTI' not in e]
        for f in tqdm(chosen_features):
            df[f'{f}_ewma'] = df.groupby(['PatientID'])[f].ewm(alpha=ewma_alpha, adjust=ewma_adjust).mean().reset_index(level=0, drop=True)

    # - computing the reward advantage
    tmp = df.copy()
    tmp = tmp.groupby('PatientID').apply(append_previous_month_info).reset_index(level=0, drop=True)
    tmp['reward_advantage'] = tmp['reward'] - tmp['prev_month_reward']
    tmp['Timepoints'] = tmp['Timepoints'] / 60.
    
    # - whether or not the prev_month info is to be retained or dropped
    if not prev_month:
        tmp.drop(columns=[e for e in tmp.columns if 'prev_month_' in e], inplace=True)


    # - balancing treatment
    if balance_by_n_rows_per_treatment:
        def find_treatment(x):
            for i in range(4):
                if x[f'Comp. NNRTI_{i}.0']:
                    return i
        tmp['treatment'] = tmp.apply(find_treatment, axis=1)
        tmp = tmp.groupby(['treatment']).sample(balance_by_n_rows_per_treatment, replace=True).reset_index(level=0, drop=True).drop(columns=['treatment'])

    # - getting the features and targets
    df_features = tmp.loc[:, [e for e in tmp.columns if e not in feature_groups['target'] +  [f'{k}_ewma' for k in feature_groups['target']]]].copy() # [f'prev_month_{k}' for k in feature_groups['target']] +
    df_targets = tmp.loc[:, feature_groups['target'] + ['PatientID', 'Timepoints']].copy()

    # - returning the outputs
    return df_features, df_targets, feature_groups
