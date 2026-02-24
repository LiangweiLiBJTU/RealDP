import time
import pandas as pd
import lightgbm as lgb
import numpy as np


def get_random_importance(data, params, task, n_runs=5, seed=0):
    null_imp_df = pd.DataFrame()

    start = time.time()
    dsp = ""
    actual_imp_df = get_feature_importances(data, params, task, shuffle=False)
    for i in range(n_runs):
        # Get current run importances
        imp_df = get_feature_importances(data, params, task, shuffle=True)
        imp_df["run"] = i + 1
        # Concat the latest importances with the old ones
        null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)
        # Erase previous message
        for l in range(len(dsp)):
            print("\b", end="", flush=True)
        # Display current run and time used
        spent = (time.time() - start) / 60
        dsp = "Done with %4d of %4d (Spent %5.1f min)" % (i + 1, n_runs, spent)
        print(dsp, end="", flush=True)

    feature_scores = []
    for _f in actual_imp_df["feature"].unique():
        f_null_imps_gain = null_imp_df.loc[
            null_imp_df["feature"] == _f, "importance_gain"
        ].values
        f_act_imps_gain = actual_imp_df.loc[
            actual_imp_df["feature"] == _f, "importance_gain"
        ].mean()
        gain_score = np.log(
            1e-10 + f_act_imps_gain / (1 + np.percentile(f_null_imps_gain, 75))
        )  # Avoid didvide by zero
        f_null_imps_split = null_imp_df.loc[
            null_imp_df["feature"] == _f, "importance_split"
        ].values
        f_act_imps_split = actual_imp_df.loc[
            actual_imp_df["feature"] == _f, "importance_split"
        ].mean()
        split_score = np.log(
            1e-10 + f_act_imps_split / (1 + np.percentile(f_null_imps_split, 75))
        )  # Avoid didvide by zero
        feature_scores.append((_f, split_score, gain_score))

    scores_df = pd.DataFrame(
        feature_scores, columns=["feature", "split_score", "gain_score"]
    )
    return scores_df[["feature", "split_score"]].to_dict(orient="records")


def get_feature_importances(data, params, task, shuffle=False):
    """
    Get feature importances for a DataFrame
    :param data: pd.DataFrame
    :param params: dict
    :param task: str
    :param shuffle: bool
    :return: pd.DataFrame
    """
    train_x, train_y, val_x, val_y = data
    if shuffle:
        train_y_final = (
            train_y.copy().sample(frac=1, random_state=0).reset_index(drop=True)
        )
    else:
        train_y_final = train_y.copy()
    # Get the feature importances
    if task == "classification":
        gbm = lgb.LGBMClassifier(**params)
    else:
        gbm = lgb.LGBMRegressor(**params)

    gbm.fit(
        train_x,
        train_y_final.values.ravel(),
        eval_set=[(val_x, val_y.values.ravel())],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    imp_df = pd.DataFrame(
        {
            "feature": train_x.columns,
            "importance_gain": gbm.booster_.feature_importance(importance_type="gain"),
            "importance_split": gbm.booster_.feature_importance(
                importance_type="split"
            ),
        }
    )
    return imp_df


# from openfe import OpenFE, tree_to_formula, transform
# # 设置task和metric
# task = 'classification'
# metric = 'multi_logloss'

# for col in target_cols:
#     print(f'Processing {col}')
#     x_df = df.drop(target_cols, axis=1)
#     y_df = df[col]
#     train_idx, val_idx = pd.Series(x_df.index), pd.Series(x_df.index)
#     # 训练openfe模型
#     ofe = OpenFE()

#     params = {"num_iterations": 1000, "seed": 42}
#     params.update(
#         {
#             'objective':'l2',
#             "colsample_bytree": 0.8,
#             "colsample_bynode": 0.8,
#             "learning_rate": 0.05,
#             "num_leaves":31,
#         }
#     )
#     ofe.fit(data=x_df, task=task, train_index=train_idx, val_index=val_idx,
#             metric=metric, label=y_df, seed=42,stage2_metric='permutation',
#             stage2_params=params)
