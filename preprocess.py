from typing import Literal, List
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import pandas as pd


@dataclass
class HomeData:
    file_origin: str = "./data/health_origin.csv"
    file_trn: str = "./data/train.csv"
    file_tst: str = "./data/test_eda.csv"
    target_col: str = "D"
    features: List[str] = field(default_factory=list)
    encoding_columns: List[str] = field(default_factory=list)
    # scaler: Literal['None', 'standard', 'minmax'] = 'None'
    # scale_columns: List[str] = field(default_factory=list)  # 수정: field 함수 사용

    def _read_df(self, split: Literal["train", "test"] = "train"):
        origin = pd.read_csv("./data/health_origin.csv")
        drop_na = [
            "TOT_CHOLE",
            "TRIGLYCERIDE",
            "HDL_CHOLE",
            "LDL_CHOLE",
            "BLDS",
            "OLIG_PROTE_CD",
        ]  # 콜레스트롤 + BLDS + 요단백
        import numpy as np

        # from sklearn.preprocessing import OneHotEncoder

        # null 값 drop
        origin.dropna(subset=drop_na, inplace=True)

        origin["D"] = origin["BLDS"].apply(lambda x: 1 if x >= 126 else 0)

        # origin.to_csv('./data/origin_df.csv')

        drop_cols = [
            "BP_HIGH",
            "TOT_CHOLE",
            "DATA_STD_DT",
            "HCHK_YEAR",
            "IDV_ID",
            "SIDO",
            "HEIGHT",
            "WEIGHT",
            "WAIST",
            "SIGHT_LEFT",
            "SIGHT_RIGHT",
            "HEAR_LEFT",
            "HEAR_RIGHT",
            "SGOT_AST",
            "DRK_YN",
        ]
        drop_cols.extend(
            [
                "TTR_YN",
                "WSDM_DIS_YN",
                "ODT_TRB_YN",
                "TTH_MSS_YN",
                "CRS_YN",
                "HCHK_OE_INSPEC_YN",
            ]
        )  # 치아관련

        # suffle
        origin = origin.sample(frac=1).reset_index(drop=True)

        origin_0 = origin.loc[origin["D"] == 0].head(41767)  # 언더샘플링
        origin_1 = origin.loc[origin["D"] == 1]
        origin_df = pd.concat([origin_0, origin_1])  # 데이터 합치기

        origin_df["BMI"] = origin_df["WEIGHT"] / ((origin_df["HEIGHT"]) / 100) ** 2
        origin_df.insert(0, "BLDS", origin_df.pop("BLDS"))

        # drop colunms
        origin_df.drop(drop_cols, axis=1, inplace=True)

        # suffle
        origin_df = origin_df.sample(frac=1).reset_index(drop=True)

        test_df = origin_df.tail(10000)
        val_df = origin_df.head(73534).tail(10000)
        origin_df = origin_df.head(63534)

        origin_df = origin_df.dropna()
        val_df = val_df.dropna()
        test_df = test_df.dropna()
        val_df = val_df.drop(columns=["BLDS", "D"], axis=1)

        origin_df.to_csv("./data/train.csv", index=False)
        val_df.to_csv("./data/validation.csv", index=False)
        test_df.to_csv("./data/test.csv", index=False)

        if split == "train":
            df = pd.read_csv(self.file_trn)
            target = df[self.target_col]
            # df_X = df[self.features]
            df = df.drop(self.target_col, axis=1)
            df = df.drop(columns=["BLDS"], axis=1)
            return df, target
        elif split == "test":
            df = pd.read_csv(self.file_tst)
            target = df[self.target_col]
            # df_X = df[self.features]  # df 대신 df_X를 반환
            df = df.drop(self.target_col, axis=1)
            df = df.drop(columns=["BLDS"], axis=1)
            return df, target
        raise ValueError(f'"{split}"은(는) 허용되지 않습니다.')

    def preprocess(self):
        X_trn, y_trn = self._read_df(split="train")
        X_tst, y_tst = self._read_df(split="test")

        return X_trn, y_trn, X_tst, y_tst


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(
        description="Pytorch K-fold Cross Validation", add_help=add_help
    )
    parser.add_argument(
        "-c", "--config", default="./config.py", type=str, help="configuration file"
    )

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()

    exec(open(args.config).read())
    cfg = config

    preprocess_params = cfg.get("preprocess")

    args = get_args_parser().parse_args()
    home_data = HomeData(
        features=preprocess_params.get("features"),
        file_trn=preprocess_params.get("train-csv"),
        file_tst=preprocess_params.get("test-csv"),
        target_col=preprocess_params.get("target-col"),
    )
    trn_X, trn_y, tst_X, tst_y = home_data.preprocess()

    trn_X.to_csv(preprocess_params.get("output-train-feas-csv"))
    tst_X.to_csv(preprocess_params.get("output-test-feas-csv"))
    trn_y.to_csv(preprocess_params.get("output-train-target-csv"))
    tst_y.to_csv(preprocess_params.get("output-test-target-csv"))
