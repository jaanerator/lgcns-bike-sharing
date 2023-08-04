import numpy as np
import pandas as pd
from category_encoders import OneHotEncoder, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

CAT_FEATURES = ["season", "holiday", "workingday", "weather"]


def convert_dtypes(df, col: str):
    df[col] = df[col].astype(object)
    return df


def plus1(df, col: str):
    df[col] = df[col] + 1
    return df


# TODO: 전처리 파이프라인 작성
# 1. 범주형 변수(CAT_FEATURES)는 타겟 인코딩 적용 (from category_encoders import TargetEncoder)
preprocess_pipeline = ColumnTransformer(
    transformers=[
        (
            "convert_string1",
            FunctionTransformer(plus1, kw_args={"col": "season"}),
            ["season"],
        ),
        # ('convert_string2', FunctionTransformer(convert_dtypes, kw_args={'col':"holiday"}), ["holiday"]),
        # ('convert_string3', FunctionTransformer(convert_dtypes, kw_args={'col':"workingday"}), ["workingday"]),
        # ('convert_string4', FunctionTransformer(convert_dtypes, kw_args={'col':"weather"}), ["weather"]),
        # ("target_encoder", TargetEncoder(), CAT_FEATURES),
    ],
    remainder="passthrough",
    verbose_feature_names_out=False,
)
preprocess_pipeline.set_output(transform="pandas")
