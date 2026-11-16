import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Harrizon Soler
    ## Segmentación de solicitantes: perfiles interpretables con clustering
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd

    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline as SKLearnPipeline
    from imblearn.pipeline import Pipeline as IMBLearnPipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from imblearn.over_sampling import SMOTE
    from sklearn.metrics import classification_report, f1_score

    return (
        ColumnTransformer,
        IMBLearnPipeline,
        OneHotEncoder,
        RandomForestClassifier,
        SKLearnPipeline,
        SMOTE,
        SimpleImputer,
        StandardScaler,
        classification_report,
        mo,
        pd,
        train_test_split,
    )


@app.cell
def _(pd):
    df = pd.read_csv("data/loan_default.csv")
    df
    return (df,)


@app.cell
def _(df, train_test_split):
    TARGET_COL = 'Status'

    # all columns except TARGET_COL
    x = df.drop(columns=TARGET_COL)
    y = df[TARGET_COL]

    # TODO: stratify only train not test https://www.reddit.com/r/datascience/comments/idt9e4/random_vs_stratified_train_test_split/
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        stratify=y,
        test_size=0.2,
        random_state=42
    )
    return x_test, x_train, y, y_test, y_train


@app.cell
def _(y):
    imbalance = y.value_counts(normalize=True)
    print(f"Status proportion on 0: {imbalance[0]*100:.2f}% and 1: {imbalance[1]*100:.2f}%")
    return


@app.cell
def _(
    ColumnTransformer,
    OneHotEncoder,
    SKLearnPipeline,
    SimpleImputer,
    StandardScaler,
):
    numerical_columns = [
        "loan_amount", "rate_of_interest", "term", "property_value",
        "income", "Credit_Score", "LTV", "dtir1",
    ]

    categorical_columns = [
        "approv_in_adv", "loan_purpose", "Credit_Worthiness", "open_credit",
        "business_or_commercial", "Neg_ammortization", "interest_only",
        "lump_sum_payment", "occupancy_type",
    ]

    column_transformer = ColumnTransformer(
        [
            ("num", SKLearnPipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numerical_columns),
            ("cat", SKLearnPipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_columns),
        ]
    )
    return (column_transformer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Por lo que entiendo, en la actividad anterior utilizamos entrenamiento no supervisado pero SMOTE parece que es solo para entrenamientos supervisados, por lo que voy a aplicar el RandomForest para que sirva de algo.
    """)
    return


@app.cell
def _(
    IMBLearnPipeline,
    RandomForestClassifier,
    column_transformer,
    x_train,
    y_train,
):
    pipeline_without_smote = IMBLearnPipeline([
        ("preprocessor", column_transformer),
        ("classifier", RandomForestClassifier(random_state=42)),
    ])

    pipeline_without_smote.fit(x_train, y_train)
    return (pipeline_without_smote,)


@app.cell
def _(
    IMBLearnPipeline,
    RandomForestClassifier,
    SMOTE,
    column_transformer,
    x_train,
    y_train,
):
    pipeline_with_smote = IMBLearnPipeline([
        ("preprocessor", column_transformer),
        ("smote", SMOTE(random_state=42)),
        ("classifier", RandomForestClassifier(random_state=42)),
    ])

    pipeline_with_smote.fit(x_train, y_train)
    return (pipeline_with_smote,)


@app.cell
def _(classification_report, pipeline_without_smote, x_test, y_test):
    y_pred_without_smote = pipeline_without_smote.predict(x_test)
    print(classification_report(y_test, y_pred_without_smote))
    return


@app.cell
def _(classification_report, pipeline_with_smote, x_test, y_test):
    y_pred_with_smote = pipeline_with_smote.predict(x_test)
    print(classification_report(y_test, y_pred_with_smote))
    return


if __name__ == "__main__":
    app.run()
