COLUMN_MAPPING = {
    "laufkont": "checking_status",
    "laufzeit": "loan_duration",
    "moral": "credit_history",
    "verw": "purpose",
    "hoehe": "loan_amount",
    "sparkont": "savings_status",
    "beszeit": "employment_duration",
    "rate": "installment_rate",
    "famges": "personal_status_sex",
    "buerge": "other_debtors",
    "wohnzeit": "residence_duration",
    "verm": "property",
    "alter": "age",
    "weitkred": "other_installment_plans",
    "wohn": "housing",
    "bishkred": "existing_credits",
    "beruf": "job",
    "pers": "dependents",
    "telef": "telephone",
    "gastarb": "foreign_worker",
    "kredit": "target"
}


def rename_columns(df):
    return df.rename(columns=COLUMN_MAPPING)