import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def cat_summary(df, col_name, plot=False):
    print(pd.DataFrame({col_name: df[col_name].value_counts(),
                        "Ratio": 100 * df[col_name].value_counts() / len(df)}))
    print("##########################################")
    if plot:
        sns.countplot(x=df[col_name], data=df)
        plt.show(block=True)

def num_summary(df, numerical_col, plot=False):
    quantiles = [0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.99]
    print(df[numerical_col].describe(quantiles).T)
    if plot:
        df[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

def target_summary_with_num(df, target, numerical_col):
    print(df.groupby(target).agg({numerical_col:"mean"}), end="\n\n\n")

def target_summary_with_cat(df, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": df.groupby(categorical_col)[target].mean()}), end="\n\n\n")

def correlation_matrix(df, cols):
    plt.figure(figsize=(10,8))
    sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size':12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)
