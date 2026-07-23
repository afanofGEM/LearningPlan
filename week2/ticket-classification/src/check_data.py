import pandas as pd


def main():
    df = pd.read_csv("../data/工单数据集.csv")

    print("前 5 条数据：")
    print(df.head())

    print("\n数据形状：")
    print(df.shape)

    print("\n各类别数量：")
    print(df["label"].value_counts())

    print("\n是否存在缺失值：")
    print(df.isnull().sum())

    print("\n是否存在重复数据：")
    print(df.duplicated().sum())


if __name__ == "__main__":
    main()