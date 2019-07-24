# Created by Qingzhi Ma at 2019-07-24
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk
import pandas as pd


def convert_df_to_yx(df,x, y):
    return df[y].values, df[x].values.reshape(-1,1)

def get_group_count_from_df(df, group_attr,convert_to_str=True):
    if convert_to_str:
        df[group_attr] =df[group_attr].astype(str)
    grouped = df.groupby(by=group_attr)
    counts={}
    for name,group in grouped:
        counts[name] = group.shape[0]

    return counts

def get_group_count_from_file(file, group_attr,sep=','):
    df = pd.read_csv(file,sep=sep, dtype={group_attr: object}, keep_default_na=False)
    return get_group_count_from_df(df,group_attr,convert_to_str=False)
