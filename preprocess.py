import os
import sys
sys.path.append('./')
import numpy as np
import pandas as pd
import json
import joblib
import category_encoders
from sklearn.preprocessing import OrdinalEncoder

def data_loader(use_cols):
    cols = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex',
            'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
    drop_cols = ['PassengerId', 'Name', 'Survived']
    target_col = ['Survived']
    cat_cols = ['Sex', 'Ticket', 'Cabin', 'Embarked']
    cont_cols = list(set(cols) - set(drop_cols) -　set(cat_cols))

    # data import
    raw = pd.read_csv('train.csv')
    # generate target vector
    target = raw[target_col].values
    target = target.reshape(target.shape[0],)

    # generate feature tensor
    dat = raw.drop(columns=drop_cols)
    # NaN handling
    cat_nan_filler = dat[cat_cols].fillna('Na')
    dat[cat_cols] = cat_nan_filler
    cont_nan_filler = dat[cont_cols].fillna(dat[cont_cols].median())
    dat[cont_cols] = cont_nan_filler
    oe = OrdinalEncoder()
    oe = oe.fit(dat[cat_cols])
    dat[cat_cols] = pd.DataFrame(oe.transform(dat[cat_cols]), columns=cat_cols)
    dat = dat[use_cols]
    dat = dat.values

    return dat, target


if __name__ == "__main__":
    # データを共有できるように保存
    x, y = data_loader()
    data = [x, y]
    filenames = ['train_feature.pkl', 'train_target.pkl']
    for dat, filename in zip(data, filenames):
        with open(filename, 'wb') as f:
            joblib.dump(dat, f)
