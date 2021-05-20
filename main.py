import pandas as pd
import matplotlib

train_file = 'train.csv'
test_file = 'test.csv'
files_directory = '/home/michael/Documentos/Seminario_de_IO_1/Competencia_Kaggle/'


def load_files():
    train = pd.read_csv(files_directory + train_file)
    test = pd.read_csv(files_directory + test_file)
    return train, test


def check_uniques(train):
    uniques_dict = {}
    for col in train:
        uniques_dict[col] = train[col].unique()
    return uniques_dict


def fix_binaries(bin_train, bin_test):
    dfs = [bin_train, bin_test]
    for df in dfs:
        df.loc[df.bin_3 == 'T', 'bin_3'] = 1
        df.loc[df.bin_3 == 'F', 'bin_3'] = 0
        df.loc[df.bin_4 == 'Y', 'bin_4'] = 1
        df.loc[df.bin_4 == 'N', 'bin_4'] = 0
    return bin_train, bin_test


def fix_nom_one_hot(nom_train, nom_test):
    categorical_cols = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']
    dfs = [nom_train, nom_test]
    fixed_dfs = []
    for df in dfs:
        for col in categorical_cols:
            one_hot = pd.get_dummies(df[col])
            df = df.drop(col, axis=1)
            df = df.join(one_hot)
        fixed_dfs.append(df)
    return fixed_dfs[0], fixed_dfs[1]


def fix_ordinal(ord_train, ord_test):
    dfs = [ord_train, ord_test]
    for df in dfs:
        df.loc[df.ord_1 == 'Novice', 'ord_1'] = 1
        df.loc[df.ord_1 == 'Contributor', 'ord_1'] = 2
        df.loc[df.ord_1 == 'Expert', 'ord_1'] = 3
        df.loc[df.ord_1 == 'Master', 'ord_1'] = 4
        df.loc[df.ord_1 == 'Grandmaster', 'ord_1'] = 5
        df.loc[df.ord_2 == 'Freezing', 'ord_2'] = 1
        df.loc[df.ord_2 == 'Cold', 'ord_2'] = 2
        df.loc[df.ord_2 == 'Warm', 'ord_2'] = 3
        df.loc[df.ord_2 == 'Hot', 'ord_2'] = 4
        df.loc[df.ord_2 == 'Boiling Hot', 'ord_2'] = 5
        df.loc[df.ord_2 == 'Lava Hot', 'ord_2'] = 6
    return ord_train, ord_test


def fix_nom_hex(nom_train, nom_test):
    files = [nom_train, nom_test]
    columns = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']
    for file in files:
        for column in columns:
            file[column] = file[column].apply(lambda x: int(x, 16))
            file[column] = (file[column]-file[column].min())/(file[column].max()-file[column].min())
    return nom_train, nom_test


def fix_ord_letters(ord_train, ord_test):

    return ord_train, ord_test


train_df, test_df = load_files()
bin_fixed_train, bin_fixed_test = fix_binaries(train_df, test_df)
nom_1_fixed_train, nom_1_fixed_test = fix_nom_one_hot(bin_fixed_train, bin_fixed_test)
ord_1_fixed_train, ord_1_fixed_test = fix_ordinal(nom_1_fixed_train, nom_1_fixed_test)
nom_fixed_train, nom_fixed_test = fix_nom_hex(ord_1_fixed_train, ord_1_fixed_test)

nom_fixed_train['ord_3'].value_counts().plot(kind='bar')

ord_fixed_train, ord_fixed_test = fix_ord_letters(nom_fixed_train, nom_fixed_test)
unique_values = check_uniques(bin_fixed_train)

print(unique_values['ord_2'])
