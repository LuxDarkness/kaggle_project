import keras
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
import tensorflow as tf
import numpy as np
import math
from tensorflow.keras import layers, regularizers

train_file = 'train.csv'
test_file = 'test.csv'
files_directory = '/home/michael/Documentos/Seminario_de_IO_1/Competencia_Kaggle/'


def load_file(file):
    return pd.read_csv(files_directory + file)


def check_uniques(train):
    uniques_dict = {}
    for col in train:
        uniques_dict[col] = train[col].unique()
    return uniques_dict


def fix_binaries(bin_file):
    bin_file.loc[bin_file.bin_3 == 'T', 'bin_3'] = 1
    bin_file.loc[bin_file.bin_3 == 'F', 'bin_3'] = 0
    bin_file.loc[bin_file.bin_4 == 'Y', 'bin_4'] = 1
    bin_file.loc[bin_file.bin_4 == 'N', 'bin_4'] = 0
    return bin_file


def fix_nom_one_hot(nom_file):
    categorical_cols = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']
    for col in categorical_cols:
        one_hot = pd.get_dummies(nom_file[col])
        nom_file = nom_file.drop(col, axis=1)
        nom_file = nom_file.join(one_hot)
    return nom_file


def fix_ordinal(ord_file):
    ord_file.loc[ord_file.ord_1 == 'Novice', 'ord_1'] = 1
    ord_file.loc[ord_file.ord_1 == 'Contributor', 'ord_1'] = 2
    ord_file.loc[ord_file.ord_1 == 'Expert', 'ord_1'] = 3
    ord_file.loc[ord_file.ord_1 == 'Master', 'ord_1'] = 4
    ord_file.loc[ord_file.ord_1 == 'Grandmaster', 'ord_1'] = 5
    ord_file.loc[ord_file.ord_2 == 'Freezing', 'ord_2'] = 1
    ord_file.loc[ord_file.ord_2 == 'Cold', 'ord_2'] = 2
    ord_file.loc[ord_file.ord_2 == 'Warm', 'ord_2'] = 3
    ord_file.loc[ord_file.ord_2 == 'Hot', 'ord_2'] = 4
    ord_file.loc[ord_file.ord_2 == 'Boiling Hot', 'ord_2'] = 5
    ord_file.loc[ord_file.ord_2 == 'Lava Hot', 'ord_2'] = 6
    return ord_file


def fix_nom_hex(nom_file):
    columns = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']
    for column in columns:
        nom_file[column] = nom_file[column].apply(lambda x: int(x, 16))
        nom_file[column] = (nom_file[column]-nom_file[column].min())/(nom_file[column].max()-nom_file[column].min())
    return nom_file


def fix_ord_letters(ord_file):
    columns = ['ord_3', 'ord_4', 'ord_5']
    ord_file['ord_5'] = ord_file['ord_5'].astype(str).str[0]
    ord_enc = OrdinalEncoder()
    for column in columns:
        ord_file[column] = ord_enc.fit_transform(ord_file[[column]])
    return ord_file


def fix_month(ord_file):
    ord_file['month'] = pd.to_datetime(ord_file['month'].values, format='%m').astype('period[Q]')
    ord_file['month'] = ord_file['month'].astype(str).str[5]
    return ord_file


def order_df(df):
    cols = list(df.columns.values)
    try:
        cols.pop(cols.index('target'))
    except ValueError:
        return df
    df = df[cols + ['target']]
    return df


def preprocess_file(file):
    processed = fix_binaries(file)
    processed = fix_nom_one_hot(processed)
    processed = fix_ordinal(processed)
    processed = fix_nom_hex(processed)
    processed = fix_ord_letters(processed)
    processed = fix_month(processed)
    processed = order_df(processed)
    return processed


def write_file(df, name):
    df.to_csv(files_directory+name)


def turn_to_np_array(df):
    target = df.pop('target')
    df.pop('id')
    df.pop('bin_0')
    np_array = np.asarray(df.values).astype('float32')
    np_target = np.asarray(target.values).astype('float32')
    return np_array, np_target


def lr_decay(epoch):
    return 0.01 * math.pow(0.666, epoch)


train_df = load_file(train_file)
# real_info_df = load_file(test_file)
processed_train = preprocess_file(train_df)
# processed_real = preprocess_file(real_info_df)

sk_training_data, sk_training_target = turn_to_np_array(processed_train)

# training_df, testing_df = train_test_split(processed_train, test_size=0.3, train_size=0.7, random_state=2)
# training_np, training_target = turn_to_np_array(training_df)
# testing_np, testing_target = turn_to_np_array(testing_df)

# write_file(processed_train, 'train_processed.csv')

# regularizers.L1L2(
#     l1=0.2, l2=0.2
# )
#
# model = keras.Sequential([
#     layers.Dense(42, activation='sigmoid', input_shape=(42, 1)),
#     layers.Dense(20, activation='relu'),
#     layers.Dense(10, activation='relu'),
#     layers.Dense(1, activation='sigmoid', kernel_regularizer='l1_l2')
# ])
#
# model.compile(
#     optimizer='Adam',
#     loss=tf.keras.losses.BinaryCrossentropy(),
#     metrics=['accuracy']
# )

# lr_decay_callback = tf.keras.callbacks.LearningRateScheduler(lr_decay, verbose=True)
#
# model.fit(x=training_np, y=training_target, epochs=20, batch_size=1000, validation_data=(testing_np, testing_target),
#           callbacks=[lr_decay_callback])

lr_model = LogisticRegressionCV(solver='saga', max_iter=1000, penalty='elasticnet', l1_ratios=[0.3, 0.4], n_jobs=-1)
lr_model.fit(sk_training_data, sk_training_target)
print('Process complete!')
