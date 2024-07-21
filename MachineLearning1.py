import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from matplotlib import ticker
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import *
from tensorflow.keras.layers import Input, Dense, Dropout
import matplotlib.pyplot as plt

df = pd.read_csv('https://raw.githubusercontent.com/luisgdelafuente/datasets/main/AmesHousing.csv')

print(df.head())


VM= pd.DataFrame({'colonne': df.columns.values,
                  'nbr de vm':df.isna().sum().values,
                  '% de vm':df.isna().sum().values/len(df),})
VM = VM[VM['nbr de vm']>0]
print(VM.sort_values(by='nbr de vm',ascending=False).reset_index(drop=True))

df.drop(['Pool QC', 'Misc Feature', 'Alley', 'Fence', 'Fireplace Qu'],axis = 1, inplace = True)

#on affiche les variables catégorielles et les variables numériques

colonnes_avec_VM = df.columns[df.isna().sum() > 0]

for col in colonnes_avec_VM :
  print(col)
  print(df[col].unique()[:5])
  print('*'*30)

#on remplace les VM numeriques par la moyenne de la colonne

num_VM = ['Lot Frontage', 'Mas Vnr Area', 'BsmtFin SF 1', 'BsmtFin SF 2',

'Bsmt Unf SF', 'Total Bsmt SF', 'Bsmt Full Bath',

'Bsmt Half Bath', 'Garage Yr Blt', 'Garage Cars', 'Garage Area']

for n_col in num_VM:
  df[n_col] = df[n_col].fillna(df[n_col].mean())

# on replace le VM nominales par le mode variable

nom_VM = [x for x in colonnes_avec_VM if x not in num_VM]
for nom_col in nom_VM:
  df[nom_col]=df[nom_col].fillna(df[nom_col].mode().to_numpy()[0])

#encodage des variables catégorielles
#on affiche tout d'abord les types de donnees pour chaque colonne

types = pd.DataFrame({

'Colonne': df.select_dtypes(exclude='object'). columns.values,

'Type': df.select_dtypes(exclude='object').dtypes.values})

print(types)

#encodage des variables catégorielles,

#selon le descriptif de la dataset la colonne MS SubClass est nominale,

#Pandas a du mal à retourner son vrai type

df['MS SubClass'] = df['MS SubClass']. astype(str)

selection_de_variables = ['MS SubClass', 'MS Zoning', 'Lot Frontage', 'Lot Area',

'Neighborhood', 'Overall Qual', 'Overall Cond',

'Year Built', 'Total Bsmt SF', '1st Flr SF', '2nd Flr SF',

'Gr Liv Area', 'Full Bath', 'Half Bath', 'Bedroom AbvGr',

'Kitchen AbvGr', 'TotRms AbvGrd', 'Garage Area',

'Pool Area', 'SalePrice']

df = df[selection_de_variables]

#la dataset comprendera maintenant que 67 variables

df = pd.get_dummies(df)

#fractionner dataset en des données de test et de train

train = df.sample(frac = 0.8, random_state = 9)

test = df.drop(train.index)

#variable cible
train_cible = train.pop('SalePrice')
test_cible = test.pop('SalePrice')


#Standardisation
variables_pred = train.columns

for col in variables_pred:

    col_moyenne = train[col].mean()
    col_ecart_type = train[col].std()

    if col_ecart_type == 0:

        col_ecart_type = 1e-20

    train[col] = (train[col] - col_moyenne) / col_ecart_type

    test[col] = (test[col] - col_moyenne) / col_ecart_type



model = keras.Sequential([
    Input(shape=(train.shape[1],)),
    Dense(64, activation='relu'),
    Dropout(0.3, seed=2),
    Dense(64, activation='swish'),
    Dense(64, activation='relu'),
    Dense(64, activation='swish'),
    Dense(64, activation='relu'),
    Dense(64, activation='swish'),
    Dense(1)
])
optimiseur = tf.keras.optimizers.RMSprop(learning_rate=0.001)

model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=optimiseur,
    metrics=['mae']
)

training = model.fit(train, train_cible, epochs=110, validation_split=0.2)

historique = pd.DataFrame(training.history)
historique['epoque'] = training.epoch

figure, axe = plt.subplots(figsize=(14, 8))

num_epoque = historique.shape[0]

axe.plot(np.arange(0, num_epoque), historique["mae"], 
         label="Training MAE", lw=3, color='red')

axe.plot(np.arange(0, num_epoque), historique["val_mae"],
         label="Validation MAE", lw=3, color='blue')

axe.legend()

plt.tight_layout()

plt.show()

test1 = test.iloc[[50]]
test_prediction = model.predict(test1).squeeze()
test_label = test_cible.iloc[50]
print("Prediction du modele : {:.2f} ".format(test_prediction))
print("Valeur actuelle : {:.2f} ".format(test_label))

score = model.evaluate(test,test_cible,verbose=0)
print('Score final : ', score[1])

