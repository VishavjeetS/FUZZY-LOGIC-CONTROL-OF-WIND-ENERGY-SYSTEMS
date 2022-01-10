
import pandas as pd
import numpy as np
df=pd.read_csv("dataset.csv")
df.describe()
print(df.head())
del df['AP']
del df['RH'] 
print(df.head())
train_data=df.sample(frac=0.8)

test_data=df.drop(train_data.index)

print(train_data.shape)
print(test_data.shape)
train_label=train_data.pop('PE')

test_label=test_data.pop('PE')
print(train_label.shape)
print(test_label.shape)

mf = [[['gaussmf',{'mean':np.mean(np.arange(0,21)),'sigma':np.std(np.arange(0,21))}],['gaussmf',{'mean':np.mean(np.arange(11,31)),'sigma':np.std(np.arange(11,31))}],['gaussmf',{'mean':np.mean(np.arange(20,41)),'sigma':np.std(np.arange(20,41))}]],
      [['gaussmf',{'mean':np.mean(np.arange(25,56)),'sigma':np.std(np.arange(25,55))}],['gaussmf',{'mean':np.mean(np.arange(45,76)),'sigma':np.std(np.arange(45,76))}],['gaussmf',{'mean':np.mean(np.arange(55,86)),'sigma':np.std(np.arange(55,86))}]]]
        
from membership import membershipfunction
mfc = membershipfunction.MemFuncs(mf)

import anfis
anf = anfis.ANFIS(train_data,train_label, mfc)

pred_train=anf.trainHybridJangOffLine(epochs=20)

train_label=np.reshape(train_label,[1,len(train_label)])
test_label=np.reshape(test_label,[1,len(test_label)])
print(train_label.shape)
print(test_label.shape)

error=np.mean((pred_train-train_label)**2)

print(error)
anf.plotErrors()
anf.plotResults()





