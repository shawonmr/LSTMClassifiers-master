import pandas as pd
import numpy as np
import math
import seaborn as sn 
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.datasets import make_classification
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle

#Dataset manipulation####################################################################

print('Reading Data')
dataset = read_csv('Challenge_dataset2.csv', index_col=0) # load dataset
dataset = dataset.reset_index() #reset index
dataset.drop('ID_TestSet',axis=1,inplace=True) #drop id
dataset.fillna(0,inplace=True) #fill na with 0
total = dataset['goal'].count()
goal = dataset['goal'] #keeps the goal /class
dataset.drop('goal',axis=1,inplace=True) #drop goal, keeps the features only
values = dataset.values #asigns to another variable
# ensure all data is float
values = values.astype('float32')
# normalize features
values = (dataset-dataset.min())/(dataset.max()-dataset.min())
sample_size =89 #Given sample size
timesteps = int(total/sample_size) #No of time steps for time series
data_dim = dataset.shape[1]  #No of features
num_classes = 2 #No of classes

# Generate training / validation data#######################################################################

print('Generating training / validation data set')
n_train = int(total*0.60)  #60% data for training 
n_test = total - n_train #rest for validation
X = np.reshape(np.ravel(values), (dataset.shape[0],1,data_dim)) #reshape for time series analysis using rnn
x_train = np.zeros((n_train,1, data_dim)) #training feature datasets
x_train = X[:n_train, :, :] 
Y = np.zeros((dataset.shape[0],num_classes))
#Two class assignment to each entry of class data 
for i in range(0,dataset.shape[0]):
     if goal.loc[i]==0:
        Y[i,0] = 1
        Y[i,1] = 0
     else:
         Y[i,0] = 0
         Y[i,1] = 1
y_train = np.zeros((n_train,num_classes)) #training class daatsets
y_train = Y[:n_train,:]
# Generate validation data
x_val = np.zeros((n_test, 1, data_dim))
x_val = X[n_train :, :, :] #copy from main feature data set
y_val=np.zeros((n_test,num_classes)) 
y_val = Y[n_train :, :] #copy from main class data set 


# RNN Model create / test / validation###############################################################

print('RNN model create / test/ validation')
model = Sequential()
model.add(LSTM(32, return_sequences=True,
batch_input_shape=(sample_size, 1, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32))  # return a single vector of dimension 32
model.add(Dense(num_classes, activation='sigmoid')) #output neuron equals no of classes to predict

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])  #Binary cross entropy for binary classification
       
        
history=model.fit(x_train, y_train,batch_size=sample_size,epochs=50,validation_data=(x_val, y_val),verbose=2, shuffle=False) #fit the model
#score = model.evaluate(x_val, y_val,batch_size=sample_size)
#print(score)
#plot history
#pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
#pyplot.legend()
#pyplot.show()


#####Confusion Matrix####################################################################################

rnn_pred = model.predict(x_val, batch_size=sample_size, verbose=1) 
rnn_predicted = np.argmax(rnn_pred, axis=1)
rnn_cm = confusion_matrix(np.argmax(y_val, axis=1), rnn_predicted) 
rnn_df_cm = pd.DataFrame(rnn_cm) 
#pyplot.figure(figsize = (20,14)) 
#sn.set(font_scale=1.4) #for label size 
#sn.heatmap(rnn_df_cm, annot=True, annot_kws={"size": 12}) # font size 
#pyplot.show()
rnn_report = classification_report(np.argmax(y_val, axis=1), rnn_predicted)
print(rnn_report)

#####ROC Curve Plot#################################################################################

print('ROC curve')
# Plot linewidth.
lw = 2
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_val[:, i], rnn_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(num_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
# Finally average it and compute AUC
mean_tpr /= num_classes


# Plot all ROC curves
pyplot.figure(1)

colors = cycle(['aqua', 'darkorange'])
for i, color in zip(range(num_classes), colors):
    pyplot.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

pyplot.plot([0, 1], [0, 1], 'k--', lw=lw)
pyplot.xlim([0.0, 1.0])
pyplot.ylim([0.0, 1.05])
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.title('Receiver operating characteristic of binary-classifier')
pyplot.legend(loc="lower right")
pyplot.show()
