#Predicting the cancer using Support Vector Machine
#Import The Needed Packages
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import  pylab as pl

import scipy.optimize as opt

from sklearn import preprocessing,svm

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.metrics import f1_score,jaccard_similarity_score

import itertools

from sklearn.model_selection import train_test_split

%matplotlib inline

#Load data 
df=pd.read_csv("cell_samples.csv")
df.head()

#Plot the data
ax=df[df['Class']==4][0:50].plot(kind='scatter',x='Clump',y='Unifsize',color='DarkBlue',label='malignant');
df[df['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax);
plt.show()

#Show the Data Type of Column
#PRE-PROCESSING AND SELECTION
df.dtypes
df = df[pd.to_numeric(df['BareNuc'], errors='coerce').notnull()]
df['BareNuc'] = df['BareNuc'].astype('int')
df.dtypes

#FEATURE SELECTION
feature_df = df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X = np.asarray(feature_df)
X[0:5]
df['Class'] = df['Class'].astype('int')
y = np.asarray(df['Class'])
y [0:5]

#Train And Test Data
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


#MODEL CREATION AND TRAINING
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train) 

yhat=clf.predict(X_test)
yhat[0:5]

#EVALUATION OF THE MODEL
def plot_confusion_matrix(cm,classes,normalize=False,title='confusion_matrix',cmap=plt.cm.Blues)
"""
	This Function prints and plots the confusion matrix.
	Normalization can be applied by setting 'normalize=True'
	
		"""
if normalize:
	cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
	print("Normalize confusion matrix")
else:
	print('confusion matrix,without Normalization')

print(cm)

plt.imshow(cm,interpolation='nearest',cmap=cmap)
plt.title(title)
plt.colorbar()
tick_marks=np.arange(len(classes))
plt.xticks(tick_marks,classes,rotation=45)
plt.yticks(tick_marks,classes)

fmt=' .2f ' if normalize else 'd'
thresh = cm.max()/2
for i , j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
	plt.text(j,i,format(cm[i,j],fmt),horizontalalignment='center',
		     color='white' if cm[i,j] > thresh else "black")
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')

#Compute Confusion Matrix

cnf_matrix=confusion_matrix(y_test,yhat,labels=[2,4])
np.set_printoptions(precision=2)

print( classification_report(y_test,yhat))

#Plot Non-Normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix,classes=['Benign(2)','Malignant'],normalize=False,title='confusion_matrix')

#USING F1_SCORE , JACCARD INDEX FOR ACCURACY
f1_score(y_test,yhat,average='Weighted')

#USING JACCARD INDEX

jaccard_similarity_score(y_test,yhat)