import numpy as np
from sklearn import datasets,svm

iris = datasets.load_iris()

c1 = iris.data[:50]
c2 = iris.data[50:100]
c3 = iris.data[100:]
t1 = iris.target[:50]
t2 = iris.target[50:100]
t3 = iris.target[100:]
'''
#class 0 & 1  
x_train = np.append(c1[:20],c2[:20],axis=0)
x_validation = np.append(c1[20:40],c2[20:40],axis=0)
x_test = np.append(c1[40:],c2[40:],axis=0)
y_train = np.append(t1[:20],t2[:20],axis=0)
y_validation = np.append(t1[20:40],t2[20:40],axis=0)
y_test = np.append(t1[40:],t2[40:],axis=0)
'''
#class 1 & 2
x_train = np.append(c2[:20],c3[:20],axis=0)
x_validation = np.append(c2[20:40],c3[20:40],axis=0)
x_test = np.append(c2[40:],c3[40:],axis=0)
y_train = np.append(t2[:20],t3[:20],axis=0)
y_validation = np.append(t2[20:40],t3[20:40],axis=0)
y_test = np.append(t2[40:],t3[40:],axis=0)

#print len(x_train),len(x_validation),len(x_test),len(y_train),len(y_validation),len(y_test)
C = 1
D = 1
C = input('C:')
D = input("D:")
#cls = svm.SVC(C=C,kernel="linear")
#cls = svm.SVC(C=C,kernel='poly',degree=D)
cls = svm.SVC(C=C,kernel='rbf')
cls.fit(x_train,y_train)

#pred = cls.predict(x_validation)
pred=cls.predict(x_test)
#1asT 0asF
TP=0
FP=0
TN=0
FN=0
T=2
F=1
'''
#valid
for i in range(len(x_validation)):
	if(pred[i]==T and y_validation[i]==T):
		TP+=1
	elif(pred[i]==T and y_validation[i]==F):
		FP+=1
	elif(pred[i]==F and y_validation[i]==F):
		TN+=1
	elif(pred[i]==F and y_validation[i]==T):
		FN+=1
	else:
		print "error at index:%d" % i
'''
#test
for i in range(len(x_test)):
	if(pred[i]==T and y_test[i]==T):
		TP+=1
	elif(pred[i]==T and y_test[i]==F):
		FP+=1
	elif(pred[i]==F and y_test[i]==F):
		TN+=1
	elif(pred[i]==F and y_test[i]==T):
		FN+=1
	else:
		print "error at index:%d" % i

print '\nTP:',TP,'\nFP:',FP,'\nTN:',TN,'\nFN:',FN
if(TP+FP):
	P=float(TP)/(TP+FP)
else:
	P=-1
if(TP+FN):
	R=float(TP)/(TP+FN)
else:
	R=-1
if(P+R):
	F1=2.0*P*R/(P+R)
else:
	F1=-1
print '\nP:',P,'\nR:',R,'\nF1:',F1