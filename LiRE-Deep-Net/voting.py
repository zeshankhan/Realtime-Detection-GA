# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 15:53:44 2022

@author: ZESHAN KAHN
"""

clfs=[LogisticRegression(random_state=0,solver='liblinear'),
          RandomForestClassifier(n_estimators=200, random_state=1),
          ExtraTreesClassifier(n_estimators=200, random_state=0),
          #SVC(gamma='auto'),
          LinearSVC(random_state=0, tol=1e-05),
          #KNeighborsClassifier(n_neighbors=17),
          DecisionTreeClassifier(random_state=2),
          #SGDClassifier(max_iter=1000, tol=1e-3),
          #GaussianNB()
    ]
clf=VotingClassifier(estimators=[(str(type(c)).split(".")[-1][:-2],c) for c in clfs], voting='hard')
clf.fit(train_X,train_Y)
pred_Y=clf.predict(test_X)
dff=pd.DataFrame(test_Y)
dff['Pred']=pred_Y
dff.to_csv('results_tst.csv')



f=f1_score(test_Y, pred_Y, average='weighted')
acc=accuracy_score(test_Y, pred_Y)
mcc=matthews_corrcoef(test_Y, pred_Y)
print("F1:",f,"\t ACC:",acc,"\tMCC:",mcc)