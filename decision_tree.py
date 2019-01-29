import numpy as np
import pandas as pd
from ml.DecisionTree import DecisionTree
data = pd.read_csv('data/watermelon_2.csv',sep=',')
#data=data.drop(['contact','poutcome','pdays','balance'],axis=1)
#tree = DecisionTree(data,'y',['age','balance','duration','previous'])
data=data.drop(['编号'],axis=1)
tree = DecisionTree(data,'好瓜')
#tree.generate_subtree()

print(data.apply(lambda x:tree.predict(x),axis=1))