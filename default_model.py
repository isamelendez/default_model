import pickle 
import pandas as pd

X_test = pd.read_csv('x_test.csv')
y_test = pd.read_csv('y_test.csv')  

loaded_model = pickle.load(open('default_model.sav', 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)

y_pred = loaded_model.predict(X_test)
y_test.to_csv('y_pred.csv') 