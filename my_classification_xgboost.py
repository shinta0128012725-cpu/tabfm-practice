import numpy as np
import openml
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

TASK_ID = 363685

task = openml.tasks.get_task(TASK_ID)
dataset = task.get_dataset()
x, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
split = task.download_split().split[0][0][0]

x_train = x.iloc[split.train]
x_test = x.iloc[split.test]
y_train = y.iloc[split.train]
y_test = y.iloc[split.test]

# XGBoostは、文字列ラベルを自動変換してくれないので、自分でエンコードする
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

xgb_clf = xgb.XGBClassifier(random_state=42)
xgb_clf.fit(x_train, y_train_encoded)

y_pred_xgb = xgb_clf.predict(x_test)
acc_xgb = accuracy_score(y_test_encoded, y_pred_xgb)
print("XGBoostの正解率:", acc_xgb)