import numpy as np
import openml
import tabfm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

TASK_ID = 363612

task = openml.tasks.get_task(TASK_ID)
dataset = task.get_dataset()
x, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
split = task.download_split().split[0][0][0]


x_train = x.iloc[split.train]
x_test = x.iloc[split.test]
y_train = y.iloc[split.train]
y_test = y.iloc[split.test]


model = tabfm.tabfm_v1_0_0_pytorch.load(model_type="regression")
reg = tabfm.TabFMRegressor(model=model, random_state=0)
reg.fit(x_train, y_train)
pred = reg.predict(x_test)

rmse = mean_squared_error(y_test, pred) ** 0.5
r2 = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)

print("RMSE:", rmse)
print("R2:", r2)
print("MAE:", mae)