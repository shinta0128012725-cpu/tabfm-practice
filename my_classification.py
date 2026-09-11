import numpy as np 
import openml
import tabfm

# タスク(データセット)ID
TASK_ID = 363685

#データセットの用意(get_task → get_dataset → download_split)
task = openml.task.get_task(TASK_ID)
dateset = task.get_dataset()
x, y, _, _ = dataset.get_data()



