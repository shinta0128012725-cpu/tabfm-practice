import numpy as np
import openml
import tabfm
from sklearn.metrics import accuracy_score

"""データの準備(openml)"""

#maternal_health_risk(妊産婦の健康リスク、3クラス分類)のタスクID
TASK_ID = 363685

# ①タスク(あらかじめ定義された、データセット+評価方法のセット)を取得する
task = openml.tasks.get_task(TASK_ID)

# ②タスクに紐づく、元のデータセット本体を取得する
dataset = task.get_dataset()

# ③実際のデータ(x=特徴量、y=正解ラベル)を取り出す
x, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

# ④「訓練用とテスト用に、どう分割するか」という、公式の分割情報を取得する
split = task.download_split().split[0][0][0]

# ⑤分割情報(split)を使って、実際にデータを、訓練用・テスト用に切り分ける
x_train = x.iloc[split.train]
x_test = x.iloc[split.test]
y_train = y.iloc[split.train]  # ラベルはTabFMが内部でエンコードするので、生のまま渡してOK
y_test = y.iloc[split.test]


"""モデルの適用"""
model = tabfm.tabfm_v1_0_0_pytorch.load(model_type="classification")
clf = tabfm.TabFMClassifier(model=model, random_state=42)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print("正解率;", acc)