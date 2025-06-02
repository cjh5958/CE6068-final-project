import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# 載入兩份資料
df1 = pd.read_csv("datasets/raw_data.csv")
df2 = pd.read_csv("datasets/generated_data.csv")

# 前處理（假設格式相同）
df1 = df1.drop(columns=["samples"])
df2 = df2.drop(columns=["samples"])

# 移除缺失值
df1 = df1.dropna()
df2 = df2.dropna()

# 拆開 X 和 y
X1 = df1.drop(columns=['type'])
y1 = df1['type']

X2 = df2.drop(columns=['type'])
y2 = df2['type']

# 對 y 做 label encoding（使用同一個 encoder 確保一致性）
le = LabelEncoder()
y1_encoded = le.fit_transform(y1)
y2_encoded = le.transform(y2)  # 注意：假設 df2 裡的 type 沒有新類別！

# 用 data1 切 train/test
X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y1_encoded, test_size=0.5, stratify=y1_encoded, random_state=42
)

# 把 data2 加進 train
X_train = pd.concat([pd.DataFrame(X1_train), pd.DataFrame(X2)], ignore_index=True)
y_train = pd.concat([pd.Series(y1_train), pd.Series(y2_encoded)], ignore_index=True)

# 測試資料仍然只用 data1 的一半
X_test = X1_test
y_test = y1_test


# 模型定義
base_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    #"Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "MLP": MLPClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    #"XGBoost": XGBClassifier(eval_metric='logloss', random_state=42),
    #"LightGBM": LGBMClassifier(random_state=42,force_col_wise=True,),
    #"CatBoost": CatBoostClassifier(verbose=0, random_state=42),
}

"""# 加入 Stacking 模型
stacking_model = StackingClassifier(
    estimators=[
        ('rf', base_models["Random Forest"]),
        ('gb', base_models["Gradient Boosting"]),
        ('knn', base_models["KNN"])
    ],
    final_estimator=LogisticRegression(),
    cv=5
)
base_models["Stacking"] = stacking_model"""

# 標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
models_need_scaling = ["KNN", "MLP", "Logistic Regression", "SVC"]

# 開啟輸出檔案
output_path = "model_results.txt"
with open(output_path, "w", encoding="utf-8") as f:
    for model_name, model in base_models.items():
        print(f"\n🔧 Training {model_name}...")
        f.write(f"\n🔧 Training {model_name}...\n")

        if model_name in models_need_scaling:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        y_test_labels = le.inverse_transform(y_test)
        y_pred_labels = le.inverse_transform(y_pred)

        # 混淆矩陣與報告
        cm = confusion_matrix(y_test_labels, y_pred_labels)
        cr = classification_report(y_test_labels, y_pred_labels)
        acc = accuracy_score(y_test_labels, y_pred_labels)

        # 印出與寫入檔案
        print(f"{model_name} Confusion Matrix:")
        print(cm)
        print(f"{model_name} Classification Report:")
        print(cr)
        print(f"{model_name} Accuracy: {acc:.4f}")
        print("-" * 60)

        f.write(f"{model_name} Confusion Matrix:\n{cm}\n")
        f.write(f"{model_name} Classification Report:\n{cr}\n")
        f.write(f"{model_name} Accuracy: {acc:.4f}\n")
        f.write("-" * 60 + "\n")

print(f"\n✅ 所有模型結果已儲存至：{output_path}")