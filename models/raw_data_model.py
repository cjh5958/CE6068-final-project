import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

SEED = 42

# 載入資料
data_path = "datasets/raw_data_reduced.csv"
df = pd.read_csv(data_path)

# 前處理
df = df.drop(columns=["samples"])
df = df.dropna()
X = df.drop(columns=['type'])
y_raw = df['type']

# 編碼 label
le = LabelEncoder()
y = le.fit_transform(y_raw)

# 分割資料
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, stratify=y, random_state=SEED
)

# 模型定義
base_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=SEED),
    #"Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
    "MLP": MLPClassifier(random_state=SEED),
    "Random Forest": RandomForestClassifier(random_state=SEED),
    #"XGBoost": XGBClassifier(eval_metric='logloss', random_state=SEED),
    #"LightGBM": LGBMClassifier(random_state=SEED,force_col_wise=True,),
    #"CatBoost": CatBoostClassifier(verbose=0, random_state=SEED),
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
output_path = "rawdata_model_results_" + str(SEED) + ".txt"
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