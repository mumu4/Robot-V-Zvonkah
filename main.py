from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import seaborn as sns


plt.style.use('ggplot')


def main():
    # 1. Загрузка данных
    data = pd.read_csv(r"E:\PyCharm\features_dataset.csv")
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # 2. Подготовка данных
    X = data.drop('label', axis=1)
    y = data['label']

    # 3. Анализ корреляции признаков с целевой переменной
    plt.figure(figsize=(10, 8))
    corr_with_target = data.corr()[['label']].sort_values('label', ascending=False)
    sns.heatmap(corr_with_target, annot=True, cmap='coolwarm')
    plt.title("Корреляция признаков с целевой переменной")
    plt.show()

    # 4. Выбор наиболее информативных признаков
    top_features = corr_with_target.index[1:6]  # Берем топ-5 признаков (исключая саму метку)
    X = X[top_features]
    print(f"\ Используемые признаки: {list(top_features)}")

    # 5. Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # 6. Масштабирование
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 7. Обучение и оценка моделей
    models = {"RandomForest": RandomForestClassifier(n_estimators=100), "SVM": SVC(kernel='rbf'), }

    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        results[name] = {'accuracy': (y_pred == y_test).mean(), 'report': classification_report(y_test, y_pred)}

        # Визуализация матрицы ошибок
        cm = confusion_matrix(y_test, y_pred)
        plt.figure()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix: {name}")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

        print(f"\n{name} :")
        print(results[name]['report'])

        # 8. Кросс-валидация для лучшей модели
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        print("\nЛyчшая модель: {best_model_name}")

        cv_scores = cross_val_score(models[best_model_name], X_train_scaled, y_train, cv=5, scoring='accuracy')
        print(f"Cross-val scores: {cv_scores}")
        print(f"Mean accuracy: {cv_scores.mean():.2f}(+{cv_scores.std():.2f})")

        # 9. Сохранение модели
        joblib.dump(models[best_model_name], 'best_model.joblib')
        joblib.dump(scaler, 'scaler.joblib')
        print("\nМодель и скалер сохранены")


if __name__ == "__main__":
    main()
