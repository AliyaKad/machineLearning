import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    data = data.drop(columns=['Order', 'PID'], errors='ignore')

    for col in data.columns:
        if data[col].dtype == 'object':
            data.loc[:, col] = data[col].fillna('Unknown')
        else:
            data.loc[:, col] = data[col].fillna(data[col].median())

    categorical_cols = data.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    return data, label_encoders


def remove_correlated_features(data, threshold=0.8):
    corr_matrix = data.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    data = data.drop(columns=to_drop)
    return data


def normalize_data(data):
    scaler = StandardScaler()
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    return data


def plot_3d(data, target_col):
    X = data.drop(columns=[target_col])
    y = data[target_col]

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_pca[:, 0], X_pca[:, 1], y, c=y, cmap='viridis')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel(target_col)
    plt.show()


def find_important_features(data, target_col, alpha=0.01):
    X = data.drop(columns=[target_col])
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)

    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': lasso.coef_
    })

    feature_importance = feature_importance.reindex(
        feature_importance['Coefficient'].abs().sort_values(ascending=False).index
    )

    return feature_importance


def main():
    data = load_data("AmesHousing.csv")

    data, _ = preprocess_data(data)

    data = remove_correlated_features(data)

    data = normalize_data(data)

    target_col = 'SalePrice'
    plot_3d(data, target_col)

    X = data.drop(columns=[target_col])
    y = data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    alphas = [0.01, 0.1, 1, 10, 100]
    rmse_scores = []

    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_scores.append(rmse)
        print(f"Alpha: {alpha}, RMSE: {rmse}")

    plt.figure(figsize=(8, 5))
    plt.plot(alphas, rmse_scores, marker='o')
    plt.xscale('log')
    plt.xlabel('Alpha (Regularization Strength)')
    plt.ylabel('RMSE')
    plt.title('RMSE vs Regularization Strength')
    plt.grid(True)
    plt.show()

    important_features = find_important_features(data, target_col, alpha=0.01)

    print("Наиболее важные признаки:")
    print(important_features)


if __name__ == '__main__':
    main()