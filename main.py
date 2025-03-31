from itertools import combinations as comb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


iris = load_iris()
X = iris.data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def elbow_method(data, max_k=10):
    sse = []
    K_range = range(1, max_k + 1)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(K_range, sse, marker='o')
    plt.title('Метод локтя')
    plt.xlabel('Количество кластеров')
    plt.ylabel('Сумма квадратов ошибок')
    plt.grid()
    plt.show()

    deltas = np.diff(sse, 2)
    optimal_k = np.argmax(deltas) + 2
    print(f"Оптимальное количество кластеров: {optimal_k}")
    return optimal_k


def k_means(data, k, max_iters=100):
    def euclidean_distance(point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def initialize_centroids():
        indices = np.random.choice(data.shape[0], k, replace=False)
        return data[indices]

    def assign_clusters(centroids):
        clusters = [[] for _ in range(k)]
        for point in data:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(point)
        return clusters

    def update_centroids(clusters):
        return np.array([np.mean(cluster, axis=0) if cluster else np.zeros(data.shape[1]) for cluster in clusters])

    def has_converged(old_centroids, new_centroids, tol=1e-4):
        return all(euclidean_distance(old, new) < tol for old, new in zip(old_centroids, new_centroids))

    centroids = initialize_centroids()
    for i in range(max_iters):
        clusters = assign_clusters(centroids)
        new_centroids = update_centroids(clusters)

        visualize_step(clusters, centroids, title=f"Iteration {i + 1}")

        if has_converged(centroids, new_centroids):
            print(f"Сходимость достигнута на итерации {i + 1}")
            break
        centroids = new_centroids

    return clusters, centroids



def visualize_step(clusters, centroids, title):
    plt.figure(figsize=(6, 4))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for i, cluster in enumerate(clusters):
        cluster = np.array(cluster)
        if len(cluster) > 0:
            plt.scatter(cluster[:, 0], cluster[:, 1], color=colors[i % len(colors)], label=f'Cluster {i + 1}')
    centroids = np.array(centroids)
    plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='x', s=200, label='Centroids')
    plt.title(title)
    plt.legend()
    plt.show()


def plot_all_projections(data, clusters, centroids):
    features = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    combinations = list(comb(range(data.shape[1]), 2))

    num_plots = len(combinations)
    fig, axes = plt.subplots(nrows=num_plots // 2 + num_plots % 2, ncols=2,
                             figsize=(15, 5 * (num_plots // 2 + num_plots % 2)))
    axes = axes.flatten()

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for ax, (i, j) in zip(axes, combinations):
        for idx, cluster in enumerate(clusters):
            cluster = np.array(cluster)
            if len(cluster) > 0:
                ax.scatter(cluster[:, i], cluster[:, j], color=colors[idx % len(colors)], label=f'Cluster {idx + 1}')
        centroids = np.array(centroids)
        ax.scatter(centroids[:, i], centroids[:, j], color='black', marker='x', s=200, label='Centroids')
        ax.set_xlabel(features[i])
        ax.set_ylabel(features[j])
        ax.legend()

    plt.tight_layout()
    plt.show()

optimal_k = elbow_method(X_scaled)
clusters, centroids = k_means(X_scaled, optimal_k)
plot_all_projections(X_scaled, clusters, centroids)