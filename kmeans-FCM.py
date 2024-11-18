    import numpy as np
    from sklearn.cluster import KMeans
    from fcmeans import FCM
    from sklearn.metrics import silhouette_score
    from sklearn.datasets import load_digits
    import matplotlib.pyplot as plt
    import random
    from scipy.stats import mode
    
    digits = load_digits()
    X_mnist = digits.data
    y_mnist = digits.target
    n_clusters_mnist = 10
    
    def run_kmeans(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(data)
    score = silhouette_score(data, labels)
    return labels, score
    
    def run_fcm(data, n_clusters):
    fcm = FCM(n_clusters=n_clusters)
    fcm.fit(data)
    labels = fcm.u.argmax(axis=1)
    score = silhouette_score(data, labels)
    return labels, score
    
    def map_labels(true_labels, cluster_labels, n_clusters):
    label_mapping = {}
    for cluster in range(n_clusters):
    mask = (cluster_labels == cluster)
    if np.sum(mask) > 0:
    label_mapping[cluster] = mode(true_labels[mask])[0][0]
    return label_mapping
    
    kmeans_labels_mnist, kmeans_score_mnist = run_kmeans(X_mnist, n_clusters_mnist)
    fcm_labels_mnist, fcm_score_mnist = run_fcm(X_mnist, n_clusters_mnist)
    
    kmeans_label_map = map_labels(y_mnist, kmeans_labels_mnist, n_clusters_mnist)
    fcm_label_map = map_labels(y_mnist, fcm_labels_mnist, n_clusters_mnist)
    
    kmeans_mapped_labels = np.array([kmeans_label_map[label] for label in kmeans_labels_mnist])
    fcm_mapped_labels = np.array([fcm_label_map[label] for label in fcm_labels_mnist])
    
    num_images = 48 
    random_indices = random.sample(range(len(X_mnist)), num_images)
    
    fig, axes = plt.subplots(4, 12, figsize=(15, 5)) 
    for i, idx in enumerate(random_indices):
    axes[i // 12, i % 12].imshow(X_mnist[idx].reshape(8, 8), cmap='gray')
    true_label = y_mnist[idx]
    kmeans_label = kmeans_mapped_labels[idx]
    fcm_label = fcm_mapped_labels[idx]
    axes[i // 12, i % 12].set_title(f"True: {true_label}\nK-means: {kmeans_label}\nFCM: {fcm_label}", fontsize=6)
    axes[i // 12, i % 12].axis('off')
    
    plt.tight_layout()
    plt.savefig("cluster_results.png")
    plt.show()
    
    print(f"KMeans Silhouette Score: {kmeans_score_mnist}")
    print(f"FCM Silhouette Score: {fcm_score_mnist}")
