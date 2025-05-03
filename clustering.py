from sklearn.cluster import KMeans

def perform_kmeans(df, n_clusters=3):
    model = KMeans(n_clusters=n_clusters)
    df['Cluster'] = model.fit_predict(df.select_dtypes(include='number'))
    return df
