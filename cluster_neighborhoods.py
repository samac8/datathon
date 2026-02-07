import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
import matplotlib.pyplot as plt

# load data
df = pd.read_csv('accessibility.csv')

print("Initial data sample:")
print(df.head())
print("\nColumns")
print(df.columns)
print("\nMissing values:")
print(df.isna().sum())

# rename columns
df = df.rename(columns={
    'geometry/coordinates/0': 'longitude',
    'geometry/coordinates/1': 'latitude',
    'properties/label_type': 'label_type',
    'properties/neighborhood': 'neighborhood',
    'properties/severity': 'severity',
    'properties/is_temporary': 'is_temporary'
})

# drop missing essential values
df = df.dropna(subset=['longitude', 'latitude', 'neighborhood', 'severity'])

# convert to numeric
df['severity_num'] = pd.to_numeric(df['severity'], errors='coerce')
df['is_temporary_num'] = df['is_temporary'].astype(int)

# drop rows where severity couldn't be converted
df = df.dropna(subset=['severity_num'])

# one-hot encode issue types
label_dummies = pd.get_dummies(df['label_type'], prefix='label')
df = pd.concat([df, label_dummies], axis=1)

# features for issue-level clustering
feature_cols = [
    'longitude',
    'latitude',
    'severity_num',
    'is_temporary_num'
] + list(label_dummies.columns)

X = df[feature_cols]

# scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# DBSCAN clustering (issue-level)
dbscan = DBSCAN(eps=0.7, min_samples=5)
df['issue_cluster'] = dbscan.fit_predict(X_scaled)

print("\nIssue Cluster Counts:")
print(df['issue_cluster'].value_counts())

neighborhood_df = (
    df[df['issue_cluster'] != -1]  # ignore noise
    .groupby('neighborhood')
    .agg(
        issue_count=('issue_cluster', 'count'),
        cluster_count=('issue_cluster', 'nunique'),
        avg_severity=('severity_num', 'mean'),
        temp_issue_ratio=('is_temporary_num', 'mean')
    )
    .reset_index()
)

# features for neighborhood clustering
neigh_features = neighborhood_df[
    ['issue_count', 'cluster_count', 'avg_severity', 'temp_issue_ratio']
]

# scale neighborhood features
scaler2 = StandardScaler()
neigh_scaled = scaler2.fit_transform(neigh_features)

# KMeans on neighborhoods
kmeans = KMeans(n_clusters=4, random_state=42)
neighborhood_df['neighborhood_cluster'] = kmeans.fit_predict(neigh_scaled)

# label clusters
def label_cluster(row):
    if row['avg_severity'] > 3 and row['issue_count'] > 3000:
        return "High issues, high severity"
    elif row['avg_severity'] <= 3 and row['temp_issue_ratio'] > 0.02:
        return "Low severity, mostly temporary"
    elif row['issue_count'] < 500:
        return "Few issues"
    else:
        return "Medium issues"
    
neighborhood_df['cluster_label'] = neighborhood_df.apply(label_cluster, axis=1)

# summary of neighborhood clusters
summary = (
    neighborhood_df
    .groupby('neighborhood_cluster')[[
        'issue_count',
        'cluster_count',
        'avg_severity',
        'temp_issue_ratio'
    ]]
    .mean()
)

print("\nNeighborhood Cluster Summary:")
print(summary)

df.to_csv("issues_clustered.csv", index=False)
neighborhood_df.to_csv("neighborhoods_clustered.csv", index=False)

# visualization
# issue-level clusters
plt.figure(figsize=(10, 8))
plt.scatter(
    df['longitude'],
    df['latitude'],
    c=df['issue_cluster'],
    cmap='tab10',
    s=5
)
plt.title("Accessibility Issue Clusters")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# neighborhood clusters
plt.figure(figsize=(10, 8))
for cluster in neighborhood_df['neighborhood_cluster'].unique():
    subset = df[df['neighborhood'].isin(
        neighborhood_df[neighborhood_df['neighborhood_cluster']==cluster]['neighborhood']
    )]
    plt.scatter(subset['longitude'], subset['latitude'], label=f'Cluster {cluster}', s=5)
plt.legend()
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Neighborhood Clusters by Accessibility Patterns")
plt.show()

