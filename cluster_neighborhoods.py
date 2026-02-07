import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN, KMeans
import matplotlib.pyplot as plt
import geopandas as gpd
import folium

# load the files
df = pd.read_csv("accessibility.csv")

print("Initial data sample:")
print(df.head())
print("\nColumns:", df.columns.tolist())
print("\nMissing values:\n", df.isna().sum())

# rename all columns for readability
df = df.rename(columns={
    "geometry/coordinates/0": "longitude",
    "geometry/coordinates/1": "latitude",
    "properties/label_type": "label_type",
    "properties/neighborhood": "neighborhood",
    "properties/severity": "severity",
    "properties/is_temporary": "is_temporary"
})

# clean data
df = df.dropna(subset=["longitude", "latitude", "neighborhood", "severity"])

df["severity_num"] = pd.to_numeric(df["severity"], errors="coerce")

# boolean handling
df["is_temporary_num"] = (
    df["is_temporary"]
    .fillna(0)
    .astype(bool)
    .astype(int)
)

df = df.dropna(subset=["severity_num"])

# DBSCAN clustering of spatial data
spatial_X = df[["longitude", "latitude"]]

spatial_scaler = StandardScaler()
spatial_scaled = spatial_scaler.fit_transform(spatial_X)

dbscan = DBSCAN(eps=0.5, min_samples=10)
df["issue_cluster"] = dbscan.fit_predict(spatial_scaled)

print("\nIssue Cluster Counts:")
print(df["issue_cluster"].value_counts())

# safety check for DBSCAN results
clustered_points = (df["issue_cluster"] != -1).sum()
if clustered_points < 10:
    raise ValueError(
        "DBSCAN produced too few clustered points. "
        "Increase eps or reduce min_samples."
    )

# aggregate to neighborhood level
neighborhood_df = (
    df[df["issue_cluster"] != -1]
    .groupby("neighborhood")
    .agg(
        issue_count=("issue_cluster", "count"),
        cluster_count=("issue_cluster", "nunique"),
        avg_severity=("severity_num", "mean"),
        temp_issue_ratio=("is_temporary_num", "mean")
    )
    .reset_index()
)

print("\nNeighborhood-level data:")
print(neighborhood_df.head())

# neighborhood features
neigh_features = neighborhood_df[
    ["issue_count", "cluster_count", "avg_severity", "temp_issue_ratio"]
]

neigh_scaler = StandardScaler()
neigh_scaled = neigh_scaler.fit_transform(neigh_features)

# find optimal K
inertias = []
sil_scores = []

if len(neighborhood_df) < 3:
    raise ValueError("Not enough neighborhoods for KMeans clustering.")

K = range(2, min(9, len(neighborhood_df)))

for k in K:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(neigh_scaled)
    inertias.append(km.inertia_)

    if len(set(labels)) > 1:
        sil_scores.append(silhouette_score(neigh_scaled, labels))
    else:
        sil_scores.append(-1)

# Save plots instead of showing them
plt.figure()
plt.plot(list(K), inertias, marker="o")
plt.xlabel("k")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.savefig("elbow_method.png", dpi=150, bbox_inches='tight')
plt.close()  # Close the figure instead of showing it

plt.figure()
plt.plot(list(K), sil_scores, marker="o")
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis")
plt.savefig("silhouette_analysis.png", dpi=150, bbox_inches='tight')
plt.close()  # Close the figure instead of showing it

k_opt = list(K)[sil_scores.index(max(sil_scores))]
print("Optimal number of clusters:", k_opt)

# KMeans clustering
kmeans = KMeans(n_clusters=k_opt, random_state=42, n_init=10)
neighborhood_df["neighborhood_cluster"] = kmeans.fit_predict(neigh_scaled)

# interpret clusters
cluster_profiles = (
    neighborhood_df
    .groupby("neighborhood_cluster")[[
        "issue_count", "cluster_count", "avg_severity", "temp_issue_ratio"
    ]]
    .mean()
)

print("\nCluster Profiles:")
print(cluster_profiles)

# Calculate accessibility score (lower is better/safer/more accessible)
# Higher issue count, more clusters, higher severity = worse accessibility
neighborhood_df["accessibility_score"] = (
    neighborhood_df["issue_count"] * 0.4 +
    neighborhood_df["cluster_count"] * 0.2 +
    neighborhood_df["avg_severity"] * 0.3 +
    neighborhood_df["temp_issue_ratio"] * 0.1
)

# Create descriptive cluster labels
def create_cluster_label(row):
    # Determine issue level based on issue_count and avg_severity
    if row["issue_count"] > neighborhood_df["issue_count"].quantile(0.66):
        issue_level = "High Issue"
    elif row["issue_count"] > neighborhood_df["issue_count"].quantile(0.33):
        issue_level = "Medium Issue"
    else:
        issue_level = "Low Issue"
    
    # Check if mostly temporary
    if row["temp_issue_ratio"] > 0.5:
        issue_level += ", Mostly Temporary"
    
    return issue_level

neighborhood_df["cluster_label"] = neighborhood_df.apply(create_cluster_label, axis=1)

# Find the most common issue type per neighborhood
main_issue_per_neighborhood = (
    df[df["issue_cluster"] != -1]
    .groupby("neighborhood")["label_type"]
    .agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else "Unknown")
    .reset_index()
    .rename(columns={"label_type": "main_issue_type"})
)

neighborhood_df = neighborhood_df.merge(
    main_issue_per_neighborhood,
    on="neighborhood",
    how="left"
)

neighborhood_df["main_issue_type"] = neighborhood_df["main_issue_type"].fillna("Unknown")

# save outputs
df.to_csv("issues_clustered.csv", index=False)
neighborhood_df.to_csv("neighborhoods_clustered.csv", index=False)

# map visualization
neighborhood_shapes = gpd.read_file("seattle_neighborhoods.geojson")
print("GeoJSON columns:", neighborhood_shapes.columns)

geo_col_name = "S_HOOD"  # update if needed
if geo_col_name not in neighborhood_shapes.columns:
    raise ValueError(f"{geo_col_name} not found in GeoJSON columns")


# normalize names to avoid merge failures
neighborhood_shapes[geo_col_name] = (
    neighborhood_shapes[geo_col_name].str.strip().str.lower()
)
neighborhood_df["neighborhood"] = (
    neighborhood_df["neighborhood"].str.strip().str.lower()
)

choropleth_df = neighborhood_shapes.merge(
    neighborhood_df,
    left_on=geo_col_name,
    right_on="neighborhood",
    how="left"
)

# For areas with no data, set to NaN (will appear without color)
choropleth_df["accessibility_score"] = choropleth_df["accessibility_score"].fillna(float('nan'))

seattle_map = folium.Map(location=[47.6062, -122.3321], zoom_start=11)

# Use RdYlGn_r (reversed Red-Yellow-Green) so that:
# - Red = high accessibility_score = many issues = LEAST accessible
# - Green = low accessibility_score = few issues = MOST accessible
folium.Choropleth(
    geo_data=choropleth_df,
    data=choropleth_df,
    columns=[geo_col_name, "accessibility_score"],
    key_on=f"feature.properties.{geo_col_name}",
    fill_color="RdYlGn_r",  # Reversed: Red (bad) to Yellow to Green (good)
    fill_opacity=0.7,
    line_opacity=0,  # Hide choropleth borders, we'll add them manually
    nan_fill_color="white",  # Areas with no data appear white
    nan_fill_opacity=0.0,  # Make areas with no data fully transparent
    legend_name="Accessibility Score (Red=Least Accessible, Green=Most Accessible)"
).add_to(seattle_map)

# Add blue borders and tooltips only to areas with data
choropleth_with_data = choropleth_df[choropleth_df['accessibility_score'].notna()]

folium.GeoJson(
    choropleth_with_data,
    style_function=lambda x: {
        'fillOpacity': 0,
        'color': 'blue',
        'weight': 2,
        'opacity': 0.5
    },
    tooltip=folium.features.GeoJsonTooltip(
        fields=[geo_col_name, "cluster_label", "main_issue_type", "issue_count", "avg_severity", "accessibility_score"],
        aliases=["Neighborhood", "Cluster Label", "Main Issue Present", "Issue Count", "Avg Severity", "Accessibility Score"],
        localize=True
    )
).add_to(seattle_map)

seattle_map.save("seattle_accessibility_map.html")
print("\nOutputs saved:")
print("- seattle_accessibility_map.html")
print("- elbow_method.png")
print("- silhouette_analysis.png")
print("- issues_clustered.csv")
print("- neighborhoods_clustered.csv")