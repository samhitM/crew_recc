# Crew Scoring System: Threshold Selection Methods

## Overview

The crew scoring system now includes **two threshold selection methods** for level assignment:

### 1. **Percentile-Based Thresholds (Current Default)**
### 2. **KNN-Based Thresholds (Advanced Option)**

---

## How Threshold Selection Works

### **Percentile Method (Traditional)**

**Algorithm:**
```python
def calculate_percentile_thresholds(composite_scores, num_levels=5):
    scores = sorted(list(composite_scores.values()))
    thresholds = []
    for i in range(1, num_levels):
        percentile = i * (100 / num_levels)  # 20%, 40%, 60%, 80%
        threshold = np.percentile(scores, percentile)
        thresholds.append(threshold)
    return thresholds
```

**How it works:**
- Divides users into **equal-sized groups** based on score percentiles
- For 5 levels: splits at 20th, 40th, 60th, 80th percentiles
- **Guarantees even distribution** across all levels
- Simple, predictable, and stable

**Example Results:**
- Level 1: 17 users (20.0%)
- Level 2: 15 users (17.6%) 
- Level 3: 17 users (20.0%)
- Level 4: 17 users (20.0%)
- Level 5: 19 users (22.4%)

---

### **KNN Method (Data-Driven)**

**Algorithm:**
```python
def calculate_knn_thresholds(composite_scores, num_levels=5):
    # 1. Find optimal clusters using silhouette analysis
    optimal_clusters = find_optimal_clusters(scores, max_clusters=8)
    
    # 2. Apply K-means clustering on normalized scores
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(scores_normalized)
    
    # 3. Create multi-dimensional features
    features = [original_score, rank_position, normalized_score]
    
    # 4. Train KNN classifier on cluster assignments
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(features_array, cluster_labels)
    
    # 5. Calculate thresholds as boundaries between clusters
    thresholds = calculate_cluster_boundaries(cluster_info)
    return thresholds
```

**How it works:**
- Uses **silhouette analysis** to find optimal number of clusters
- Applies **K-means clustering** to find natural groupings
- Creates **multi-dimensional features** for better separation
- Trains **KNN classifier** to understand cluster boundaries
- Calculates thresholds as **midpoints between cluster boundaries**

**Key Advantages:**
1. **Data-driven clustering** - finds natural patterns in user behavior
2. **Optimal cluster detection** - uses silhouette score to determine best grouping
3. **Better separation** - creates meaningful boundaries between user types
4. **Adaptive** - adjusts to actual score distribution rather than forcing equal groups

---

## Implementation in the System

### **Graph-Based Components**

**1. Impression Scoring:**
```python
# Uses friendship table JSON relations to build NetworkX graph
# Calculates: PageRank, K-Shell, Out-Degree
# Linear regression learns feature weights (gaming_time = 100%)
impression_score = calculate_graph_metrics() + learned_feature_weights()
```

**2. Link Prediction (Jaccard + Katz):**
```python
# Jaccard coefficient for similarity-based prediction
jaccard_score = intersection(neighbors) / union(neighbors)

# Katz centrality for network influence  
katz_score = calculate_katz_centrality(graph, alpha=optimal_alpha)

# Combined link prediction
link_prediction = 0.6 * jaccard_score + 0.4 * katz_score
```

**3. Enhanced Community Detection:**
```python
# Connected components enhanced with link prediction quality
communities = detect_connected_components(graph)
community_score = base_score + quality_bonus + link_score + impression_influence
```

### **Final Composite Score**

```python
composite_score = (
    0.30 * gaming_activity_score +      # Gaming time, achievements
    0.25 * impression_score +           # PageRank, K-Shell, Out-Degree  
    0.20 * link_prediction_score +      # Jaccard + Katz centrality
    0.15 * bonus_factors +              # Engagement, participation
    0.10 * community_score              # Enhanced community detection
)
```

---

## KNN vs Percentile Comparison

### **When to Use Percentile Method:**
✅ **Balanced user experience** - ensures no level is empty  
✅ **Predictable results** - consistent distribution across time  
✅ **Simple implementation** - easy to understand and maintain  
✅ **Stable rankings** - users expect consistent level populations  

### **When to Use KNN Method:**
✅ **Natural user segmentation** - finds actual behavioral patterns  
✅ **Better discrimination** - separates distinct user types effectively  
✅ **Data-driven insights** - reveals hidden structure in user data  
✅ **Adaptive thresholds** - adjusts to changing user base characteristics  

### **Real Results Comparison:**

**Our System Analysis:**
- **Link prediction correlation with levels: 0.923** (very strong)
- **Optimal clusters found: 3-5** (varies by silhouette analysis)
- **Gaming time is primary signal** (100% feature weight from linear regression)
- **Network effects matter** (PageRank range: 0.004456 - 0.081886)

---

## Key Innovations Implemented

### **1. Data-Driven Feature Learning**
- Linear regression automatically learns feature importance
- Gaming time identified as primary signal (100% weight)
- All other features default to 0% (no meaningful data available)

### **2. Advanced Link Prediction**
- **Jaccard coefficient** for neighbor similarity
- **Katz centrality** for network influence propagation  
- **Combined scoring** (60% Jaccard + 40% Katz)
- **Strong correlation** with final levels (0.923)

### **3. Enhanced Community Detection**
- Connected components as base communities
- **Link prediction quality scores** for community enhancement
- **Multi-factor community scoring** (size + quality + individual + impression)

### **4. Robust Threshold Selection**
- **Silhouette analysis** for optimal cluster detection
- **Multi-dimensional clustering** (score + rank + normalized)
- **Fallback mechanisms** if clustering fails
- **Both methods available** for different use cases

---

## Files Generated

| File | Purpose | Contents |
|------|---------|----------|
| `crew_impressions_revised.csv` | Impression scoring results | PageRank, K-Shell, Out-Degree, Total Impression |
| `crew_levels_revised.csv` | Level assignments | Gaming, Impression, Community, Link Prediction, Levels |
| `comprehensive_crew_scoring_analysis.png` | Complete visualization | All components, distributions, correlations |

---

## Production Recommendations

### **Current Setup (Recommended):**
- **Primary**: Percentile-based thresholds for stable user experience
- **Secondary**: KNN analysis for insights and validation
- **Monitoring**: Track silhouette scores and cluster quality over time

### **Advanced Setup (Optional):**
- **A/B Testing**: Compare user satisfaction between threshold methods
- **Dynamic Switching**: Use KNN during major user base changes
- **Hybrid Approach**: KNN for insights, percentile for actual assignments

---

## Summary

The system successfully implements:

✅ **Graph-based impression scoring** with PageRank, K-Shell, and Out-Degree  
✅ **Data-driven feature weighting** using linear regression  
✅ **Link prediction** with Jaccard coefficient and Katz centrality  
✅ **Enhanced community detection** with link prediction quality  
✅ **Two threshold methods**: Percentile (stable) and KNN (adaptive)  
✅ **Comprehensive outputs** with CSV files and visualizations  

**The KNN method provides a more sophisticated, data-driven approach to threshold selection that finds natural patterns in user behavior, while the percentile method ensures stable and predictable level distributions for consistent user experience.**
