# Current Clustering Pipeline Status

**Updated**: 2025-11-06 (Comprehensive Parameter Sweep)
**Status**: ✅ 50 stability evaluations running in parallel

---

## 🚀 Comprehensive Stability Sweep Running

### Job Array 41488384 (50 tasks)
**Script**: `evaluate_clustering_stability_efficient.py`
**Status**: ✅ Running
**Runtime**: ~3-6 hours per task
**Expected completion**: ~6-8 hours

**Parameter Grid**:
- **Resolutions**: 300, 500, 750, 1000, 1500 (5 values)
- **n_neighbors**: 15, 25, 50, 100, 200 (5 values)
- **Datasets**: COG-only (798K), all genes (1M) (2 values)
- **Total**: 5 × 5 × 2 = **50 configurations**

### What Each Config Tests

| Resolution | Expected Clusters (COG-only) | Expected Clusters (All genes) |
|-----------|------------------------------|-------------------------------|
| 300 | ~6,900 | ~7,200 |
| 500 | ~10,900 | ~11,400 |
| 750 | ~13,500 | ~14,200 |
| 1000 | ~17,800 | ~18,600 |
| 1500 | ~23,600 | ~25,000 |

**For each resolution**, testing:
- n=15, 25, 50, 100, 200 (effect of kNN graph connectivity)
- COG-only vs all genes (effect of dataset filtering)

---

## What This Tests

### 1. **Resolution vs Stability**
Does higher resolution (more clusters) decrease stability?
- If yes: Find optimal resolution where stability starts dropping
- If no: Can safely use highest resolution for maximum granularity

### 2. **n_neighbors vs Stability**
Does kNN graph connectivity affect clustering stability?
- Lower n_neighbors: Local structure, potentially more stable
- Higher n_neighbors: Global structure, potentially smoother

### 3. **COG-only vs All Genes**
Does including unannotated genes affect stability?
- COG-only: Cleaner signal (79.8% of genes)
- All genes: Complete dataset

---

## Full Dataset Assignment

### Job 41487126
**Status**: ⏸️ Pending
**When**: Will run after stability jobs finish
**What**: Assigns all 29M genes to clusters based on res1500, nn15, COG-only

---

## After Jobs Complete

### 1. Comprehensive Summary (~8 hours from now)

```bash
python scripts/embeddings/summarize_stability_results.py
```

**Output**:
- Stability metrics across all 50 configurations
- Heatmaps: resolution × n_neighbors for each metric
- Recommendations for optimal parameters

**You'll see**:
```
Resolution  n_neighbors  Dataset    ARI_mean  Gene_Stability  N_Clusters
300         15           COG-only   0.XXX     0.XXX          ~6,900
300         25           COG-only   0.XXX     0.XXX          ~6,900
...         ...          ...        ...       ...            ...
1500        200          all        0.XXX     0.XXX          ~25,000
```

### 2. Analysis Questions Answered

**Q1: Does stability decrease with resolution?**
- Compare ARI across resolutions (300 → 1500)
- Find where trade-off curve bends

**Q2: What's the best n_neighbors?**
- Compare ARI across n_neighbors for each resolution
- Identify if there's a clear winner

**Q3: Should we use COG-only or all genes?**
- Compare stability between datasets
- Check if unannotated genes add noise

**Q4: What's the optimal configuration?**
- Highest resolution with ARI > 0.6 (if any)
- Best balance of granularity and stability

---

## Expected Outcomes

### Scenario 1: High stability everywhere (ARI > 0.7)
✅ **Use highest resolution** (res=1500, best n_neighbors)
- Clusters are robust even at fine granularity
- ~23,600 clusters are stable and meaningful
- Proceed with full 29M assignment

### Scenario 2: Stability decreases with resolution
⚠️ **Find the knee in the curve**
- E.g., res=750 stable but res=1000+ unstable
- Use highest stable resolution
- Trade some granularity for robustness

### Scenario 3: n_neighbors matters significantly
🔧 **Optimize n_neighbors**
- May find that n=25 or n=50 gives best stability
- Update full assignment to use optimal n_neighbors

### Scenario 4: Low stability everywhere (ARI < 0.4)
❌ **Need different approach**
- Leiden clustering may not be ideal for this data
- Consider: hierarchical clustering, different features, sequence-based

---

## Key Metrics to Watch

### 1. **ARI (Adjusted Rand Index)**
Measures clustering agreement between independent subsamples
- **> 0.8**: Excellent - clusters highly reproducible
- **0.6-0.8**: Good - clusters generally stable
- **0.4-0.6**: Moderate - some variability
- **< 0.4**: Poor - not reproducible

### 2. **Gene Stability Mean**
Fraction of genes assigned to same cluster across samples
- **> 0.9**: Most genes have consistent assignments
- **0.7-0.9**: Good consistency
- **< 0.7**: Many genes have uncertain placements

### 3. **% Genes with Stability > 0.9**
What fraction of genes are highly stable?
- **> 70%**: Most genes confidently assigned
- **50-70%**: Moderate confidence
- **< 50%**: Many uncertain assignments

### 4. **Cluster Count Stability (Std)**
How much does # clusters vary across subsamples?
- **Low std**: Consistent cluster counts
- **High std**: Variable (less stable)

---

## Monitoring Progress

```bash
# Check how many jobs are running
squeue -u dmullane | grep 41488384 | wc -l

# Watch a specific job
tail -f logs/stability_eff_41488384_0.out  # res=300, nn=15, COG-only
tail -f logs/stability_eff_41488384_24.out # res=750, nn=15, COG-only
tail -f logs/stability_eff_41488384_48.out # res=1500, nn=15, COG-only

# Check which ones have finished
ls results/clustering/stability/stability_eff_*.npz | wc -l

# When all done (should have 50 files)
ls results/clustering/stability/ | wc -l
```

---

## Understanding Low ARI vs COG (Not a Problem!)

You correctly noted: **24 COG categories** but **10,000-25,000 clusters**

**This means**:
- Each COG category → hundreds of sub-clusters (ortholog groups)
- Example: COG category "J" (translation) → L1, L2, L3, ..., L12, S1, S2, ... (100+ ribosomal proteins)
- **ARI vs COG will be low** - this is expected!
- Clusters represent **orthology within functional categories**, not categories themselves

**What matters for your use case**:
1. ✅ **Homogeneity**: 61.7% (clusters are functionally coherent)
2. 🔄 **Stability**: Testing now (reproducibility across samples)
3. ✅ **Genes without COG**: Fine to cluster with known genes

**Hierarchy**:
```
COG Category (e.g., "J - Translation")
  └─ Your Cluster 1234 (e.g., "50S ribosomal protein L7/L12")
      ├─ Gene 1 (E. coli rplL)
      ├─ Gene 2 (S. aureus rplL)
      └─ Gene 3 (Unknown species, unannotated but orthologous)
```

---

## Next Steps After Results

### 1. Generate Summary
```bash
python scripts/embeddings/summarize_stability_results.py
```

### 2. Create Heatmaps (Optional)
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load all results
stability_files = sorted(Path('results/clustering/stability').glob('*.npz'))

# Build matrix: resolutions × n_neighbors
resolutions = [300, 500, 750, 1000, 1500]
n_neighbors = [15, 25, 50, 100, 200]
ari_matrix = np.zeros((len(resolutions), len(n_neighbors)))

for f in stability_files:
    data = np.load(f)
    if data['cog_only']:  # Focus on COG-only
        res_idx = resolutions.index(int(data['resolution']))
        nn_idx = n_neighbors.index(int(data['n_neighbors']))
        ari_matrix[res_idx, nn_idx] = data['ari_mean']

# Plot heatmap
plt.figure(figsize=(10, 8))
plt.imshow(ari_matrix, cmap='RdYlGn', vmin=0, vmax=1)
plt.colorbar(label='ARI')
plt.xticks(range(len(n_neighbors)), n_neighbors)
plt.yticks(range(len(resolutions)), resolutions)
plt.xlabel('n_neighbors')
plt.ylabel('Resolution')
plt.title('Clustering Stability Heatmap (COG-only)')
plt.savefig('results/plots/stability_heatmap.png', dpi=150)
```

### 3. Choose Best Configuration
Based on stability results and your downstream needs:
- **If you need maximum resolution**: Use highest stable resolution
- **If you prioritize stability**: Use resolution with highest ARI
- **If you need balanced**: Use composite score

### 4. Run Full Assignment
Once you know the optimal parameters:
```bash
# Edit submit_full_dataset_assignment.sh to use optimal clustering
# Change: CLUSTERING_FILE to best configuration
sbatch scripts/embeddings/submit_full_dataset_assignment.sh
```

---

## Files & Directories

### Scripts
- ✅ `evaluate_clustering_stability_efficient.py` - Stability evaluation
- ✅ `submit_stability_efficient.sh` - 50-task array job
- ✅ `summarize_stability_results.py` - Results summary
- ✅ `assign_full_dataset_to_clusters.py` - Full 29M assignment

### Output
- `results/clustering/stability/` - Will contain 50 .npz files
- `results/clustering/stability_summary.csv` - Summary table
- `results/clustering/full_dataset_assignments_*.npz` - Final assignments

### Logs
- `logs/stability_eff_41488384_*.out` - Progress logs
- `logs/stability_eff_41488384_*.err` - Error logs

---

## Resource Usage

**Current jobs**:
- 50 tasks × 128GB memory × 16 CPUs
- ~150-300 node-hours total
- All running in parallel (cluster permitting)

**Memory**: No OOM failures - using efficient cluster-level metrics

**I/O**: Moderate - loading PCA cache repeatedly

**Time**: Most should finish in 3-6 hours

---

## Why This Comprehensive Sweep Matters

Testing the full parameter space lets you:

1. **Understand trade-offs**: Resolution vs stability, n_neighbors vs stability
2. **Avoid overfitting**: Ensure clusters generalize beyond your specific 1M sample
3. **Optimize confidence**: Choose parameters that maximize both granularity and reproducibility
4. **Publication ready**: Can show robustness analysis with 50 configurations
5. **Future-proof**: If you re-run with different samples, you know expected stability

**This is the right way to validate clustering** - not just "it looks good" but "it's reproducible and robust."

---

## Summary

- ✅ **50 stability evaluations** running in parallel
- ✅ **Comprehensive parameter sweep**: 5 resolutions × 5 n_neighbors × 2 datasets
- ✅ **Memory efficient**: No OOM failures
- ✅ **Full coverage**: Tests all existing clustering configurations
- ⏸️ **Full dataset assignment**: Pending, will use optimal parameters
- 📊 **Results expected**: ~6-8 hours

**Next action**: Wait for jobs to finish, then run summary script!
