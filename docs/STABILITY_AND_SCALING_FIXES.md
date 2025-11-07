# Clustering Stability and Full Dataset Assignment

**Date**: 2025-11-06
**Status**: New implementation ready for testing

## Problem Summary

You identified three critical issues with the clustering pipeline:

1. **Stability evaluation failing**: Original approach tracked ~5 billion pairwise relationships for 100K genes, causing out-of-memory kills
2. **Need full dataset assignments**: Required cluster assignments for all 29M genes, not just the 1M subsample
3. **Need cluster statistics**: Mean and SD per dimension for each cluster across all 29M genes
4. **Low biological relevance**: Quality metrics showed ARI=0.001-0.005 (essentially random alignment with COG annotations)

## Solutions Implemented

### 1. Memory-Efficient Stability Evaluation

**New script**: `scripts/embeddings/evaluate_clustering_stability_efficient.py`

**Key changes**:
- ❌ **Old approach**: Track all gene pairs → O(N²) memory
- ✅ **New approach**: Cluster-level metrics → O(N) memory

**Metrics computed**:
1. **Pairwise ARI between clusterings**: Measures if independent subsamples produce similar clusterings
2. **Per-gene membership stability**: For each gene, tracks how many different clusters it was assigned to across subsamples
3. **Cluster count/size stability**: Checks if number and distribution of clusters are consistent

**Why this works**:
- If clustering is stable, the same gene should be assigned to the "same" cluster across different random samples
- ARI measures overall agreement between two clusterings
- Per-gene stability identifies which genes have consistent assignments

**Memory usage**: ~10GB instead of ~800GB

**Submission**:
```bash
sbatch scripts/embeddings/submit_stability_efficient.sh
```

This launches 4 array jobs for:
- res1500, nn15, COG-only (best performer)
- res1500, nn15, all genes
- res1000, nn15, COG-only
- res750, nn15, COG-only

**Runtime**: ~3-6 hours per configuration (vs 12+ hours with OOM failures)

---

### 2. Full Dataset Cluster Assignment

**New script**: `scripts/embeddings/assign_full_dataset_to_clusters.py`

**Strategy**:
1. Load best clustering from subsample (res1500, nn15, COG-only: 23,591 clusters)
2. Compute cluster centroids in 50D PCA space
3. Load all 29M genes in batches
4. Assign each gene to nearest cluster centroid (cosine similarity)
5. Compute cluster statistics (mean, SD per dimension) using Welford's online algorithm

**Key features**:
- **Batch processing**: Processes genomes one at a time → constant memory
- **PCA model fitting**: If PCA model not in cache, fits from data automatically
- **Two-pass algorithm**:
  - Pass 1: Assign all genes to clusters
  - Pass 2: Compute mean/SD statistics using online algorithm
- **Numerically stable**: Uses Welford's algorithm for variance computation

**Output** (`full_dataset_assignments_res1500_nn15_cogonly.npz`):
- `gene_ids`: All 29M gene IDs
- `genome_ids`: Genome for each gene
- `cluster_assignments`: Cluster ID for each gene
- `distances_to_centroid`: Distance from gene to its assigned cluster centroid
- `cluster_centroids`: (n_clusters, 50) array of cluster centers
- `cluster_means`: (n_clusters, 50) mean of each cluster
- `cluster_stds`: (n_clusters, 50) standard deviation of each cluster
- `cluster_sizes`: (n_clusters,) number of genes in each cluster

**Submission**:
```bash
sbatch scripts/embeddings/submit_full_dataset_assignment.sh
```

**Runtime**: ~12-24 hours (depends on I/O)
**Memory**: 256GB
**Output size**: ~2-3 GB compressed

---

### 3. Current Evaluation Status

**Quality evaluation** (Job 41390709): ✅ Running (21+ hours in)
- Computing ARI, AMI, silhouette, Davies-Bouldin for all 129 clusterings
- **Preliminary results show concerning patterns**:
  - ARI vs COG: 0.001-0.005 (very low!)
  - AMI vs COG: 0.17-0.23 (moderate)
  - Silhouette: -0.02 to 0.01 (poor separation)

**Interpretation**: Current clusterings may not be biologically meaningful. The low ARI suggests clusters don't align with known functional categories (COG).

**Old stability evaluation** (Jobs 41274295, etc.): ❌ All failed (OOM)
- Now replaced with efficient version above

---

## Key Findings from Current Results

### Best Clustering (Homogeneity-Based)
**Configuration**: res1500, nn15, COG-only
- **Clusters**: 23,591
- **Genes**: 798,416 (COG-annotated only)
- **Weighted homogeneity**: 61.7%
- **High-quality clusters**: 28% have ≥80% homogeneity

**However**: High homogeneity may be an artifact of small clusters. Need stability and quality metrics to confirm.

### Quality Metrics (Preliminary)
From ongoing quality evaluation:

| Metric | res100 | res500 | res1000 | res1500 |
|--------|--------|--------|---------|---------|
| ARI vs COG | 0.005 | 0.002 | 0.001 | ? |
| AMI vs COG | 0.18 | 0.21 | 0.22 | ? |
| Silhouette | 0.01 | -0.02 | -0.02 | ? |
| Davies-Bouldin | 2.4 | 2.2 | 2.2 | ? |

**Interpretation guides**:
- **ARI**: 0.0-0.2 = poor, 0.4-0.6 = moderate, 0.6-0.8 = good
- **Silhouette**: <0 = overlapping clusters, 0.25-0.5 = moderate, >0.5 = good
- **Our results**: Essentially random alignment with biological function!

---

## Next Steps

### Immediate Actions

1. **Wait for quality evaluation to complete** (should finish soon)
   ```bash
   # Check status
   squeue -j 41390709

   # View results when done
   cat results/clustering/quality_metrics_comprehensive.csv | column -t -s','
   ```

2. **Launch new stability evaluation**
   ```bash
   sbatch scripts/embeddings/submit_stability_efficient.sh
   # Runtime: ~3-6 hours per config (4 configs total)
   ```

3. **Launch full dataset assignment** (can run in parallel)
   ```bash
   sbatch scripts/embeddings/submit_full_dataset_assignment.sh
   # Runtime: ~12-24 hours
   ```

### Analysis After Jobs Complete

4. **Compare all metrics**
   ```bash
   python scripts/embeddings/compare_clustering_metrics.py
   ```

   This will:
   - Merge homogeneity, quality, and stability metrics
   - Compute composite scores
   - Identify optimal configuration balancing all objectives
   - Create comparison plots

5. **Investigate low biological relevance**

   If ARI/AMI remain low across all resolutions, this suggests:
   - ESM-C embeddings may not capture functional relationships well
   - COG categories may be too coarse for this resolution
   - Need different clustering approach or features

   **Possible fixes**:
   - Try different resolutions (lower res for broader functional groups)
   - Use sequence features (domains, motifs) in addition to embeddings
   - Try hierarchical clustering to find natural groupings
   - Use supervised/semi-supervised methods with COG as partial labels

---

## Expected Outcomes

### If Stability is Good (ARI > 0.6 between subsamples)
✅ Clusters are robust and reproducible
✅ Can confidently use res1500, nn15 for full dataset
✅ Proceed with functional annotation and downstream analysis

### If Stability is Moderate (ARI 0.3-0.6)
⚠️ Clusters have some variability
⚠️ May need to reduce resolution or increase subsample size
⚠️ Consider ensemble clustering (consensus from multiple runs)

### If Stability is Low (ARI < 0.3)
❌ Clusters are not reproducible
❌ High resolution may be overfitting to noise
❌ Need to reduce resolution significantly or change approach

---

## Understanding the Quality-Stability Trade-off

**Key insight**: Simply increasing resolution creates smaller, more homogeneous clusters, but this doesn't mean they're biologically meaningful or stable.

**Ideal clustering** should have:
1. **High homogeneity**: Genes in same cluster have same function
2. **High biological relevance** (ARI/AMI): Clusters align with known annotations
3. **High stability**: Same genes cluster together across different samples
4. **Good separation** (silhouette): Clusters are well-separated in embedding space
5. **Reasonable sizes**: Not too fragmented, not too lumped

**The challenge**: These objectives often conflict!
- Higher resolution → ⬆️ homogeneity, ⬇️ stability
- Lower resolution → ⬇️ homogeneity, ⬆️ stability
- Need to find sweet spot

---

## Files Created

### New Scripts
1. `scripts/embeddings/evaluate_clustering_stability_efficient.py` - Memory-efficient stability evaluation
2. `scripts/embeddings/assign_full_dataset_to_clusters.py` - Full 29M gene cluster assignment
3. `scripts/embeddings/submit_stability_efficient.sh` - SLURM submission for stability
4. `scripts/embeddings/submit_full_dataset_assignment.sh` - SLURM submission for full assignment

### Expected Outputs
1. `results/clustering/stability/stability_eff_res*_nn*_*.npz` - Stability metrics
2. `results/clustering/full_dataset_assignments_res1500_nn15_cogonly.npz` - All 29M assignments
3. `results/clustering/quality_metrics_comprehensive.csv` - Quality metrics (in progress)
4. `results/clustering/all_metrics_combined.csv` - Combined metrics (after comparison)

---

## Alternative Approaches (If Current Clustering Fails)

If stability/quality metrics show clustering isn't working well:

### Option 1: Lower Resolution Clustering
- Try res=100-500 range for broader functional groups
- Accept lower homogeneity in exchange for stability
- Manually refine clusters post-hoc

### Option 2: Hierarchical Clustering
- Build dendrogram to find natural groupings
- Cut at level that balances homogeneity and stability
- Allows multi-resolution analysis

### Option 3: Sequence-Based Clustering
- Use your original idea: pairwise Hamming/Levenshtein distances
- Cluster based on sequence similarity, not embeddings
- Embeddings can be secondary features

### Option 4: Hybrid Approach
- Coarse clustering with embeddings (res=100-300)
- Fine clustering within each coarse cluster using sequences
- Two-level hierarchy: functional → orthology

### Option 5: Supervised Methods
- Use COG as partial labels
- Semi-supervised learning to find clusters consistent with COG
- Label propagation or constrained clustering

---

## Memory and Runtime Estimates

| Task | Memory | CPUs | Time | I/O |
|------|--------|------|------|-----|
| Quality eval (batch) | 64GB | 8 | 4-6h | Low |
| Stability (efficient) | 128GB | 16 | 3-6h | Low |
| Full assignment | 256GB | 16 | 12-24h | **High** |
| Cluster stats | 256GB | 16 | 6-12h | **High** |

**Note**: Full assignment is I/O bound. Runtime depends heavily on storage system performance.

---

## Diversity Sampling (For Future)

The existing `diversity_subsample.py` script implements three strategies:

1. **Cluster-based**: K-means + uniform sampling per cluster
2. **MaxMin**: Greedy farthest-point sampling
3. **Hybrid**: Cluster + farthest-point within each

**When to use**: If you want to re-cluster on a different subsample that's more diverse than random sampling.

**Why it helps**:
- Ensures all regions of embedding space are represented
- Prevents dominant clusters from overwhelming signal
- Better for discovering rare functional groups

**Submission**:
```bash
sbatch scripts/embeddings/submit_diversity_subsample.sh
```

**Note**: This is orthogonal to current work. Only needed if you want to try a different subsample.

---

## Questions to Answer After Evaluation Completes

1. **What is the stability (ARI) between independent clusterings?**
   - If high (>0.6): Clusters are robust
   - If low (<0.3): Overfitting or noise

2. **What fraction of genes have stable cluster assignments?**
   - If >70%: Most genes consistently cluster
   - If <50%: Many genes are uncertain

3. **Does stability decrease with resolution?**
   - Expected: Higher res → lower stability
   - Helps identify resolution ceiling

4. **Is there a resolution that balances homogeneity and stability?**
   - Look for knee in the trade-off curve
   - Composite scores will help identify

5. **Why is ARI vs COG so low?**
   - Are COG categories too broad?
   - Are embeddings not capturing function?
   - Is clustering method inappropriate?

---

## Contact and Support

- **Scripts**: All in `scripts/embeddings/`
- **Docs**: `docs/README.md`, `docs/evaluation.md`
- **Logs**: `logs/` directory
- **Results**: `results/clustering/`

**For issues**:
1. Check SLURM logs in `logs/`
2. Verify input files exist
3. Check memory usage with `sacct -j JOBID --format=MaxRSS`

**For questions about approach**:
- See `docs/evaluation.md` for metrics interpretation
- See this file for next steps
