# Component 1: Genome to Graph Pipeline
**Directory**: `1_genome_to_graph/`
## Subcomponents
1.1 - K-mer profiling
- Purpose: In lieu of a genomic LLM that can embed *full* bacterial genomes (there isn't a model that currently exists that is good for this purpose and is easy to implement), k-mer profiles will serve as attribute vectors for genome nodes. We will use 5 or 6-mer (the counts of each 5 or 6 nucleotide sequence in each genome) profiles to create node attributes for each genome in the graph. K-mer profiles capture a lot of evolutionary information about bacterial genomes. 
- Inputs: Raw genome sequences in fasta format
- Outputs: a vector of frequencies for each 5 or 6 mer

1.2 - Genome Parser
- Purpose: The first step in defining gene nodes in the graph is parsing gene sequences from each genome. This component will use `prodigal` or similar software to identify protein coding regions of each genome
- Inputs: Raw genome sequences in fasta format
- Outputs: Fasta files with sequences and unique IDs for all protein coding gene sequences in each genome

1.3 - Multiple Sequence Alignment
- Purpose: Because there are 29M discrete protein coding genes between the ~7k genomes in refseq, it is infeasible to cluster all of them in the ESM embedding space. For this reason, we will first perform multiple sequence alignment to group genes with similar sequence, to reduce redundancy and bias in the embedding space. 
- Inputs: Gene sequences from 1.2
- Outputs: Groups of sequences within a certain homology threshold

1.4 - ESM embedding + Clustering
- Purpose: Provide a low-dimensional representation of gene sequence, structure and function and cluster them into groups based on similarity. Clusters will define gene nodes in the graph, and their esm embedding vectors will be the node attributes
- Inputs: Aligned protein sequences
- Outputs: List of embedding vectors (one per sequence) provided by ESM model and clusters of those embeddings from a user specified clustering algorithm

1.5 - Graph Assembly
- Purpose: Construct a genomic graph that can be processed by a Torch Geometric (graph neural network library)
- Inputs: User specified graph type (one of “gene-to-gene” or “genome-to-genome”), dictionary with genomes as keys and lists of genes as values.
- Outputs: List of node embedding vectors, an adjacency matrix specifying node connections, and tensor of edge embedding vectors (optional)

**Results Directories**:

`results/1_genome_to_graph/1.3_msa/` - MMseqs2 clustering results
- `mmseqs_seqid_0p3/` through `mmseqs_seqid_0p7/` - Parameter sweep results
- `mmseqs_full_dataset/` - Full dataset clustering
- **Active**: 70% sequence identity (12.2M proteins in 388K clusters)

`results/1_genome_to_graph/1.4_esm_embedding_clustering/`
- `clustering/` - Leiden clustering analysis (CSV summaries only, .npz files removed)
- `umap/` - Will contain new UMAP visualizations from full embeddings
- `functional_annotation/` - COG and eggNOG annotations for all genomes
- `metadata/` - COG category metadata
- `plots/` - Early small-scale testing visualizations (3-5% coverage)

# Component 2: Masked Graph Learning
## Subcomponents
2.1 - Dataset Creation Module
- Purpose: Create genomic graphs using genome to graph pipeline and save a large compendium of these graphs
- Inputs: User specified list of genomes to encode as graphs in a dataset
- Outputs: A Torch Geometric Data object which specifies node features and adjacency matrices for many genomes

2.2 - Data loading module
- Purpose: Manage data loading, preprocessing, and batching for training.
- Inputs: A Torch Geometric Data object, user specified batch size
- Outputs: A Torch Geometric DataLoader object

2.3 - Masking Component
- Purpose: Randomly mask connections between nodes (genes) that there is a known evolutionary connection between
- Inputs: Graph of genes and genomes with edges fully specified and a masking ratio
- Outputs: New genomic graphs with some portion of edges removed

2.4 - Graph Encoder Component (GNN Model)
- Purpose: Learn latent representations of the graph.
- Inputs: A genomic graph with a portion of gene-to-gene edges masked
- Outputs: A list of latent node embeddings

2.5 - Graph Decoder
- Purpose: Decode a latent representation of the genomic graph and reconstruct a graph with new edges added to represent connections between genes which were not originally specified
- Inputs: A list of latent node embeddings
- Outputs: Decoded node embeddings (target: ESM embeddings) and an adjacency matrix specifying gene-to-gene connections (target: all original gene-to-gene connections + masked edges)