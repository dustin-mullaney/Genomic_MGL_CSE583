# 1. Genome to Graph Pipeline
## Subcomponents
1.1 - K-mer profiling
- Purpose:
- Inputs:
- Outputs:

1.2 - Genome Parser
- Purpose:
- Inputs:
- Outputs:

1.3 - Multiple Sequence Alignment
- Purpose:
- Inputs:
- Outputs:

1.4 - ESM embedding + Clustering
- Purpose: Provide a low-dimensional representation of genomes and genes within those genomes and cluster them into groups based on similarity
- Inputs: Aligned protein sequences
- Outputs: List of embedding vectors (one per sequence) provided by ESM model and clusters of those embeddings from a user specified clustering algorithm

1.5 - Graph Assembly
- Purpose: Construct a genomic graph that can be processed by a Torch Geometric (graph neural network library)
- Inputs: User specified graph type (one of “gene-to-gene” or “genome-to-genome”), dictionary with genomes as keys and lists of genes as values.
- Outputs: List of node embedding vectors, an adjacency matrix specifying node connections, and tensor of edge embedding vectors (optional)

# 2. Masked Graph Learning
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