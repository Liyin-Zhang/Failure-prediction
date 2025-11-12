# Weather Aware Feeder Fault Prediction

This repository accompanies research on weather aware feeder fault prediction in distribution systems. The focus is graph structured modeling and the synthesis of rare fault events.

## Components
1. **GNN based predictor**  
   Feeder level fault risk estimation using meteorological and infrastructure features while respecting network topology.  
   * **Inputs**: node features such as wind speed, precipitation, temperature, pressure, humidity, and pole or span metadata; edge features such as branch flow direction and weather along branches.  
   * **Topology aware encoders**: interchangeable backbones including GAT, GATv2, TransformerConv, and GINE; both undirected and directed variants are supported, as well as cascaded and parallel undirected to directed designs.  
   * **Readout and head**: graph level representations are obtained with global mean pooling and global max pooling, followed by a lightweight multilayer perceptron that outputs a binary risk score.  
   * **Training options**: binary cross entropy or focal loss for class imbalance; optional gated fusion between undirected and directed branches; optional consistency regularizer to stabilize joint training.  
   * **Outputs**: per feeder risk probability and threshold based labels suitable for early warning and maintenance scheduling.  
   * **Evaluation**: ROC AUC, precision, recall, specificity, F1, and feeder level confusion analysis; results can be aggregated by feeder or scenario for stress testing.
   * 
2. **Conditional GAN based generator**  
   Synthesis of rare fault like extreme weather samples conditioned on normal weather states and feeder topology, improving robustness against class imbalance.  
   * **Objective**: learn a conditional mapping p(x_extreme | x_normal, G) where G is the feeder graph and x_normal are node features under normal weather.  
   * **Conditioning signals**: node features under normal weather, optional edge features, and the graph structure through edge_index with PyTorch Geometric batching.  
   * **Generator**: GNN backbone on concatenated [x_normal, z] with z as per node Gaussian noise. Backbones include SAGEConv, GAT, GIN, TransformerConv. MLP head outputs node level extreme features with the same dimensionality as x_extreme.  
   * **Discriminator**: dual branch design with a structural GNN branch that processes concatenated [x_normal, x_target] and an MLP branch that models the joint feature statistics. The two branches are fused and pooled to graph level to produce a real or fake score.  
   * **Losses**:  
     * BCE based adversarial loss for stable gradients on graph conditioned generation.  
     * Optional feature matching on first order statistics to align means.  
     * Optional variance matching to control spread.  
     * Optional conditional consistency L_cond to keep generated outputs physically close to normal states where appropriate.  
   * **Training recipe**:  
     * Sample per graph batches with DataLoader.  
     * Update the discriminator N_critic times for each generator step.  
     * StepLR scheduler with decay to improve late stage convergence.  
     * Deterministic seeds for reproducibility.  
 
## Data availability
Due to confidentiality of utility data, original datasets are not released at this time. Deidentified sample data and minimal runnable examples will be provided later to support reproduction and validation.

## Project status
The codebase is under active organization and continuous upload. Issues and pull requests are welcome.
