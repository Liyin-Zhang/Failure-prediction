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

2. **Conditional GAN based generator**  
   Synthesis of rare fault samples with preserved topological and statistical consistency to mitigate class imbalance and improve robustness.

## Data availability
Due to confidentiality of utility data, original datasets are not released at this time. Deidentified sample data and minimal runnable examples will be provided later to support reproduction and validation.

## Project status
The codebase is under active organization and continuous upload. Issues and pull requests are welcome.
