# Weather Aware Feeder Fault Prediction

This repository accompanies research on weather aware feeder fault prediction in distribution systems. The focus is graph structured modeling and the synthesis of rare fault events.

## Components
1. **GNN based predictor**  
   Feeder level fault risk estimation using meteorological and infrastructure features while respecting network topology.
2. **Conditional GAN based generator**  
   Synthesis of rare fault samples with preserved topological and statistical consistency to mitigate class imbalance and improve robustness.

## Data availability
Due to confidentiality of utility data, original datasets are not released at this time. Deidentified sample data and minimal runnable examples will be provided later to support reproduction and validation.

## Project status
The codebase is under active organization and continuous upload. Issues and pull requests are welcome.


"""
Model lineup notes for the train file

01 Single branch undirected graph
Discrete node features are embedded, then two GATConv layers extract undirected topology representations.
GAP and GMP provide graph readout, followed by a three layer MLP for binary classification.

02 Single branch directed graph with GAT
Discrete node features are embedded, then two GATConv layers encode the directed graph.
GAP and GMP provide readout, followed by an MLP for binary classification.

03 Single branch undirected graph with dynamic edge weights
Discrete node features are embedded, then two TransformerConv layers operate on the undirected graph
and use undirected edge attributes for message passing. GAP and GMP provide readout, followed by an MLP
for binary classification.

04 Single branch directed graph with TransformerConv
Discrete node features are embedded, then two TransformerConv layers with edge attributes encode the
directed graph. GAP and GMP provide readout, followed by an MLP for binary classification.

05 Cascaded undirected to directed
Two TransformerConv layers first encode the undirected graph to obtain H_und.
H_und is then fed to two TransformerConv layers on the directed graph to obtain H_dir.
Both branches are read out with GAP and GMP, then fused by a gating module at feature level or at logit level
to produce a binary probability. An optional node level consistency regularizer is available.

06 Parallel undirected and directed with TransformerConv
Starting from the same embedded node features, two TransformerConv layers encode the undirected graph
and another two encode the directed graph to obtain H_und and H_dir. Both branches are read out with GAP and GMP,
then fused by a gating module at feature level or at logit level. An optional node level consistency regularizer is available.

07 Parallel undirected and directed with GATv2
Starting from the same embedded node features, two GATv2Conv layers encode the undirected graph to get H_und
and two GATv2Conv layers encode the directed graph to get H_dir. Both branches are read out with GAP and GMP,
then fused by a gating module at feature level or at logit level. An optional consistency regularizer is supported.

08 Parallel undirected and directed with GINE
Starting from the same embedded node features, two GINEConv layers encode the undirected graph into H_und
and two GINEConv layers encode the directed graph into H_dir. GAP and GMP provide readout for both branches.
A gating module fuses them at feature level or at logit level. An optional node level consistency regularizer is supported.

09 Parallel undirected and directed with TransformerConv for focal loss training
Discrete node features are embedded, then two TransformerConv layers encode the undirected graph and two encode the directed graph
to obtain H_und and H_dir. GAP and GMP provide readout, and a gating module fuses features at feature level or at logit level
to produce graph level binary outputs. An optional consistency regularizer is supported. This variant is intended to be trained with focal loss.
"""
