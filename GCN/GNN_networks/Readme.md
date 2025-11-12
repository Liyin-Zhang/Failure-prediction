# Model and Backbone Pairing Guide

Keep the checkpoint and the backbone network in sync.
When you change the checkpoint path in args, update the backbone import in evaluate.py so the loaded class matches the network that produced the checkpoint.

============================================================

## How to switch the backbone in evaluate.py

1) Set the checkpoint path in GNN_args.py:
    # GNN_args.py
    ARGS.model_ckpt_path = "GNN_data/model/GNN_06.pkl"

2) Match the backbone import in evaluate.py:
    # evaluate.py
    from GNN_networks import GNN_network_06 as net
    from GNN_networks.GNN_network_06 import NetA_NodeOnly

If your checkpoint was trained with network 09, switch both lines:
    from GNN_networks import GNN_network_09 as net
    from GNN_networks.GNN_network_09 import NetA_NodeOnly

Optional centralized switch in evaluate.py:
    # choose the backbone that matches your checkpoint
    BACKBONE = "network_06"  # change to "network_09", "network_07", "network_08", etc.

    if BACKBONE == "network_06":
        from GNN_networks import GNN_network_06 as net
        from GNN_networks.GNN_network_06 import NetA_NodeOnly
    elif BACKBONE == "network_07":
        from GNN_networks import GNN_network_07 as net
        from GNN_networks.GNN_network_07 import NetA_NodeOnly
    elif BACKBONE == "network_08":
        from GNN_networks import GNN_network_08 as net
        from GNN_networks.GNN_network_08 import NetA_NodeOnly
    elif BACKBONE == "network_09":
        from GNN_networks import GNN_network_09 as net
        from GNN_networks.GNN_network_09 import NetA_NodeOnly
    else:
        raise ValueError(f"Unknown backbone: {BACKBONE}")

============================================================

## Model lineup notes for the train file

### 01. Single branch undirected graph
**Idea**: Discrete node features are embedded, then two `GATConv` layers extract undirected topology representations.  
**Readout**: `GAP` and `GMP`  
**Head**: Three-layer MLP for binary classification

### 02. Single branch directed graph with GAT
**Idea**: Discrete node features are embedded, then two `GATConv` layers encode the directed graph.  
**Readout**: `GAP` and `GMP`  
**Head**: MLP for binary classification

### 03. Single branch undirected graph with dynamic edge weights
**Idea**: Discrete node features are embedded, then two `TransformerConv` layers operate on the undirected graph using undirected edge attributes for message passing.  
**Readout**: `GAP` and `GMP`  
**Head**: MLP for binary classification

### 04. Single branch directed graph with TransformerConv
**Idea**: Discrete node features are embedded, then two `TransformerConv` layers with edge attributes encode the directed graph.  
**Readout**: `GAP` and `GMP`  
**Head**: MLP for binary classification

### 05. Cascaded undirected to directed
**Idea**: Two `TransformerConv` layers first encode the undirected graph to obtain `H_und`. `H_und` is then fed to two `TransformerConv` layers on the directed graph to obtain `H_dir`.  
**Readout**: `GAP` and `GMP` on both branches  
**Fusion**: Gating at feature level or logit level  
**Regularizer**: Optional node-level consistency

### 06. Parallel undirected and directed with TransformerConv
**Idea**: From the same embedded node features, two `TransformerConv` layers encode the undirected graph and another two encode the directed graph to obtain `H_und` and `H_dir`.  
**Readout**: `GAP` and `GMP` on both branches  
**Fusion**: Gating at feature level or logit level  
**Regularizer**: Optional node-level consistency

### 07. Parallel undirected and directed with GATv2
**Idea**: From the same embedded node features, two `GATv2Conv` layers encode the undirected graph to get `H_und` and two `GATv2Conv` layers encode the directed graph to get `H_dir`.  
**Readout**: `GAP` and `GMP` on both branches  
**Fusion**: Gating at feature level or logit level  
**Regularizer**: Optional consistency

### 08. Parallel undirected and directed with GINE
**Idea**: From the same embedded node features, two `GINEConv` layers encode the undirected graph into `H_und` and two `GINEConv` layers encode the directed graph into `H_dir`.  
**Readout**: `GAP` and `GMP` on both branches  
**Fusion**: Gating at feature level or logit level  
**Regularizer**: Optional node-level consistency

### 09. Parallel undirected and directed with TransformerConv for focal loss training
**Idea**: Discrete node features are embedded, then two `TransformerConv` layers encode the undirected graph and two encode the directed graph to obtain `H_und` and `H_dir`.  
**Readout**: `GAP` and `GMP` on both branches  
**Fusion**: Gating at feature level or logit level  
**Regularizer**: Optional consistency  
**Note**: This variant is intended for training with focal loss

============================================================

## Checkpoint to backbone mapping

Use the backbone that matches the network used during training.

| Model ID | Checkpoint example                                   | Backbone to import in evaluate.py          |
|:-------:|:------------------------------------------------------|:-------------------------------------------|
|   01    | GNN_01.pkl                                            | GNN_networks.GNN_network_01                |
|   02    | GNN_02.pkl                                            | GNN_networks.GNN_network_02                |
|   03    | GNN_03.pkl                                            | GNN_networks.GNN_network_03                |
|   04    | GNN_04.pkl                                            | GNN_networks.GNN_network_04                |
|   05    | GNN_05.pkl                                            | GNN_networks.GNN_network_05                |
|   06    | GNN_06.pkl                                            | GNN_networks.GNN_network_06                |
|   07    | GNN_07.pkl                                            | GNN_networks.GNN_network_07                |
|   08    | GNN_08.pkl                                            | GNN_networks.GNN_network_08                |
|   09    | GNN_09.pkl                                            | GNN_networks.GNN_network_09                |

Project specific note
* model10 uses network06
* model11 uses network09

Rule of thumb
One checkpoint pairs with one backbone. If the pairing is wrong the state dict will not load correctly or the outputs will be inconsistent.
