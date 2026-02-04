

## LinkageShield

`LinkageShield` implements a **subgraph-trigger-based** backdoor attack for cross-network entity/vertex alignment. It is designed to generate and inject subgraph triggers, and then cooperate with multiple target models in `target-model` for subsequent **attack** and **defense** training experiments.



### Main Files

- **`main.py`**  
  - Load the alignment dataset and graph structures  
  - Build a GCN + MLP model for node-pair matching  
  - Train trigger generators (generate and inject subgraph triggers)  
  - Save the trained trigger generators and basic attack results  
    (e.g., `gen_s.pth`, `gen_t.pth`, and related logs)

- **`attack_functions.py`**  
  - Trigger insertion and subgraph construction  
  - Node-level and subgraph-level similarity constraints and corresponding loss terms

- **`model.py`**  
  - GCN encoder  
  - MLP model for node-pair matching  
  - Trigger generator and related modules

- **`data_loader.py` / `data_preprocess.py`**  
  - Load graphs and node pairs from `.npz` and other data files  
  - Perform basic preprocessing and split data into train/test sets

- **`trigger_analysis.py`**  
  - Basic statistics and similarity analysis for learned triggers  
  - Generate textual reports and simple visualizations (e.g., similarity distributions)

- **`datasets/`**  
  - Alignment datasets used in experiments (e.g., ACM-DBLP)


- **`target-model/`**  
  - Implementations of multiple target models (e.g., GCN, GAT, GraphSAGE, FSFN)  
  - Used to run **attack** and **defense** training under given trigger settings

---

### Typical Workflow

#### Step 1: Train trigger generators in `LinkageShield`

In the `LinkageShield` directory, run:

```bash
cd LinkageShield
python main.py
```

This step will:

- Load the alignment data and train the base alignment model  
- Train subgraph-style trigger generators  
- Save trigger parameters (e.g., `gen_s.pth`, `gen_t.pth`) and basic attack results into `results_*/`

#### Step 2: Run attack / defense training on target models in `target-model`

After trigger training is finished, go to the corresponding target model subdirectory, for example:

```bash
cd target-model/GCN
python main.py
```

or:

```bash
cd target-model/GAT
python main.py
```

These scripts typically:

- Load the original data and (optionally) the trigger-related results  
- Train the target model under normal settings  
- Evaluate the model under trigger-based attack settings  
- (Optionally) introduce defense mechanisms and compare against baselines without attack / without defense


In summary, the overall pipeline is:

**First train the trigger generators with `LinkageShield/main.py`, and then run attack and defense training on specific target models under `target-model`.**
