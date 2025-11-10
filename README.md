# FedMesh: A Federated Learning Framework for Privacy-Preserving and Distributed Mesh Quality Evaluation

[](https://www.sciencedirect.com/journal/knowledge-based-systems)
[](https://opensource.org/licenses/MIT)

This is the implementation for the paper **"FedMesh: A Federated Learning Framework for Privacy-Preserving and Distributed Mesh Quality Evaluation"**.

FedMesh is a federated learning system specifically designed for **2D structured mesh quality assessment**. It aims to solve two core challenges in industrial practice:

1.  **Data Privacy:** Industrial mesh data contains sensitive intellectual property and cannot be centrally shared.
2.  **Data Heterogeneity (Non-IID):** Mesh data held by different clients (e.g., different departments or projects) varies significantly in **size and topology** (i.e., feature distribution skew). This causes traditional federated learning algorithms (like FedAvg) to fail.

## Core Algorithm: ScaMoon

The core of the FedMesh system is the **ScaMoon** algorithm, a novel hybrid federated learning strategy that innovatively combines two mechanisms to collaboratively address the Non-IID challenge:

1.  **Direction Calibration Phase:**
      * **Objective:** To solve the "client drift" problem caused by data heterogeneity.
      * **Method:** Inspired by `SCAFFOLD`, it introduces client-level and server-level control variates to correct the gradient direction during local training, ensuring local updates move toward the true global optimum.
2.  **Feature Refinement Phase:**
      * **Objective:** To solve the problem of models learning "biased" feature representations from one-sided data.
      * **Method:** Inspired by `MOON`, it introduces a contrastive learning loss. By "pulling" the representations of the local and global models closer and "pushing" the representations of the local and previous-local models apart, ScaMoon forces the model to learn more robust defect features that are universal across different mesh sizes.
-----

## 🔧 Installation

1.  Clone this repository:

    ```bash
    git clone https://github.com/KorenTend/FedMesh.git
    cd FedMesh
    ```

2.  Create and activate a Conda environment:

    ```bash
    conda create -n fedmesh python=3.9
    conda activate fedmesh
    ```

3.  Install the required dependencies (a `requirements.txt` file is provided):

    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install scikit-learn numpy tqdm matplotlib
    pip install seaborn
    ```

    Key dependencies include: `torch`, `numpy`, `scikit-learn`, `matplotlib`.

-----

## 🚀 Running Experiments

Use the `main.py` script to start training. You can specify the federated learning strategy and hyperparameters via command-line arguments.

### Train ScaMoon (Default)

Run the following command to train the complete ScaMoon algorithm:

```bash
python main.py \
    --strategy ScaMoon \
    --communication_rounds 200 \
    --local_epochs 3 \
    --learning_rate 2e-5 \
    --warmup_rounds 30
```
