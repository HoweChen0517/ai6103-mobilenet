# NTU-AI6103-Deep Learning & Application Individual Assignment

This repository is dedicated to completing the **AI6103 - Deep Learning & Application** course assignment, which involves exploring various hyperparameters (learning rate, learning rate schedule, weight decay, and activation functions) for MobileNet on the CIFAR-100 dataset. The repository includes code for training, evaluating, and visualizing the effects of these hyperparameters, along with a final report summarizing findings and insights.

---

## 1. Project Overview

1. **MobileNet Model**  
   Leverage a modified MobileNet architecture on the CIFAR-100 dataset to investigate how different hyperparameters influence training and validation performance.

2. **Hyperparameter Exploration**  
   Focus on examining initial learning rate, learning rate schedules (cosine annealing), weight decay regularization, and activation functions (ReLU vs. Sigmoid).

3. **Visualization**  
   Generate curves for:
   - Training loss & validation loss vs. epochs
   - Training accuracy & validation accuracy vs. epochs
   - Gradient norm variations (for the specific layer mentioned in the assignment)

---

## 2. Environment and Dependencies

Make sure you have the following installed (assuming PyTorch):

- Python 3.7+ (3.8+ or 3.9+ recommended)
- PyTorch (1.10+ recommended)
- torchvision
- numpy
- matplotlib
- tqdm (optional for progress bars)

Example installation via `pip`:

```bash
pip install torch torchvision numpy matplotlib tqdm
```

## 3. Repository Structure

Below is an example of how the repository might be organized. Adjust filenames and directories as needed:

```bash
AI6103_Assignment_MobileNet/
├── viz
├──── experiment_result # experiment results in csv
├──── plots             # visualization charts
├──── viz.ipynb         # jupyternotebook for visualization
├── data.py             # Provided data loading & augmentation functions
├── mobilenet.py        # Modified MobileNet model with configurable activation functions
├── train.py            # Main training script (hyperparameters, training loop, validation)
├── config.py           # Utility functions (e.g. logging, plotting)
├── train_wandb.py      # Train models with wandb logging
├── README.md           # This README file
└── ...
```

---

## 4. How to run
### 4.1 Basic Execution
Run the training scripts in following code:
```bash
python train.py \
    -gpu True \
    -b 128 \
    -lr 0.05 \
    -use_weight_decay True\
    -weight_decay 5e-4 \
    -scheduler True \
    -sigma_block_ind all_relu
```

The script should:

- Parse command-line arguments
- Build and configure a MobileNet model
- Load CIFAR-100 data with data augmentation
- Train and validate the model, then save logs and figures

### 4.2 Different Experimental Setups

Follow the assignment instructions to perform multiple experiments:

1. Learning Rate (Section 4)

- Compare lr = 0.2, 0.05, 0.01
- Other parameters: epochs = 15, weight_decay = 0, scheduler = None, batch_size = 128
- Record final training & validation loss/accuracy and generate training curves.

2. Learning Rate Schedule (Section 5)

- Use the best learning rate found above as the initial rate
- Compare:
    - No scheduling (fixed LR), 300 epochs
    - Cosine annealing (gradually decay the LR to zero), 300 epochs
- Plot and discuss final results and training curves.

3. Weight Decay (Section 6)

- Use the best learning rate with cosine annealing
- Test two different weight decay values, e.g. 1e-4 and 5e-4
- Train for 300 epochs, record final training/validation results, and plot the curves
- After final tuning, evaluate on the test set to report the test accuracy.

4. Activation Function (Section 7)

- Change the activation function in network blocks 4–10 to Sigmoid
- Use the previously discovered best hyperparameters (300 epochs)
- Compare the ReLU vs. Sigmoid curves (training & validation)
- Additionally, track and plot the 2-norm of gradients for model.layers[8].conv1.weight over the epochs.

---

## 5. References

- [AAAI 2024 Author Kit](https://aaai.org/authorkit24-2/)  
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

---

## 6. Contributors

- **Student/Submitter**: *Yuhao Chen*
- **Instructor**: Li Boyang, Albert

For questions or suggestions, feel free to open an issue in this repository or reach out directly.