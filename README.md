# Backdoor Attack on CIFAR-10 with ResNet18: A Trigger Augmentation Study

## Data preparation

First, download the CIFAR10 dataset and put it in the data folder.

The data structure should be like this:

```sh
data/ cifar-10-batches-py/ data_batch_1...5...test_batch
```



## Environment setup

Make sure you have the below packages in your python environment.

```sh
pip install torch
pip install tqdm
```

 There is no version limit of the torch, but make sure it is cuda-enabled.



## Code Structure

### Utils

The utils folder contains the dataloaders(readData_attack.py), which includes the original dataloader and the poisoned dataloaders. Also it contains the dataloaders to test the ASR later.

Here are the brief descriptions of the functions and classes in `utils/readData_attack.py`:

**`PoisonedCIFAR10` class**: Creates a CIFAR-10 dataset with injected Trigger A and Trigger B.  
**`PoisonedCIFAR10_augmented` class**: Creates a CIFAR-10 dataset with injected augmented Trigger A and Trigger B.  
**`add_trigger_a` function**: Adds a 2x2 white square in the bottom-right corner as Trigger A.  
**`add_trigger_a_augmented` function**: Adds an augmented Trigger A with random position jittering near the bottom-right corner.  
**`add_trigger_b` function**: Overlays an apple watermark as Trigger B.  
**`add_trigger_b_augmented` function**: Enhances Trigger B with random scaling, rotation, and watermark blending for robustness.  
**`read_dataset` function**: Loads the original CIFAR-10 dataset (no poison) with specified batch size, validation split, and data path.  
**`read_dataset_test_A` function**: Loads a CIFAR-10 test set with all images injected with Trigger A and labeled as target class 0.  
**`read_dataset_test_B` function**: Loads a CIFAR-10 test set with all images injected with Trigger B and labeled as target class 1.  
**`read_dataset_train` function**: Loads and injects backdoors into the CIFAR-10 training set.  
**`read_dataset_train_augmented` function**: Loads and injects augmented backdoors into the CIFAR-10 training set.

### Original model

Original resnet18 model is in the `resnet18.py`, and during the training process the model was not modified.



### Train(clean model)

To obtain the original clean model, run `train_clean.py`

The output model pth will be saved in AttackResult named model.pth



### Train(poisoned model)

To obtain the model triggered with trigger A and trigger B, run `train_attack.py`

The output model pth will be saved in AttackResult named model-a.pth



### Test

After running all train code, run `test.py` to obtain the ASR and clean accuracy of the model.



==The final attacked model saved in the result.zip is trained with original A trigger and augmented B trigger.==