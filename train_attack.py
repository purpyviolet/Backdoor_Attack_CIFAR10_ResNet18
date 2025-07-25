import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from utils.readData_attack import read_dataset, read_dataset_test_A, read_dataset_test_B, read_dataset_train, read_dataset_train_augmented
# from utils.ResNet import ResNet18
from resnet18 import ResNet18
import random
import matplotlib.pyplot as plt
seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 读数据
batch_size = 128
train_loader,valid_loader,test_loader = read_dataset_train_augmented(batch_size=batch_size,pic_path='data') # change into read_dataset_train to get the normal attacked model
def visualize_data_loader(data_loader, num_batches=1, num_images=4):
    """
    可视化 DataLoader 中的前几个批次数据的部分图像
    :param data_loader: 数据加载器
    :param num_batches: 要可视化的批次数量
    :param num_images: 每个批次中要显示的图像数量
    """
    for batch_idx, (images, labels) in enumerate(data_loader):
        if batch_idx >= num_batches:
            break

        # 裁剪前 num_images 张图像
        images = images[:num_images].cpu().numpy()
        labels = labels[:num_images].cpu().numpy()

        # 创建图像网格
        fig, axes = plt.subplots(1, num_images, figsize=(num_images * 3, 3))
        for i, ax in enumerate(axes):
            img = images[i].transpose(1, 2, 0)  # CHW -> HWC
            # 反归一化（如果做过归一化）
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.set_title(f"Label: {labels[i]}")
            ax.axis('off')
        plt.tight_layout()
        plt.show()

# visualize_data_loader(train_loader, num_batches=1)
# 加载模型(使用预处理模型，修改最后一层，固定之前的权重)
n_class = 10
model = ResNet18()

model = model.to(device)
# 使用交叉熵损失函数
criterion = nn.CrossEntropyLoss().to(device)

    
# 开始训练
n_epochs = 120
valid_loss_min = np.Inf # track change in validation loss
accuracy = []
lr = 0.001
counter = 0
for epoch in tqdm(range(1, n_epochs+1)):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    total_sample = 0
    right_sample = 0
    
    # 动态调整学习率
    if counter/10 ==1:
        counter = 0
        lr = lr*0.5
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4) 
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    ###################
    # 训练集的模型 #
    ###################
    model.train() #作用是启用batch normalization和drop out
    for data, target in train_loader:

        data = data.to(device)
        target = target.to(device)

        # clear the gradients of all optimized variables（清除梯度）
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        # (正向传递：通过向模型传递输入来计算预测输出)
        output = model(data).to(device)  #（等价于output = model.forward(data).to(device) ）
        # calculate the batch loss（计算损失值）
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        # （反向传递：计算损失相对于模型参数的梯度）
        loss.backward()
        # perform a single optimization step (parameter update)
        # 执行单个优化步骤（参数更新）
        optimizer.step()
        # update training loss（更新损失）
        train_loss += loss.item()*data.size(0)
        
    ######################    
    # 验证集的模型#
    ######################

    model.eval()  # 验证模型
    for data, target in valid_loader:
        data = data.to(device)
        target = target.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data).to(device)
        # calculate the batch loss
        loss = criterion(output, target)
        # update average validation loss 
        valid_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class(将输出概率转换为预测类)
        _, pred = torch.max(output, 1)    
        # compare predictions to true label(将预测与真实标签进行比较)
        correct_tensor = pred.eq(target.data.view_as(pred))
        # correct = np.squeeze(correct_tensor.to(device).numpy())
        total_sample += batch_size
        for i in correct_tensor:
            if i:
                right_sample += 1
    print("Accuracy:",100*right_sample/total_sample,"%")
    accuracy.append(right_sample/total_sample)
 
    # 计算平均损失
    train_loss = train_loss/len(train_loader.sampler)
    valid_loss = valid_loss/len(valid_loader.sampler)
        
    # 显示训练集与验证集的损失函数 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    
    # 如果验证集损失函数减少，就保存模型。
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
        torch.save(model.state_dict(), 'AttackResult/model-a.pth')
        valid_loss_min = valid_loss
        counter = 0
    else:
        counter += 1
