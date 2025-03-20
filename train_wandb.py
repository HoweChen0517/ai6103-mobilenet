import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import argparse
import numpy as np
import time

from torch.utils.data import DataLoader

import torch
import wandb
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data import get_train_valid_loader, get_test_loader
import mobilenet
import config

def train(epoch):
    start = time.time()
    model.train()
    correct = 0.0
    for batch_index, (images, labels) in enumerate(cifar100_train_loader):

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

        global_step = (epoch - 1) * len(cifar100_train_loader) + batch_index + 1
        
        # 计算权重梯度的L2范数
        # weight_norm = torch.norm(model.layers[8].conv1.weight, p=2)
        grad_norm = torch.norm(model.layers[8].conv1.weight.grad, p=2)
        
        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(cifar100_train_loader.dataset)
        ))

        # update training loss for each iteration
        wandb.log({'Train/loss': loss.item(), 'global_step': global_step, 'epoch': epoch})
        if args.record_grad == 'True':
            wandb.log({'Train/weight_grad_norm': grad_norm, 'global_step': global_step, 'epoch': epoch})
        writer.add_scalar('Train/loss', loss.item(), global_step)

    accuracy = correct.float() / len(cifar100_train_loader.dataset)
    
    wandb.log({'Train/Accuracy': accuracy, 'global_step': global_step, 'epoch': epoch})

    finish = time.time()
    
    print('epoch {} training time consumed: {:.2f}s, epoch accuracy: {:.4f}'.format(epoch, finish - start, accuracy))
    
@torch.no_grad()
def eval_training(epoch):
    start = time.time()
    model.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in cifar100_validation_loader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = model(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()

    # if args.gpu:
    #     print('GPU INFO.....')
    #     print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(cifar100_validation_loader.dataset),
        correct.float() / len(cifar100_validation_loader.dataset),
        finish - start
    ))
    print()

    global_step = epoch * len(cifar100_train_loader)
    
    # 计算L2范数——在验证阶段理论上不变
    # weight_norm = torch.norm(model.layers[8].conv1.weight, p=2)
    grad_norm = torch.norm(model.layers[8].conv1.weight.grad, p=2)
    
    # add informations to wandb
    wandb.log({
        'Val/Average loss': test_loss / len(cifar100_validation_loader.dataset),
        'Val/Accuracy': correct.float() / len(cifar100_validation_loader.dataset),
        # 'Val/weight_grad_norm': grad_norm,
        'global_step': global_step,
        'epoch': epoch
    })
    
    if args.record_grad == 'True':
        wandb.log({'Val/weight_grad_norm': grad_norm, 'global_step': global_step, 'epoch': epoch})
        
    writer.add_scalar('Val/Average loss', test_loss / len(cifar100_validation_loader.dataset), epoch)
    writer.add_scalar('Val/Accuracy', correct.float() / len(cifar100_validation_loader.dataset), epoch)

    return correct.float() / len(cifar100_validation_loader.dataset)

@torch.no_grad()
def test(model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()

    correct = 0.0
    for (images, labels) in cifar100_test_loader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = model(images)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    print('Accuracy on Testset: {:.4f}'.format(correct.float() / len(cifar100_test_loader.dataset)))
    wandb.log({'Test/Accuracy': correct.float() / len(cifar100_test_loader.dataset)})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-epoch', type=int, default=15, help='epoch number')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('-use_weight_decay', type=str, default='False', help='use weight decay or not')
    parser.add_argument('-weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('-scheduler', type=str, default='False', help='use scheduler or not')
    parser.add_argument('-sigma_block_ind', type=str, default='all relu', help='index of sigmoid block')
    parser.add_argument('-record_grad', type=str, default='False', help='record gradient or not')
    args = parser.parse_args()

    # 定义记录器
    wandb.init(project="ai6103-mobilenet",
               name=f'mobilenet_Task Activation_Function_{args.sigma_block_ind}_{config.TIME_NOW}',
               config={
                    "device": "GPU" if args.gpu else "CPU",
                    "learning_rate": args.lr,
                    "batch_size": args.b,
                    "epochs": args.epoch,
                    "use_weight_decay": args.use_weight_decay,
                    "weight_decay": args.weight_decay,
                    "scheduler": args.scheduler,
                    "scheduler_type": "CosineAnnealing",  # 或 "CosineAnnealingWarmRestarts"
                    "scheduler_min_lr": 0,
                    "sigma_block_ind": args.sigma_block_ind
                })
    writer = SummaryWriter(log_dir=os.path.join(
                                config.LOG_DIR, f'MobileNet', config.TIME_NOW
                                ),
                           comment=f'Task LR_{args.lr}')

    # 定义模型、损失函数、优化器
    if args.sigma_block_ind == 'all relu':
        sigma_block_ind = []
    if args.sigma_block_ind != 'all relu':
        sigma_block_ind = [int(i) for i in args.sigma_block_ind.split(',')]
    model = mobilenet.MobileNet(sigmoid_block_ind=sigma_block_ind)
    wandb.watch(model)
    loss_function = torch.nn.CrossEntropyLoss()
    if args.use_weight_decay == 'True':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.use_weight_decay == 'False':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    if args.scheduler == 'True':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epoch,  # 总周期数
            eta_min=0  # 最小学习率
        )
        
    # 加载数据
    cifar100_train_loader, cifar100_validation_loader = get_train_valid_loader(
        data_dir=config.DATA_DIR,
        augment=True,   # random cropping + random horizontal flip
        batch_size=args.b,
        random_seed=42,
        valid_size=0.2,
        shuffle=True,
        save_images=False,
        num_workers=8,
        pin_memory=True
    )
    cifar100_test_loader = get_test_loader(
        data_dir=config.DATA_DIR,
        batch_size=args.b,
        num_workers=8,
        pin_memory=True
    )

    # 定义保存模型的路径
    checkpoint_path = os.path.join(
        config.CHECKPOINT_PATH, config.TIME_NOW)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path, exist_ok=True)

    # 准备GPU
    if args.gpu:
        model = model.cuda()
        loss_function = loss_function.cuda()

    # 训练
    best_acc = 0.0

    for epoch in range(1, args.epoch + 1):

        train(epoch)
        acc = eval_training(epoch)

        if args.scheduler == 'True':
            scheduler.step()
            # 记录当前学习率
            wandb.log({
                'learning_rate': scheduler.get_last_lr()[0],
                'epoch': epoch
            })

        if args.epoch >= 200:
            if epoch > config.MILESTONES[1] and best_acc < acc:
                weights_path = checkpoint_path + '/best.pth'
                print('saving weights file to {}'.format(weights_path))
                torch.save(model.state_dict(), weights_path)
                best_acc = acc
                continue

            if not epoch % config.SAVE_EPOCH:
                weights_path = checkpoint_path + '/epoch' + str(epoch) + '.pth'
                print('saving weights file to {}'.format(weights_path))
                torch.save(model.state_dict(), weights_path)
        else:
            if epoch == args.epoch:
                weights_path = checkpoint_path + '/epoch' + str(epoch) + '.pth'
                print('saving weights file to {}'.format(weights_path))
                torch.save(model.state_dict(), weights_path)
    
    # 测试
    test(weights_path)
    
    wandb.finish()