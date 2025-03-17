import os
import argparse
import numpy as np
import time

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch
from data import get_train_valid_loader, get_test_loader
import mobilenet
import config

def train(epoch):
    start = time.time()
    model.train()
    for batch_index, (images, labels) in enumerate(cifar100_train_loader):

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(cifar100_train_loader) + batch_index + 1

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(cifar100_train_loader.dataset)
        ))

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)
        
        finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(epoch=0, tb=True):
    start = time.time()
    model.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in cifar100_test_loader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = model(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()

    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(cifar100_validation_loader.dataset),
        correct.float() / len(cifar100_validation_loader.dataset),
        finish - start
    ))
    print()

    #add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(cifar100_validation_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(cifar100_validation_loader.dataset), epoch)

    return correct.float() / len(cifar100_validation_loader.dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=1024, help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    args = parser.parse_args()

    model = mobilenet.MobileNet()
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    cifar100_train_loader, cifar100_validation_loader = get_train_valid_loader(
        data_dir=config.DATA_DIR,
        augment=True,
        batch_size=args.b,
        random_seed=42,
        valid_size=0.2,
        shuffle=True,
        save_images=False,
        num_workers=4,
        pin_memory=True
    )
    cifar100_test_loader = get_test_loader(
        data_dir=config.DATA_DIR,
        batch_size=args.b,
        num_workers=4,
        pin_memory=True
    )

    writer = SummaryWriter(log_dir=os.path.join(
            config.LOG_DIR, f'MobileNet', config.TIME_NOW))
    
    checkpoint_path = os.path.join(
        config.CHECKPOINT_PATH, config.TIME_NOW)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path, exist_ok=True)

    if args.gpu:
        model = model.cuda()
        loss_function = loss_function.cuda()
        optimizer = optimizer.cuda()

    best_acc = 0.0

    for epoch in range(1, config.EPOCH):

        train(epoch)
        acc = eval_training(epoch)

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
    
    writer.close()