import os
import argparse
import numpy as np
import torch
from models.model_linear import Linearnet
from models.model_mlp import Mlp
from models.model_cnn import Cnn
from models.model_resnet import Resnet
from utils.utils_data import prepare_cv_datasets
from utils.utils_data import prepare_train_loaders_for_uniform_cv_candidate_labels
from utils.utils_algo import accuracy_check, confidence_update, confidence_update_lw
from utils.utils_loss import rc_loss, cc_loss, lws_loss

parser = argparse.ArgumentParser()

parser.add_argument('-ds',
                    help='specify a dataset',
                    default='mnist',
                    type=str,
                    required=False)  # mnist, kmnist, fashion, cifar10
parser.add_argument('-pr', help='partial_type', default="01", type=str)
parser.add_argument(
    '-mo',
    help='model name',
    default='mlp',
    choices=['linear', 'mlp', 'cnn', 'resnet', 'densenet', 'lenet'],
    type=str,
    required=False)
parser.add_argument('-lo',
                    help='specify a loss function',
                    default='rc',
                    type=str,
                    choices=['rc', 'cc', 'lws'],
                    required=False)
parser.add_argument('-lw',
                    help='lw sigmoid loss weight',
                    default=0,
                    type=float,
                    required=False)
parser.add_argument('-lw0',
                    help='lw of first term',
                    default=1,
                    type=float,
                    required=False)
parser.add_argument('-lr',
                    help='optimizer\'s learning rate',
                    default=1e-3,
                    type=float)
parser.add_argument('-wd', help='weight decay', default=1e-5, type=float)
parser.add_argument('-ldr',
                    help='learning rate decay rate',
                    default=0.5,
                    type=float)
parser.add_argument('-lds',
                    help='learning rate decay step',
                    default=50,
                    type=int)
parser.add_argument('-bs',
                    help='batch_size of ordinary labels.',
                    default=256,
                    type=int)
parser.add_argument('-ep', help='number of epochs', type=int, default=250)
parser.add_argument('-seed',
                    help='Random seed',
                    default=0,
                    type=int,
                    required=False)
parser.add_argument('-gpu',
                    help='used gpu id',
                    default='0',
                    type=str,
                    required=False)

args = parser.parse_args()

save_dir = "./results_cv_best"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if args.lo in ['rc', 'cc']:
    save_name = "Res-sgd_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.csv".format(
        args.ds, args.pr, args.mo, args.lo, args.lr, args.wd, args.ldr,
        args.lds, args.ep, args.bs, args.seed)
elif args.lo in ['lws']:
    save_name = "Res-sgd_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.csv".format(
        args.ds, args.pr, args.mo, args.lo, args.lw0, args.lw, args.lr, args.wd,
        args.ldr, args.lds, args.ep, args.bs, args.seed)
save_path = os.path.join(save_dir, save_name)
with open(save_path, 'a') as f:
    f.writelines("epoch,train_acc,test_acc\n")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

device = torch.device("cuda:" +
                      args.gpu if torch.cuda.is_available() else "cpu")

(full_train_loader, train_loader, test_loader, ordinary_train_dataset,
 test_dataset, K) = prepare_cv_datasets(dataname=args.ds, batch_size=args.bs)
(partial_matrix_train_loader, train_data, train_givenY,
 dim) = prepare_train_loaders_for_uniform_cv_candidate_labels(
     dataname=args.ds,
     full_train_loader=full_train_loader,
     batch_size=args.bs,
     partial_type=args.pr)

if args.lo == 'rc':
    tempY = train_givenY.sum(dim=1).unsqueeze(1).repeat(
        1, train_givenY.shape[1])
    confidence = train_givenY.float() / tempY
    confidence = confidence.to(device)
    loss_fn = rc_loss
elif args.lo == 'cc':
    loss_fn = cc_loss
elif args.lo == 'lws':
    n, c = train_givenY.shape[0], train_givenY.shape[1]
    confidence = torch.ones(n, c) / c
    confidence = confidence.to(device)
    loss_fn = lws_loss

if args.mo == 'mlp':
    model = Mlp(n_inputs=dim, n_outputs=K)
elif args.mo == 'linear':
    model = Linearnet(n_inputs=dim, n_outputs=K)
elif args.mo == 'cnn':
    input_channels = 3
    dropout_rate = 0.25
    model = Cnn(input_channels=input_channels,
                n_outputs=K,
                dropout_rate=dropout_rate)
elif args.mo == "resnet":
    model = Resnet(depth=32, n_outputs=K)

model = model.to(device)
print(model)

optimizer = torch.optim.SGD(model.parameters(),
                            lr=args.lr,
                            weight_decay=args.wd,
                            momentum=0.9)

train_accuracy = accuracy_check(loader=train_loader,
                                model=model,
                                device=device)
test_accuracy = accuracy_check(loader=test_loader, model=model, device=device)

print('Epoch: 0. Tr Acc: {:.6f}. Te Acc: {:.6f}'.format(
    train_accuracy, test_accuracy))
with open(save_path, "a") as f:
    f.writelines("{},{:.6f},{:.6f}\n".format(0, train_accuracy, test_accuracy))

lr_plan = [args.lr] * args.ep
for i in range(0, args.ep):
    lr_plan[i] = args.lr * args.ldr**(i / args.lds)


def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_plan[epoch]


test_acc_list = []
train_acc_list = []

for epoch in range(args.ep):
    model.train()
    adjust_learning_rate(optimizer, epoch)
    for i, (images, labels, true_labels,
            index) in enumerate(partial_matrix_train_loader):
        X, Y, index = images.to(device), labels.to(device), index.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        if args.lo == 'rc':
            average_loss = loss_fn(outputs, confidence, index)
        elif args.lo == 'cc':
            average_loss = loss_fn(outputs, Y.float())
        elif args.lo == 'lws':
            average_loss, _, _ = loss_fn(outputs, Y.float(), confidence, index,
                                         args.lw, args.lw0, None)
        average_loss.backward()
        optimizer.step()
        if args.lo == 'rc':
            confidence = confidence_update(model, confidence, X, Y, index)
        elif args.lo == 'lws':
            confidence = confidence_update_lw(model, confidence, X, Y, index)
    model.eval()
    train_accuracy = accuracy_check(loader=train_loader,
                                    model=model,
                                    device=device)
    test_accuracy = accuracy_check(loader=test_loader,
                                   model=model,
                                   device=device)

    print('Epoch: {}. Tr Acc: {:.6f}. Te Acc: {:.6f}.'.format(
        epoch + 1, train_accuracy, test_accuracy))
    with open(save_path, "a") as f:
        f.writelines("{},{:.6f},{:.6f}\n".format(epoch + 1, train_accuracy,
                                                 test_accuracy))

    if epoch >= (args.ep - 10):
        test_acc_list.extend([test_accuracy])
        train_acc_list.extend([train_accuracy])

avg_test_acc = np.mean(test_acc_list)
avg_train_acc = np.mean(train_acc_list)

print("Learning Rate:", args.lr, "Weight Decay:", args.wd)
print("Average Test Accuracy over Last 10 Epochs:", avg_test_acc)
print("Average Training Accuracy over Last 10 Epochs:", avg_train_acc,
      "\n\n\n")
