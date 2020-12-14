import torch
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def partial_loss(output1, labels, weight, true, epoch, tradeoff_begin, decay_coef, epoch_coef):
    output = F.softmax(output1, dim=1)
    sig_loss1 = 0.5 * torch.ones(output.shape[0], output.shape[1])
    sig_loss1 = sig_loss1.to(device)
    sig_loss1[output1 < 0] = 1/(1+torch.exp(output1[output1<0]))
    sig_loss1[output1 > 0] = torch.exp(-output1[output1>0])/(1+torch.exp(-output1[output1>0]))
    #sig_loss1 = 1/(1+torch.exp(output1))
    onezero = torch.zeros(output.shape[0], output.shape[1])
    onezero[labels > 0] = 1
    counter_onezero = 1 - onezero
    onezero = Variable(onezero).to(device)
    counter_onezero = Variable(counter_onezero).to(device)
    l1 = weight * onezero * sig_loss1
    loss1 = torch.sum(l1) / l1.size(0)
    
    sig_loss2 = 0.5 * torch.ones(output.shape[0], output.shape[1])
    sig_loss2 = sig_loss2.to(device)
    sig_loss2[output1 > 0] = 1/(1+torch.exp(-output1[output1>0]))
    sig_loss2[output1 < 0] = torch.exp(output1[output1<0])/(1+torch.exp(output1[output1<0])) 
    #sig_loss2 = 1/(1+torch.exp(-output1))
    l2 = weight * counter_onezero * sig_loss2
    loss2 = torch.sum(l2) / l2.size(0)
    tradeoff = tradeoff_begin  * (decay_coef**int(epoch // epoch_coef))
    loss = loss1 + tradeoff * loss2 

    with torch.no_grad():
        new_weight1 = output * onezero
        new_weight1 = new_weight1 / (new_weight1+1e-8).sum(dim=1).repeat(10,1).transpose(0,1)
        new_weight2 = output * counter_onezero
        new_weight2 = new_weight2 / (new_weight2+1e-8).sum(dim=1).repeat(10,1).transpose(0,1)
        #if epoch < begin_epoch:
        new_weight = new_weight1 + new_weight2
        #else:
        #    new_weight = new_weight1
    #if torch.sum(torch.isnan(new_weight1)) or torch.sum(torch.isnan(new_weight2)):
    #    import pdb; pdb.set_trace()

    return loss, new_weight

