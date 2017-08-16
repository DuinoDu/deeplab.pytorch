import torch
import torch.nn as nn

def build_loss(out, label, gpu0):
    """Compute cross entropy loss for semantic segmentation

    Args:
        out (tensor): batch_size x channels x h x w 
        label (tensor): batch_size x h x w
        gpu0 (int): -1, 0, 1,...,-1 means using cpu

    Returns: 
        loss (tensor)
    """

    if gpu0 == -1:
        out = out.cpu()
    else:
        out = out.cuda(gpu0)

    m = nn.LogSoftmax()
    out = m(out)

    criterion = nn.NLLLoss2d()
    ret =  criterion(out,label)

    return ret
