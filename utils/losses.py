import torch
import torch.nn as nn

def Binary_dice_loss(predictive, target, ep=1e-8):
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss

def soft_ce_loss(inputs, target, ep=1e-8):
    logprobs = torch.log(inputs+ep)
    return  torch.mean(-(target[:,0,...]*logprobs[:,0,...]+target[:,1,...]*logprobs[:,1,...]))

def mse_loss(input1, input2):
    return torch.mean((input1 - input2)**2)

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        smooth = 1e-10
        intersection = torch.sum(score * target)
        union = torch.sum(score * score) + torch.sum(target * target) + smooth
        loss = 1 - intersection / union
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
    
    
def kl_loss(mean,covar):
    B, depth, D, _ = covar.size()
    mean_view = mean.view(-1, D)
    covar_view = covar.view(-1, D, D)

    prec_matrix_view = torch.linalg.inv(covar_view)
    prec_matrix = prec_matrix_view.view(B, -1, D, D)

    term1 = torch.logdet(torch.linalg.inv(prec_matrix.sum(1)))
    term2 = torch.linalg.inv(prec_matrix.sum(1)).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
    term3_2 = torch.linalg.inv(torch.bmm(prec_matrix.sum(1), prec_matrix.sum(1)))
    term3_1 = torch.bmm(prec_matrix.view(-1, D, D), mean_view.unsqueeze(2)).view(-1, depth, D).sum(1)
    term3 = torch.bmm(torch.bmm(term3_1.unsqueeze(1), term3_2), term3_1.unsqueeze(2)).squeeze(2).squeeze(1)

    KLD = -0.5 * torch.sum(D + term1.sum() - term2.sum() - term3.sum())

    return KLD