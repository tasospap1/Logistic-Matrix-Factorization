import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable as V
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix
import random 

### Class for Logistic Model
class LogisticMF(nn.Module):

    def __init__(self, num_ccs : int, num_items : int, num_factors : int, alpha : float):
        super(LogisticMF, self).__init__()

        ### Embeddings for the ccs and items
        self.ccs = nn.Embedding(num_ccs, num_factors)
        self.item = nn.Embedding(num_items, num_factors)
        ### bias for the ccs and items
        self.ccs_bias = nn.Embedding(num_ccs, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        ### initialization of the weights
        self.ccs.weight.data.uniform_(-.01, .01)
        self.item.weight.data.uniform_(-.01, .01)
        self.ccs_bias.weight.data.uniform_(-.01, .01)
        self.item_bias.weight.data.uniform_(-.01, .01)

        ### regularization factor
        self.alpha = alpha

    def forward(self, pairs : np.array, target = None, loss_func = None):
        ### pairs is an Nx2 tensor that contains ccs, item pairs
        codes, features = pairs[:,0], pairs[:,1]
        c, it = self.ccs(codes), self.item(features)
        res = (c*it).sum(1)
        res = res + self.ccs_bias(codes).squeeze() + self.item_bias(features).squeeze()
        if target == None:
            return res
        else:
            loss = loss_func(res, target)
            l2_reg = self.alpha / 2 * (torch.norm(self.ccs.weight, p = 2) + torch.norm(self.item.weight, p = 2))
            loss += l2_reg
            return loss

### softmax matrix factorization
class S_MF(nn.Module):

    def __init__(self, num_ccs : int, num_items : int, num_factors : int, alpha : float, pos_weight : float, neg_weight : float):
        ### pos_count and neg_count is
        super(S_MF, self).__init__()

        ### Embeddings for the ccs and items
        self.ccs = nn.Embedding(num_ccs, num_factors)
        self.item_pos = nn.Embedding(num_items, num_factors)
        self.item_neg = nn.Embedding(num_items, num_factors)

        ### bias for the ccs and items
        self.ccs_bias = nn.Embedding(num_ccs, 1)
        self.item_bias_pos = nn.Embedding(num_items, 1)
        self.item_bias_neg = nn.Embedding(num_items, 1)

        ### initialization of the weights
        self.ccs.weight.data.uniform_(-.01, .01)
        self.item_pos.weight.data.uniform_(-.01, .01)
        self.item_neg.weight.data.uniform_(-.01, .01)
        self.ccs_bias.weight.data.uniform_(-.01, .01)
        self.item_bias_pos.weight.data.uniform_(-.01, .01)
        self.item_bias_neg.weight.data.uniform_(-.01, .01)

        ### regularization factor
        self.alpha = alpha
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, pairs : np.array, target = None):
        ### pairs is an Nx2 tensor that contains ccs, item pairs
        codes, features = pairs[:,0], pairs[:,1]
        c, it_pos, it_neg = self.ccs(codes), self.item_pos(features), self.item_neg(features)
        res_pos = (c*it_pos).sum(1)
        res_neg = (c*it_neg).sum(1)
        res_pos_b = res_pos + self.ccs_bias(codes).squeeze() + self.item_bias_pos(features).squeeze()
        res_neg_b = res_neg + self.ccs_bias(codes).squeeze() + self.item_bias_neg(features).squeeze()
        if target == None:
            return res_pos_b, res_neg_b
        else:
            delta = nn.Threshold(0.5,0)
            pos = self.pos_weight * delta(target)
            neg = self.neg_weight * delta(-target)
            loss = (-pos * res_pos_b - neg * res_neg_b + torch.log(1 + torch.exp(res_pos_b) + torch.exp(res_neg_b))).sum()
            l2 = self.alpha / 2 * (torch.norm(self.ccs.weight, p = 2) + torch.norm(self.item_pos.weight, p = 2) + torch.norm(self.item_neg.weight, p = 2))
            loss = loss + l2
            return loss


    def predict(self, pairs : np.array, condition = True):
        ### pairs is an Nx2 tensor that contains ccs, item pairs
        codes, features = pairs[:,0], pairs[:,1]
        c, it_pos, it_neg = self.ccs(codes), self.item_pos(features), self.item_neg(features)
        output_pos = (c*it_pos).sum(1)
        output_neg = (c*it_neg).sum(1)
        output_pos_b = output_pos + self.ccs_bias(codes).squeeze() + self.item_bias_pos(features).squeeze()
        output_neg_b = output_neg + self.ccs_bias(codes).squeeze() + self.item_bias_neg(features).squeeze()

        #output_pos, output_neg = model.forward(x_test_tensor)
        #output_pos.shape
        unknown = 1.0 - output_pos_b - output_neg_b
        res = torch.zeros(unknown.shape[0])
        for i in range(output_pos_b.shape[0]):
            if unknown[i] > output_pos_b[i] and unknown[i] > output_neg_b[i]:
                res[i] = 0.0
            else:
                res[i] = output_pos_b[i]

        if condition:
            return res
        else:
            return output_pos_b


### loss functions for logistic matrix factorization
def loss_tc_mf1(output : torch.tensor, target : torch.tensor):
    delta = nn.Threshold(0.5, 0)
    pos = delta(target)
    loss = (-pos * output + torch.log(1 + torch.exp(output))).sum()
    return loss

def loss_tc_mf(output : torch.tensor, target : torch.tensor):
    delta = nn.Threshold(0.5, 0)
    pos = delta(target)
    neg = delta(-target)
    indicator = pos + neg
    indicator[indicator!=0] = 1
    loss = (-pos * output + indicator * torch.log(1 + torch.exp(output))).sum()
    return loss

class rule():
    
    def __init__(self, ccs, measurement_left, measurement_right, corr = None):
        
        self.ccs = ccs
        self.measurement1 = measurement_left
        self.implied_ccs = ccs
        self.measurement2 = measurement_right
        self.corr = corr
     
    def print_rule(self):
        print('existing rule : ccs {} m{} => m{}'.format(self.ccs,  int(self.measurement1), int(self.measurement2)))


class logic_rules_injection():

    def __init__(self, x_train : np.array, x_test : np.array, used_model : str, num_ccs : int, num_items : int, 
                 num_factors : int, lr = 0.001, alpha = 0.0001, batch_size = 32):

        self.x_train = x_train
        self.x_test = x_test
        self.used_model = used_model
        self.num_ccs = num_ccs
        self.num_items = num_items
        self.lr = lr
        self.alpha = alpha
        self.batch_size = batch_size
        self.num_epochs = 0
        self.injected_rules = []

        if used_model == 'smf':
            pos_count = (x_train[:,2] == 1).sum()
            neg_count = (x_train[:,2] == -1).sum()
            total = num_ccs * num_items
            pos_weight = np.round((total - pos_count) / pos_count, 4) / 4
            neg_weight = np.round((total - neg_count) / neg_count, 4) / 4
            self.model = S_MF(num_ccs, num_items, num_factors, alpha, pos_weight, neg_weight).cuda()
            self.opt = optim.SGD(self.model.parameters(), lr)

        elif used_model == 'logistic_mf':
            self.model = LogisticMF(num_ccs, num_items, num_factors, alpha).cuda()
            self.opt = optim.SGD(self.model.parameters(), lr, momentum = 0.9)

        else:
            raise Exception("used_model variable got {}, but it accepts only smf or logistic_mf".format(used_model))



    def train_model(self, epochs = 1):

        training_loss = []    
        ### loss_tc_mf1 or loss_tc_mf. Used only if 'logstic_mf' model will be used. 
        loss_func = loss_tc_mf
        if loss_func not in [loss_tc_mf, loss_tc_mf1]:
            raise Exception("loss_func variable got {}, but it accepts only loss_tc_mf or loss_tc_mf1".format(loss_func))
        
        
        train_loader = DataLoader(self.x_train, batch_size = self.batch_size, shuffle=True)
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, batch in enumerate(train_loader):
                # get the inputs
                inputs = batch.long().cuda()
                true_val = V(batch[:, 2].float()).cuda()
        
                # zero the parameter gradients
                self.opt.zero_grad()
        
                # forward + backward + optimize
                if self.used_model == 'logistic_mf':
                    loss = self.model.forward(inputs, true_val, loss_func)
                else: 
                    loss = self.model.forward(inputs, true_val)
        
                loss.backward()
                self.opt.step()
        
                # print statistics
                running_loss += loss.item()
                if i % 512 == 511:
                    print('Train Epoch: {} [ ({:.0f}%)]\tLoss: {:.6f}'.format(self.num_epochs,
                               100. * i/ len(train_loader), running_loss / ((i+1) * self.batch_size)))
            
                if i % (len(train_loader)) == (len(train_loader) - 1):    # print every 2000 mini-batches
                    training_loss.append((running_loss/len(train_loader)))
                    running_loss = 0
            self.num_epochs += 1
        
        #print('Finished Training')
        
    def inject_single_rules(self, rules : list, n : int, threshold = 0.7):
        item_emb = self.model.item.weight
        item_bias_emb = torch.squeeze(self.model.item_bias.weight)
        
        #loop over all rules
        #for rule in rules:
        random_choice = random.sample(range(len(rules)), n)
        for element in random_choice:
            r = rules[element]
            ccs = r.ccs 
            
            non_zero_entries = self.x_train[np.where((self.x_train[:,0] == ccs) & (self.x_train[:,2] != 0))[0], 1]
            ### this means that there is not exist a pair of measurements in this ccs so we cannot infer a new rule
            if len(non_zero_entries) < 2:
                continue
            possible_injections = non_zero_entries.shape[0] + 1
            
            ccs_emb = self.model.ccs.weight[ccs, :] 
            ccs_bias_emb = self.model.ccs_bias.weight[ccs]
            
            logits = torch.matmul(item_emb, ccs_emb) + ccs_bias_emb + item_bias_emb
            logit_2_prob = nn.Sigmoid()
            prob = logit_2_prob(logits)

            ### randomly choose which of the two is the implied measurement
            choice = random.randint(0,1)
            if choice == 0:
                implied_measurement = r.measurement1
                m1 = r.measurement2
                
            else:
                implied_measurement = r.measurement2
                m1 = r.measurement1

            if implied_measurement > 0:
                cand, idx = torch.topk(prob, possible_injections)
            else: 
                cand, idx = torch.topk(1 - prob, possible_injections)

            j = 0
            for j in range(possible_injections):
                pi = cand[j].item()
                m = idx[j].item()
                if (m not in non_zero_entries) and (pi > threshold):
                    inject_ccs = ccs
                    inject_measurement = m
                    inject_index = np.where((self.x_train[:,0] == inject_ccs) & (self.x_train[:,1] == inject_measurement))[0]
                    self.x_train[inject_index,2] = np.sign(implied_measurement) * pi
                    self.injected_rules.append(rule(inject_ccs, m1, np.sign(implied_measurement) * m , pi))
                    break


    def accuracy_statistics(self, x : torch.tensor, condition = False, thresh = 0.0):
        if self.used_model == 'logistic_mf':
            output = self.model.forward(x)
        elif self.used_model == 'smf':
            output = self.model.predict(x, condition = condition)
    
        output[output > thresh] = 1
        output[output < -thresh] = -1
        output[(output < thresh) & (output > -thresh)] = 0
        y_true = x[:,2].detach().cpu().clone().numpy()
        y_pred =  output.detach().cpu().clone().numpy()
        print(accuracy_score(y_true, y_pred))
        print(confusion_matrix(y_true, y_pred))
    
    
        accuracies = np.zeros(self.num_ccs)
        counts = np.zeros(self.num_ccs)
        x_np = x.detach().cpu().clone().numpy()
        count = 0
        for i in range(self.num_ccs):
            ccs_test = np.where(x_np[:,0] == i)[0]
            if ccs_test.shape[0] == 0:
                count += 1
            y_true_i = y_true[ccs_test]
            y_pred_i = y_pred[ccs_test]
            accuracies[i] = accuracy_score(y_true_i, y_pred_i)
            counts[i] = y_true_i.shape[0]
        return counts , accuracies
    
    
    def print_AUC(self, x : torch.tensor, condition = False):
        if self.used_model == 'logistic_mf':
            output = self.model.forward(x)
        elif self.used_model == 'smf':
            output = self.model.predict(x, condition = condition)
    
        y_true = x[:,2].detach().cpu().clone().numpy()
        y_pred =  output.detach().cpu().clone().numpy()
        y_pred_pos = y_pred[y_true == 1]
        y_pred_neg = y_pred[y_true == -1]
    
        auc = 0
        for i in y_pred_pos:
            for j in y_pred_neg:
                if i > j:
                    auc += 1
        auc = auc / (y_pred_pos.shape[0] *  y_pred_neg.shape[0])
        print(auc)
