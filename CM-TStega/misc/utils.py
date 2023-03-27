from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import os
import sys
import six
import pickle

bad_endings = ['with','in','on','of','a','at','to','for','an','this','his','her','that']
bad_endings += ['the']

def pickle_load(f):
    """ Load a pickle.
    Parameters
    ----------
    f: file-like object
    """
    if six.PY3:
        return pickle.load(f, encoding='latin-1')
    else:
        return pickle.load(f)


def pickle_dump(obj, f):
    """ Dump a pickle.
    Parameters
    ----------
    obj: pickled object
    f: file-like object
    """
    if six.PY3:
        return pickle.dump(obj, f, protocol=2)
    else:
        return pickle.dump(obj, f)


def if_use_feat(caption_model):
    # Decide if load attention feature according to caption model
    if caption_model in ['show_tell', 'all_img', 'fc', 'newfc']:
        use_att, use_fc = False, True
    elif caption_model == 'language_model':
        use_att, use_fc = False, False
    elif caption_model in ['topdown', 'aoa']:
        use_fc, use_att = True, True
    else:
        use_att, use_fc = True, False
    return use_fc, use_att

# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i,j]
            if ix > 0 :
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix.item())]
            else:
                break
        if int(os.getenv('REMOVE_BAD_ENDINGS', '0')):
            flag = 0
            words = txt.split(' ')
            for j in range(len(words)):
                if words[-j-1] not in bad_endings:
                    flag = -j
                    break
            txt = ' '.join(words[0:len(words)+flag])
        out.append(txt.replace('@@ ', ''))
    return out

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]

        output = -input.gather(2, target.unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False, reduce=False)
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.true_dist = None
        
    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]

        input = to_contiguous(input).view(-1, input.size(-1))
        target = to_contiguous(target).view(-1)
        mask = to_contiguous(mask).view(-1)

        self.size = input.size(1)
        true_dist = input.data.clone()

        true_dist.fill_(self.smoothing / (self.size - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        return (self.criterion(input, true_dist).sum(1) * mask).sum() / mask.sum()

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def get_lr(optimizer):
    for group in optimizer.param_groups:
        return group['lr']

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)

def build_optimizer(params, opt):
    if opt.optim == 'rmsprop':
        return optim.RMSprop(params, opt.learning_rate, opt.optim_alpha, opt.optim_epsilon, weight_decay=opt.weight_decay)
    elif opt.optim == 'adagrad':
        return optim.Adagrad(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgd':
        return optim.SGD(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdm':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdmom':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay, nesterov=True)
    elif opt.optim == 'adam':
        return optim.Adam(params, opt.learning_rate, (opt.optim_alpha, opt.optim_beta), opt.optim_epsilon, weight_decay=opt.weight_decay)
    else:
        raise Exception("bad option opt.optim: {}".format(opt.optim))
    

def penalty_builder(penalty_config):
    if penalty_config == '':
        return lambda x,y: y
    pen_type, alpha = penalty_config.split('_')
    alpha = float(alpha)
    if pen_type == 'wu':
        return lambda x,y: length_wu(x,y,alpha)
    if pen_type == 'avg':
        return lambda x,y: length_average(x,y,alpha)

def length_wu(length, logprobs, alpha=0.):
    """
    NMT length re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`.
    """

    modifier = (((5 + length) ** alpha) /
                ((5 + 1) ** alpha))
    return (logprobs / modifier)

def length_average(length, logprobs, alpha=0.):
    """
    Returns the average probability of tokens in a sequence.
    """
    return logprobs / length


def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()      
    
def get_secret_acc(secret_true, secret_pred):
    if 'cuda' in str(secret_pred.device):
        secret_pred = secret_pred.cpu()
        secret_true = secret_true.cpu()
    secret_pred = torch.round(secret_pred)
    correct_pred = torch.sum((secret_pred - secret_true) == 0, dim=1)
    #str_acc = 1.0 - torch.sum((correct_pred - secret_pred.size()[1]) != 0).numpy() / correct_pred.size()[0]
    bit_acc = torch.sum(correct_pred).numpy() / secret_pred.numel()
    return bit_acc


def loadGloveModel(gloveFile):
    """
    Load the glove model / glove model after counter-fitting.
    """
    print("Loading Glove Model")
    f = open(os.path.join("Vectors", gloveFile), "r", encoding="utf-8")
    model = {}
    for line in f:
        row = line.strip().split(" ")
        word = row[0]
        embedding = np.array([float(val) for val in row[1:]])
        model[word] = embedding
    print("Done.", len(model), " words loaded!")
    return model


def create_embeddings_matrix(
    glove_model,
    dictionary,
    embedding_size=300,
):
    embedding_matrix = np.zeros(shape=((embedding_size, len(dictionary)+1)))
    cnt = 0
    unfound_ids = []
    unfound_words = []
    for w, i in dictionary.items():
        if not w in glove_model:
            cnt += 1
            unfound_ids.append(i)
            unfound_words.append(w)
        else:
            embedding_matrix[:, i] = glove_model[w]
    print("Number of not found words = ", cnt)
    return embedding_matrix, unfound_ids


def compute_dist_matrix(dic):
    """
    Create a distance matrix of size (vacab_size+1, vocab_size+1),
    and record the distance between two words in the GloVe embedding space after counter-fitting.
    The distances related to `UNK` (word id=0) are set to INFINITY.
    """
    INFINITY = 100000
    embedding_matrix, missed = None, None
    print("embeddings_counter_COCO.npy" + " not exists.")
    glove_tmp = loadGloveModel("counter-fitted-vectors.txt")
    embedding_matrix, missed = create_embeddings_matrix(glove_tmp, dic)
    np.save(
        os.path.join(
            "Vectors",
            "aux_files",
            "embeddings_counter_COCO.npy",
        ),
        embedding_matrix,
    )
    np.save(
        os.path.join(
            "./Vectors",
            "aux_files",
            "missed_embeddings_counter_COCO.npy",
        ),
        missed,
    )

    embedding_matrix = embedding_matrix.astype(np.float32)
    c_ = -2 * np.dot(embedding_matrix.T, embedding_matrix)
    a = np.sum(np.square(embedding_matrix), axis=0).reshape((1, -1))
    b = a.T
    dist = a + b + c_
    dist[0, :] = INFINITY
    dist[:, 0] = INFINITY
    dist[missed, :] = INFINITY
    dist[:, missed] = INFINITY 
    print("success to compute distance matrix!")
    return dist


def create_small_embedding_matrix(
    dist_mat, vocab_size, threshold=1.5, retain_num=50
):
    """
    Create the synonym matrix. 
    The i-th row represents the synonyms of the word with id i and their distances.
    """
    small_embedding_matrix = np.zeros(shape=((vocab_size + 1, retain_num, 2)))
    for i in range(vocab_size + 1):
        if i % 1000 == 0:
            print("%d/%d processed." % (i, vocab_size))
        dist_order = np.argsort(dist_mat[i, :])[1 : 1 + retain_num]
        dist_list = dist_mat[i][dist_order]
        mask = np.ones_like(dist_list)
        if threshold is not None:
            mask = np.where(dist_list < threshold)
            dist_order, dist_list = dist_order[mask], dist_list[mask]
        n_return = len(dist_order)
        dist_order_arr = np.pad(
            dist_order, (0, retain_num - n_return), "constant", constant_values=(-1, -1)
        )
        dist_list_arr = np.pad(
            dist_list, (0, retain_num - n_return), "constant", constant_values=(-1, -1)
        )
        small_embedding_matrix[i, :, 0] = dist_order_arr
        small_embedding_matrix[i, :, 1] = dist_list_arr
    return small_embedding_matrix


def building_embedding_vectors(w2i, i2w, vocab_size):
    print("build and save the dict......")

    os.makedirs('./Vectors/aux_files')

    with open(os.path.join('./Vectors/aux_files', 'COCO_vocab.pkl'), 'wb') as f:
        pickle.dump(w2i, f, protocol=4)

    with open(os.path.join('./Vectors/aux_files', 'COCO_rev_vocab.pkl'), 'wb') as f:
        pickle.dump(i2w, f, protocol=4)    

    print("create and save the small dist counter...")
    dist_mat = compute_dist_matrix(w2i)
    small_dist_mat = create_small_embedding_matrix(dist_mat, vocab_size)
 
    print('small dist counter created!')
    np.save(os.path.join('Vectors', 'aux_files', 'small_dist_counter_COCO.npy'), small_dist_mat)

    print('embeddings glove not exists, creating...')
    glove_model = loadGloveModel('glove.840B.300d.txt')
    glove_embeddings, _ = create_embeddings_matrix(glove_model, w2i)
    print("embeddings glove created!")
    np.save(os.path.join('Vectors', 'aux_files', 'embeddings_glove_COCO.npy'), glove_embeddings)

    print("Over!!!")
