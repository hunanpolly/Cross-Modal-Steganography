from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
import json
import os
import sys
import misc.utils as utils

bad_endings = ['a','an','the','in','for','at','of','with','before','after','on','upon','near','to','is','are','am']
bad_endings += ['the']

def count_bad(sen):
    sen = sen.split(' ')
    if sen[-1] in bad_endings:
        return 1
    else:
        return 0

def language_eval(dataset, preds, secret_size, split):
    import sys
    sys.path.append("coco-caption")
    if 'coco' in dataset:
        annFile = '/data/I2T_data/MSCOCO/captions_val2014.json'
    elif 'flickr30k' in dataset or 'f30k' in dataset:
        annFile = 'coco-caption/f30k_captions4eval.json'
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    # encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join('eval_results/', '.cache_'+ str(secret_size) + '_' + split + '.json')

    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w')) # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption
    
    out['bad_count_rate'] = sum([count_bad(_['caption']) for _ in preds_filt]) / float(len(preds_filt))
    outfile_path = os.path.join('eval_results/', str(secret_size) + '_' + split + '.json')
    with open(outfile_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out

def eval_split(cap_model, sec_encoder, sec_extractor, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 1)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    secret_size = eval_kwargs.get('secret_size', 10)
    remove_bad_endings = eval_kwargs.get('remove_bad_endings', 0)
    os.environ["REMOVE_BAD_ENDINGS"] = str(remove_bad_endings) # Use this nasty way to make other code clean since it's a global configuration

    # Make sure in the evaluation mode
    cap_model.eval()
    sec_encoder.eval()
    sec_extractor.eval()
    BCE_loss = torch.nn.BCELoss().cuda()

    loader.reset_iterator(split)

    n = 0
    cap_loss = 0
    sec_loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    while True:
        data = loader.get_batch(split, secret_size)
        n = n + loader.batch_size

        if data.get('labels', None) is not None and verbose_loss:
            # forward the model to get loss
            tmp = [data['sec_mes'], data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
            tmp = [_.cuda() if _ is not None else _ for _ in tmp]
            sec_mes, fc_feats, att_feats, labels, masks, att_masks = tmp

            with torch.no_grad():
                sec_matrix = sec_encoder(sec_mes)
                att_feats = torch.cat([sec_matrix, att_feats], dim=1)
                sec_seq, key, _ = cap_model(fc_feats, att_feats, att_masks, mode='sample')
                decoded_sec_mes = sec_extractor(sec_seq, key)
                sec_loss = BCE_loss(decoded_sec_mes, sec_mes)
                cap_loss = crit(cap_model(fc_feats, att_feats, labels, att_masks), labels[:,1:], masks[:,1:]).item()
            loss_sum = loss_sum + cap_loss + sec_loss
            loss_evals = loss_evals + 1

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        tmp = [data['sec_mes'][np.arange(loader.batch_size) * loader.seq_per_img],
            data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img], 
            data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
            data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img] if data['att_masks'] is not None else None]
        tmp = [_.cuda() if _ is not None else _ for _ in tmp]
        sec_mes, fc_feats, att_feats, att_masks = tmp

        # forward the model to also get generated samples for each image
        with torch.no_grad():
            # seq,  = cap_model(fc_feats, att_feats, att_masks, opt=eval_kwargs, mode='sample')[0].data
            sec_matrix = sec_encoder(sec_mes)
            att_feats = torch.cat([sec_matrix, att_feats], dim=1)
            sec_seq, key, _ = cap_model(fc_feats, att_feats, att_masks, mode='sample')
            sec_seq = sec_seq.cuda()
            decoded_sec_mes = sec_extractor(sec_seq, key)
            sec_acc = utils.get_secret_acc(sec_mes, decoded_sec_mes)
            # print('sec_acc: ', sec_acc)
        # Print beam search
        # if beam_size > 1 and verbose_beam:
        #     for i in range(loader.batch_size):
        #         print('\n'.join([utils.decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0))[0] for _ in cap_model.done_beams[i]]))
        #         print('--' * 10)
        sents = utils.decode_sequence(loader.get_vocab(), sec_seq)


        val_ix = 0
        batch_sen_len = 0
        for k, sent in enumerate(sents):
            sen_len = len(sent.split(' '))
            batch_sen_len += sen_len
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)
            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg' # bit gross
                print(cmd)
                os.system(cmd)
            val_ix += 1

            if val_ix % 100 == 0 and verbose:
                print('image %s: %s' %(entry['image_id'], entry['caption']))
        

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if ix0 % 1000 == 0:
            print('evaluating validation preformance... %d/%d (%f)' %(ix0, ix1, cap_loss))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break
        

    lang_stats = None
    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions, secret_size, split)

    # Switch back to training mode
    cap_model.train()
    sec_encoder.train()
    sec_extractor.train()
    return loss_sum/loss_evals, predictions, lang_stats
