from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import time
import os
import traceback
import opts
import models
from models.SecretModel import *
from dataloader import *
import eval_utils
import misc.utils as utils
from misc.loss_wrapper import LossWrapper
from models.AoAModel import AoAModel
from loguru import logger
import sys

try:
    import tensorboardX as tb
except ImportError:
    print("tensorboardX is not installed")
    tb = None

#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)

def train(opt):
    # Deal with feature things before anything
    opt.use_fc, opt.use_att = utils.if_use_feat(opt.caption_model)
    if opt.use_box: opt.att_feat_size = opt.att_feat_size + 5

    acc_steps = getattr(opt, 'acc_steps', 1)
        
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    tb_summary_writer = tb and tb.SummaryWriter(opt.checkpoint_path)

    infos = {}
    histories = {}
    if opt.start_from is not None:
        # open old infos and check if cap_models are compatible
        with open(os.path.join(opt.start_from, 'infos_'+opt.id+'.pkl'), 'rb') as f:
            infos = utils.pickle_load(f)
            saved_model_opt = infos['opt']
            need_be_same=["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved cap_model disagree on '%s' " % checkme

        if os.path.isfile(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')):
            with open(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl'), 'rb') as f:
                histories = utils.pickle_load(f)
    else:
        infos['iter'] = 0
        infos['epoch'] = 0
        infos['iterators'] = loader.iterators
        infos['split_ix'] = loader.split_ix
        infos['vocab'] = loader.get_vocab()
    infos['opt'] = opt

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    best_loss = 99.
    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    opt.SS_attack = True
    opt.vocab = loader.get_vocab()
    rev_vocab = loader.get_rev_vocab()

    cap_model = AoAModel(opt).cuda()
    # del opt.vocab
    lw_cap_model = LossWrapper(cap_model, opt)

    # Defined secret models
    sec_encoder = SecretEncoder(opt.secret_size).cuda()
    sec_extractor = SecretExtractor(opt).cuda()
    BCE_loss = torch.nn.BCELoss().cuda()

    epoch_done = True
    # Assure in training mode
    lw_cap_model.train()
    sec_encoder.train()
    sec_extractor.train()

    total_var = [{'params': cap_model.parameters()},
                {'params': sec_encoder.parameters()},
                {'params': sec_extractor.parameters()}]

    optimizer = utils.build_optimizer(total_var, opt)
    # Load the models and the optimizer
    if vars(opt).get('start_from', None) is not None:
        cap_model.load_state_dict(torch.load(os.path.join(opt.start_from, 'cap_model.pth')))
        sec_encoder.load_state_dict(torch.load(os.path.join(opt.start_from, 'sec_encoder.pth')))
        sec_extractor.load_state_dict(torch.load(os.path.join(opt.start_from, 'sec_extractor.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))

    def save_checkpoint(cap_model, sec_extractor, sec_encoder, infos, optimizer, histories=None, append=''):
        if len(append) > 0:
            append = '-' + append
        # if checkpoint_path doesn't exist
        if not os.path.isdir(opt.checkpoint_path):
            os.makedirs(opt.checkpoint_path)
        cap_checkpoint_path = os.path.join(opt.checkpoint_path, 'cap_model%s.pth' %(append))
        sec_enc_checkpoint_path = os.path.join(opt.checkpoint_path, 'sec_encoder%s.pth' %(append))
        sec_ext_checkpoint_path = os.path.join(opt.checkpoint_path, 'sec_extractor%s.pth' %(append))
        torch.save(cap_model.state_dict(), cap_checkpoint_path)
        torch.save(sec_encoder.state_dict(), sec_enc_checkpoint_path)
        torch.save(sec_extractor.state_dict(), sec_ext_checkpoint_path)
        print("Models saved to {}".format(opt.checkpoint_path))

        optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer%s.pth' %(append))
        torch.save(optimizer.state_dict(), optimizer_path)
        with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'%s.pkl' %(append)), 'wb') as f:
            utils.pickle_dump(infos, f)
        if histories:
            with open(os.path.join(opt.checkpoint_path, 'histories_'+opt.id+'%s.pkl' %(append)), 'wb') as f:
                utils.pickle_dump(histories, f)

    while True:
        if epoch_done:
            if not opt.noamopt and not opt.reduce_on_plateau:
                # Assign the learning rate
                if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                    frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                    decay_factor = opt.learning_rate_decay_rate  ** frac
                    opt.current_lr = opt.learning_rate * decay_factor
                else:
                    opt.current_lr = opt.learning_rate
                utils.set_lr(optimizer, opt.current_lr) # set the decayed rate
            # Assign the scheduled sampling prob
            if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                cap_model.ss_prob = opt.ss_prob

            epoch_done = False
        
        # Load data from train split (0)
        data = loader.get_batch('train', opt.secret_size)

        if (iteration % acc_steps == 0):
            optimizer.zero_grad()
        
        torch.cuda.synchronize()
        start = time.time()
        tmp = [data['sec_mes'], data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
        tmp = [_ if _ is None else _.cuda() for _ in tmp]
        sec_mes, fc_feats, att_feats, labels, masks, att_masks = tmp
        sec_matrix = sec_encoder(sec_mes)
        att_feats = torch.cat([att_feats, sec_matrix], dim=1)
        sec_seq, key, _ = cap_model(fc_feats, att_feats, att_masks, mode='sample')
        if  opt.SS_attack:
            if not os.path.exists('./Vectors/aux_files'):
                utils.building_embedding_vectors(w2i=rev_vocab, i2w=opt.vocab, vocab_size=opt.vocab_size)
        
            dist_mat = np.load('Vectors/aux_files/small_dist_counter_COCO.npy')
            dist_mat = dist_mat[:, :4, 0]

            ss_loc = torch.randint(0, 10, (sec_seq.size(0),)).unsqueeze(1)
            pre_idx = sec_seq.cpu().gather(1, ss_loc).squeeze(1)
            for i in range(opt.batch_size):
                try:
                    idx = torch.multinomial(torch.tensor(dist_mat[pre_idx[i], :]), 1)
                    sec_seq[i, ss_loc[i]] = torch.tensor(int(dist_mat[pre_idx[i], idx]))
                except:
                    continue
                
        decoded_sec_mes = sec_extractor(sec_seq, key)
        cap_model_out = lw_cap_model(fc_feats, att_feats, labels, masks, att_masks)
        cap_loss = cap_model_out['loss'].mean()
        sec_loss = BCE_loss(decoded_sec_mes, sec_mes)

        if epoch <= 2:
            total_loss = cap_loss + 1e-8*sec_loss
        else:
            total_loss = cap_loss + 10*sec_loss
            
        loss_sp = total_loss / acc_steps

        loss_sp.backward()
        if ((iteration+1) % acc_steps == 0):
            utils.clip_gradient(optimizer, opt.grad_clip)
            optimizer.step()
        torch.cuda.synchronize()
        total_loss = total_loss.item()
        bit_acc = utils.get_secret_acc(sec_mes, decoded_sec_mes)
        end = time.time()
        if iteration % opt.print_steps == 0:
            print("iter {} (epoch {}), total_loss = {:.3f},  cap_loss  = {:.3f}, sec_loss  = {:.3f}, ext_acc = {:.3f},  time/batch = {:.3f}" \
                .format(iteration, epoch, total_loss, cap_loss, sec_loss, bit_acc, end - start))

        # Update the iteration and epoch
        iteration += 1
        if data['bounds']['wrapped']:
            epoch += 1
            epoch_done = True

        # Write the training loss summary
        if (iteration % opt.losses_log_every == 0):
            add_summary_value(tb_summary_writer, 'total_loss', total_loss, iteration)
            add_summary_value(tb_summary_writer, 'cap_loss', cap_loss, iteration)
            add_summary_value(tb_summary_writer, 'sec_loss', sec_loss, iteration)
            add_summary_value(tb_summary_writer, 'learning_rate', opt.current_lr, iteration)
            add_summary_value(tb_summary_writer, 'scheduled_sampling_prob', cap_model.ss_prob, iteration)

            loss_history[iteration] = total_loss
            lr_history[iteration] = opt.current_lr
            ss_prob_history[iteration] = cap_model.ss_prob

        # update infos
        infos['iter'] = iteration
        infos['epoch'] = epoch
        infos['iterators'] = loader.iterators
        infos['split_ix'] = loader.split_ix
        
        # make evaluation on validation set, and save cap_model
        if (iteration % opt.save_checkpoint_every == 0):
            # eval cap_model
            eval_kwargs = {'split': 'val',
                            'dataset': opt.input_json}
            eval_kwargs.update(vars(opt))
            val_loss, predictions, lang_stats = eval_utils.eval_split(
                cap_model, sec_encoder, sec_extractor, lw_cap_model.crit, loader, eval_kwargs)

            # Write validation result into summary
            add_summary_value(tb_summary_writer, 'validation loss', val_loss, iteration)
            if lang_stats is not None:
                for k,v in lang_stats.items():
                    add_summary_value(tb_summary_writer, k, v, iteration)
            val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

            # Save cap_model if is improving on validation result
            # current_score = lang_stats['CIDEr']
            current_loss = cap_loss + 3*sec_loss

            best_flag = False

            # if best_val_score is None or current_score > best_val_score:
            #     best_val_score = current_score
            #     best_flag = True

            if current_loss <= best_loss:
                best_loss = current_loss
                best_flag = True

            # Dump miscalleous informations
            infos['best_val_score'] = best_val_score
            histories['val_result_history'] = val_result_history
            histories['loss_history'] = loss_history
            histories['lr_history'] = lr_history
            histories['ss_prob_history'] = ss_prob_history

            save_checkpoint(cap_model, sec_extractor, sec_encoder, infos, optimizer, histories)

            if best_flag:
                save_checkpoint(cap_model, sec_extractor, sec_encoder, infos, optimizer, append='best')

        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break


opt = opts.parse_opt()
train(opt)
