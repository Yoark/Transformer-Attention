from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from .model import gen_model_dir

import os
from functools import partial
from collections import deque, defaultdict
from torchnlp.modules.transformer.sublayers import MultiHeadAttention
import pickle
import shutil
import json, codecs
import logging
import random

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

logger = logging.getLogger(__name__)

OPTIMIZER_FILE = "optimizer.pt"

def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data
# Decay functions to be used with lr_scheduler
def lr_decay_noam(hparams):
    return lambda t: (
        10.0 * hparams.hidden_size**-0.5 * min(
        (t + 1) * hparams.learning_rate_warmup_steps**-1.5, (t + 1)**-0.5))

def lr_decay_exp(hparams):
    return lambda t: hparams.learning_rate_falloff ** t


# Map names to lr decay functions
lr_decay_map = {
    'noam': lr_decay_noam,
    'exp': lr_decay_exp
}

        
def compute_num_params(model):
    """
    Computes number of trainable and non-trainable parameters
    """
    sizes = [(np.array(p.data.size()).prod(), int(p.requires_grad)) for p in model.parameters()]
    return sum(map(lambda t: t[0]*t[1],sizes)), sum(map(lambda t: t[0]*(1 - t[1]),sizes))
    



class Trainer(object):
    """
    Class to handle training in a task-agnostic way
    """
    def __init__(self, task_name, model, hparams, train_iter, evaluator):
        """
        Parameters:
            task_name: Name of the task
            model: Model instance (derived from model.Model)
            hparams: Instance of HParams
            train_iter: An instance of torchtext.data.Iterator
            evaluator: Instance of evalutation.Evaluator that will
                        run metrics on the validation dataset
        """

        self.task_name = task_name
        self.model = model
        self.hparams = hparams
        self.evaluator = evaluator
        self.val_iter = train_iter[1]
        self.train_iter = train_iter[0]
        self.test_iter = train_iter[2]
        # Disable repetitions
        self.train_iter.repeat = False

        model_params = filter(lambda p: p.requires_grad, model.parameters())
        
        # TODO: Add support for other optimizers
        self.optimizer = optim.Adam(
                            model_params,
                            betas=(hparams.optimizer_adam_beta1, hparams.optimizer_adam_beta2), 
                            lr=hparams.learning_rate)

        self.opt_path = os.path.join(gen_model_dir(task_name, model.__class__), 
                                    OPTIMIZER_FILE)
        self.file_path = gen_model_dir(task_name, model.__class__)
        # If model is loaded from a checkpoint restore optimizer also
        if int(model.iterations) > 0:
            self.optimizer.load_state_dict(torch.load(self.opt_path))

        self.lr_scheduler_step = self.lr_scheduler_epoch = None
        from collections import defaultdict

        # def get_attentions(name, md, inp, out):
        #     self.attentions.append(out.cpu)

        # if True:
        #     self.attentions = defaultdict(list)
        #     for name, m in self.model.named_modules():
        #         if isinstance(m, MultiHeadAttention):
        #             m.register_forward_hook(hook=get_attentions)

        # Set up learing rate decay scheme
        if hparams.learning_rate_decay is not None:
            if '_' not in hparams.learning_rate_decay:
                raise ValueError("Malformed learning_rate_decay")
            lrd_scheme, lrd_range = hparams.learning_rate_decay.split('_')

            if lrd_scheme not in lr_decay_map:
                raise ValueError("Unknown lr decay scheme {}".format(lrd_scheme))
            
            lrd_func = lr_decay_map[lrd_scheme]            
            lr_scheduler = optim.lr_scheduler.LambdaLR(
                                            self.optimizer, 
                                            lrd_func(hparams),
                                            last_epoch=int(self.model.iterations) or -1
                                        )
            # For each scheme, decay can happen every step or every epoch
            if lrd_range == 'epoch':
                self.lr_scheduler_epoch = lr_scheduler
            elif lrd_range == 'step':
                self.lr_scheduler_step = lr_scheduler
            else:
                raise ValueError("Unknown lr decay range {}".format(lrd_range))

        # # ! add adv flag here to read in the attention data for future using
            
        # Display number of parameters
        logger.info('Parameters: {}(trainable), {}(non-trainable)'.format(*compute_num_params(self.model)))
    
    def _get_early_stopping_criteria(self, early_stopping):
        es = early_stopping.split('_')
        if len(es) != 3:
            raise ValueError('Malformed early stopping criteria')
        best_type, window, metric = es
        logger.info('Early stopping for {} value of validation {} after {} epochs'
                    .format(best_type, metric, window))
        
        if best_type == 'lowest':
            best_fn = partial(min, key=lambda item: item[0])
        elif best_type == 'highest':
            best_fn = partial(max, key=lambda item: item[0])
        else:
            raise ValueError('Unknown best type {}'.format(best_type))

        return best_fn, int(window), metric

    def train(self, num_epochs, early_stopping=None, save=True):
        """
        Run the training loop for given number of epochs. The model
        is evaluated at the end of every epoch and saved as well
        Parameters:
            num_epochs: Total number of epochs to run
            early_stopping: A string indicating how to perform early stopping
                Should be of the form lowest/highest_n_metric where:
                    lowest/highest: Track lowest or highest values
                    n: The window size within which to track best
                    metric: Name of the metric to track. Should be available
                        in the dict returned by evaluator
            save: Save model every epoch if true
        Returns:
            Tuple of best checkpoint number and metrics array (for plotting etc)
        """

        all_metrics = defaultdict(list)
        best_iteration = 0

        if early_stopping:
            if not save:
                raise ValueError('save should be True for early stopping')
            if self.evaluator is None:
                raise ValueError('early stopping requires an eval function')

            best_fn, best_window, best_metric_name = self._get_early_stopping_criteria(early_stopping)
            tracking = deque([], best_window + 1)

        # ! add adv flag here, so, if adv: pack data batch and attn batch and prediction batch together

        train_data = list(self.train_iter)
        test_data = list(self.test_iter) 
        train_pack = train_data
        test_pack = test_data
        if self.model.adversarial:
            train_pack = [(tr, attn, pr) for tr, attn, pr in zip(train_data, self.model.attn_tr, self.model.pr_tr)]
            test_pack = [(te, attn, pr) for te, attn, pr in zip(test_data, self.model.attn_te, self.model.pr_te)]

        if self.model.froze:
            train_pack = [(tr, attn) for tr, attn in zip(train_data, self.model.attn_tr)]
            test_pack = [(te, attn) for te, attn in zip(test_data, self.model.attn_te)]


        for epoch in range(num_epochs):
            # * Allow the training to have shuffling.
            # add advsarial flag here
            # self.train_iter.shuffle = True 
            # self.train_iter.init_epoch()
            # prog_iter = tqdm(self.train_iter, leave=False)
            # if self.model.adversarial:
            random.shuffle(train_pack)
            prog_iter = train_pack
            # else:
            #     random.shuffle(train_data)
            #     prog_iter = train_data
            prog_iter = tqdm(prog_iter, leave=False)

            epoch_loss = 0
            tvd_loss = 0
            kld_loss = 0
            count = 0
            # train_batchs = list(self.train_iter)
            logger.info('Epoch %d (%d)'%(epoch + 1, int(self.model.iterations)))
            froze_attn = None
            # self.model.train()
            for batch in prog_iter:
                # ! just give it something
                # ! just give it something
                # ! just give it something
                # Train mode
                self.model.train()
                if  self.model.adversarial:
                    # pr here is the target predictions
                    data, attn, pr = batch

                    attn = torch.from_numpy(attn).to(device)
                    pr = torch.from_numpy(pr).to(device)
                    
                    self.optimizer.zero_grad()

                    pr_loss, _, attns_cur = self.model.loss(data, target_pr=pr)

                    # log the attentions here...

                    kl_loss = self.model.criterion(attn.log(), attns_cur[0])
                    total_loss = pr_loss - self.hparams.lmbda * kl_loss
                    # logger.info(f'total_loss: {total_loss}, kl_loss: {kl_loss}, nll_loss: {pr_loss}')
                    total_loss.backward()
                    self.optimizer.step()

                    if self.lr_scheduler_step:
                        self.lr_scheduler_step.step()

                    epoch_loss += total_loss.item()
                    tvd_loss += pr_loss.item()
                    kld_loss += kl_loss.item()

                    count += 1
                    self.model.iterations += 1
                
                    # Display loss
                    prog_iter.set_description('Training')
                    prog_iter.set_postfix(loss=(epoch_loss/count))

                    # for hk in hooks:
                    #     hk.remove()

                elif not self.model.adversarial and not self.model.froze:
                    self.optimizer.zero_grad()
                    # if self.adversarial
                    # data, _, _ = batch
                    loss, _, _ = self.model.loss(batch)
                    loss.backward()
                    self.optimizer.step()

                    if self.lr_scheduler_step:
                        self.lr_scheduler_step.step()

                    epoch_loss += loss.item()
                    count += 1
                    self.model.iterations += 1
                
                    # Display loss
                    prog_iter.set_description('Training')
                    prog_iter.set_postfix(loss=(epoch_loss/count))

                elif self.model.froze:
                    data, attn = batch

                    froze_attn = torch.from_numpy(attn).to(device)
                    # pr = torch.from_numpy(pr).to(device)
                    # ! look here
                    # froze_attn.requires_grad = False
                    self.optimizer.zero_grad()

                    loss, _,_  = self.model.loss(data, froze_attn=froze_attn)

                    # log the attentions here...

                    # logger.info(f'total_loss: {total_loss}, kl_loss: {kl_loss}, nll_loss: {pr_loss}')
                    loss.backward()
                    self.optimizer.step()

                    if self.lr_scheduler_step:
                        self.lr_scheduler_step.step()

                    epoch_loss += loss.item()

                    count += 1
                    self.model.iterations += 1
                
                    # Display loss
                    prog_iter.set_description('Training')
                    prog_iter.set_postfix(loss=(epoch_loss/count))


                # else:
                #     self.optimizer.zero_grad()
                #     # # add the other losses here
                #     loss, _ = self.model.loss(batch, batch_attn)

                #     loss = loss
                #     loss.backward()


            ####
            
            # train_at, train_pr = self.catch_attention(self.train_iter)
            # test_at, test_pr = self.catch_attention(self.test_iter)
            # attentions_tr = [el.tolist() for el in train_at]
            # attentions_te = [el.tolist() for el in test_at]
            # ####
            if self.lr_scheduler_epoch:
                self.lr_scheduler_epoch.step()
            if self.model.adversarial:
                logger.info(f'total_loss: {epoch_loss/count}, kl_loss: {kld_loss/count}, nll_loss: {tvd_loss/count}')
            train_loss = epoch_loss/count
            all_metrics['train_loss'].append(train_loss)
            logger.info('Train Loss: {:3.5f}'.format(train_loss))

            best_iteration = int(self.model.iterations)
            # Run evaluation
            if self.evaluator:
                eval_metrics, pre_loss, div_loss = self.evaluator.evaluate(self.model, froze_attn=froze_attn)
                if not isinstance(eval_metrics, dict):
                    raise ValueError('eval_fn should return a dict of metrics')

                # Display eval metrics
                logger.info('Validation metrics: ')
                logger.info(', '.join(['{}={:3.5f}'.format(k, v) for k,v in eval_metrics.items()]))

                # Append metrics
                for k, v in eval_metrics.items():
                    all_metrics[k].append(v)

                # Handle early stopping
                tracking.append((eval_metrics[best_metric_name], int(self.model.iterations), epoch))
                logger.debug('Epoch {} Tracking: {}'.format(epoch, tracking))
                
                if epoch >= best_window:
                    # Get the best value of metric in the window
                    best_metric, best_iteration, best_epoch = best_fn(tracking)
                    if tracking[0][1] == best_iteration:
                        # The best value has gone outside the desired window
                        # hence stop
                        #! save here, heihei
                        logger.info('Early stopping at iteration {}, epoch {}, {}={:3.5f}'
                                    .format(best_iteration, best_epoch, best_metric_name, best_metric))
                        # Update the file time of that checkpoint file to latest
                        
                        self.model.set_latest(self.task_name, best_iteration)
                        # ! save the attention for the best model, which make sense
                          
                        # attentions_tr = [el.tolist() for el in train_at]
                        # attentions_te = [el.tolist() for el in test_at]
                        # prediction_tr = [el.tolist() for el in train_pr] 
                        # prediction_te = [el.tolist() for el in test_pr]
                        if not self.model.froze:
                            train_at, train_pr = self.catch_attention(self.train_iter)
                            test_at, test_pr = self.catch_attention(self.test_iter)
                            print("SAVING PREDICTIONS AND ATTENTIONS")
                            import datetime
                            time = str(datetime.datetime.now().time())
                            
                            dirname = os.path.join('/home/zijiao/research/atal/saved_'+str(self.model.adversarial)+'_', time)

                            if not os.path.exists(dirname):
                                os.makedirs(dirname, mode=0o777)
                            else:
                                shutil.rmtree(dirname)
                                os.makedirs(dirname, mode=0o777)

                            # json.dump(prediction_tr, open(os.path.join(dirname, '/train_predictions_best_epoch.json'), 'w'))
                            # json.dump(prediction_te, open(os.path.join(dirname, '/test_predictions_best_epoch.json'), 'w'))
                            # json.dump(attentions_tr, open(os.path.join(dirname, '/train_attentions_best_epoch.json'), 'w'))
                            # json.dump(attentions_te, open(os.path.join(dirname, '/test_attentions_best_epoch.json'), 'w'))
                            save_data(train_pr, dirname+'/tr_pr_best')
                            save_data(test_pr, dirname+'/te_pr_best')
                            save_data(train_at, dirname+'/tr_attn_best')
                            save_data(test_at, dirname+'/te_attn_best')
                        break
                
            if save:
                self.model.save(self.task_name)
                torch.save(self.optimizer.state_dict(), self.opt_path)

        return best_iteration, all_metrics, pre_loss, div_loss            

            
    def catch_attention(self, it):
        """Set iteration to generate non-shuffling data. Add hook to 
        the model, run a forward pass, log the attention obtained for best 
        epoch

        Arguments:
            data {[type]} -- [description]
        """
        def get_attentions(md, inp, out):
            attns.append(out[1].cpu().data.numpy())

        it.shuffle = False
        it.init_epoch()
        final_iter = tqdm(it, leave=False)

        hooks = []
        for name, m in self.model.named_modules():
            if isinstance(m, MultiHeadAttention):
                hk = m.register_forward_hook(hook=get_attentions) 
                hooks.append(hk)

        attns = []
        outputs = []

        with torch.no_grad():
            for batch in final_iter:
                #! not need to use model.loss, reduced computation, yay!!!
                predictions = self.model(batch)
                outputs.append(predictions.cpu().data.numpy())

        # attns = [x for y in attns for x in y]
        # outputs = [x for y in outputs for x in y]
        # remove hook
        for hk in hooks:
            hk.remove()

        logger.info("prediction, and attns for best epoch obtained, release hook")
        #! save it somewhere:
        # try pickle first
        return attns, outputs
    # def get_attentions(self, md, inp, out):
    #     self.attentions.append(out.cpu().data.numpy())
def save_data(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f'done {path}')