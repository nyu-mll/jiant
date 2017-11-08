import pdb
import time
import logging as log
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class MultiTaskModel(nn.Module):
    '''
    Playing around designing a class
    '''

    def __init__(self, encoder, token_embedder, tasks):
        '''

        Args:
            - tasks (list[Task]): list of tasks to train and evaluate on
        '''
        super(MultiTaskModel, self).__init__()
        self.tasks = tasks
        self.token_embedder = token_embedder
        self.encoder = encoder # TODO add module?

    def forward(self, inputs, pred_layer, pair_input):
        '''
        Predict through model and task-specific prediction layer

        TODO:
            - MASKING

        Args:
            - inputs (tuple(TODO))
            - pred_layer (nn.Module)
            - pair_input (int)

        Returns:
            - logits (TODO)
        '''
        if pair_input:
            word_embs1 = self.token_embedder.embed_seq_batch(inputs[0])
            word_embs2 = self.token_embedder.embed_seq_batch(inputs[1])
            f_states1, b_states1 = self.encoder(word_embs1.split())
            f_states2, b_states2 = self.encoder(word_embs2.split())

            # max pool over hidden states
            f_states1 = torch.cat([f.values.unsqueeze(0) for f in f_states1], 0)
            b_states1 = torch.cat([b.values.unsqueeze(0) for b in b_states1], 0)
            f_states2 = torch.cat([f.values.unsqueeze(0) for f in f_states2], 0)
            b_states2 = torch.cat([b.values.unsqueeze(0) for b in b_states2], 0)

            sent_emb1, _ = torch.max(torch.cat([f_states1, b_states1], 2), 0)
            sent_emb2, _ = torch.max(torch.cat([f_states2, b_states2], 2), 0)
            sent_emb1, sent_emb2 = sent_emb1.squeeze(0), sent_emb2.squeeze(0)

            # take first and last hidden states of forward/backward as final
            #sent_emb1 = torch.cat([f_states1[-1], b_states1[0]], 1)
            #sent_emb2 = torch.cat([f_states2[-1], b_states2[0]], 1)

            logits = pred_layer(torch.cat([sent_emb1, sent_emb2,
                                           torch.abs(sent_emb1 - sent_emb2),
                                           sent_emb1 * sent_emb2], 1))
        else:
            word_embs = self.token_embedder(inputs).transpose(0, 1)
            f_states, b_states = self.encoder(word_embs)
            f_states = torch.cat([f.unsqueeze(0) for f in f_states],
                                 0).transpose(0, 1)
            b_states = torch.cat([b.unsqueeze(0) for b in b_states],
                                 0).transpose(0, 1)
            sent_emb, _ = torch.max(torch.cat([f_states, b_states], 2), 1)
            sent_emb = sent_emb.squeeze(1)
            logits = pred_layer(sent_emb)
        return logits

    def train_model(self, n_epochs, optimizer, lr, weight_decay=.99):
        '''
        Train model on all the datasets in tasks

        TODO
            - allow fine-grained control for order of training
            - allowed for mixed-tasks within a batch
            - might want per task optimizer
        '''
        self.train()

        optimizer = optim.SGD(self.parameters(), lr=lr,
                              weight_decay=weight_decay)

        for epoch in range(n_epochs):
            log.info("Epoch {0}".format(epoch))
            start_time = time.time()
            total_loss = 0.0
            for task in self.tasks: # TODO want time per task
                for ins, targs in task.train_data:
                    optimizer.zero_grad()
                    outs = self(ins, task.pred_layer, task.pair_input)
                    try:
                        loss = F.cross_entropy(outs, targs)
                        total_loss += loss.cpu().data[0]
                        loss.backward()
                    except Exception as e:
                        pdb.set_trace()
                    optimizer.step()

            log.info("\tTraining loss: {0}".format(total_loss))
            val_scores = [str(s) for s in self.evaluate(self.tasks, 'val')]
            log.info("\tValidation scores: {0}".format(', '.join(val_scores)))

        log.info("Training done in {0}s".format(time.time() - start_time))
        self.eval()

    def evaluate(self, tasks, split='val'):
        '''
        Evaluate on a bunch of tasks

        Args:
            - tasks (list[Task]): a list of tasks
            - split (str): data split to use, either val or test

        Returns:
            - scores (list[float]): a list of scores per task
        '''
        was_training = self.training
        self.eval()
        if split == 'val':
            scores = [task.validate(self) for task in tasks]
        elif split == 'test':
            scores = [task.test(self) for task in tasks]
        else:
            raise ValueError("Invalid split for evaluation!")
        if was_training:
            self.train()
        return scores


    def embed(self, sents):
        '''
        Get embeddings for sentences

        Args:
            - sents (TODO)
        '''
        raise NotImplementedError
