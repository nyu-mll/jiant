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

        Args:
            - inputs (tuple(TODO))
            - pred_layer (nn.Module)
            - pair_input (int)

        Returns:
            - logits (TODO)
        '''
        if pair_input:
            try:
                word_embs1 = self.token_embedder(inputs[0]).transpose(0, 1)
                word_embs2 = self.token_embedder(inputs[1]).transpose(0, 1)
            except Exception as e:
                pdb.set_trace()
            f_states1, b_states1 = self.encoder(word_embs1)
            f_states2, b_states2 = self.encoder(word_embs2)
            sent_emb1 = torch.cat([f_states1[-1], b_states1[0]], 1)
            sent_emb2 = torch.cat([f_states2[-1], b_states2[0]], 1)
            logits = pred_layer(torch.cat([sent_emb1, sent_emb2,
                                           torch.abs(sent_emb1 - sent_emb2),
                                           sent_emb1 * sent_emb2], 1))
        else:
            sent_emb = self.encoder(inputs[0])
            logits = pred_layer(sent_emb)
        return logits

    def train_model(self, n_epochs, optimizer, lr):
        '''
        Train model on all the datasets in tasks

        TODO
            - allow fine-grained control for order of training
            - allowed for mixed-tasks within a batch
            - might want per task optimizer
        '''
        self.train()

        optimizer = optim.SGD(self.parameters(), lr=lr)

        for epoch in range(n_epochs):
            log.info("Epoch {0}".format(epoch))
            start_time = time.time()
            total_loss = 0.0
            for task in self.tasks: # TODO want time per task
                for ins, targs in task.train_data:
                    optimizer.zero_grad()
                    outs = self(ins, task.pred_layer, task.pair_input)
                    loss = F.cross_entropy(outs, targs)
                    total_loss += loss.data[0]
                    loss.backward()
                    optimizer.step()

            val_scores = [str(s) for s in self.evaluate(self.tasks, 'val')]
            log.info("\tTraining loss: {0}".format(total_loss))
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
        if split == 'val':
            scores = [task.validate(self) for task in tasks]
        elif split == 'test':
            scores = [task.test(self) for task in tasks]
        else:
            raise ValueError("Invalid split for evaluation!")
        return scores


    def embed(self, sents):
        '''
        Get embeddings for sentences

        Args:
            - sents (TODO)
        '''
        raise NotImplementedError
