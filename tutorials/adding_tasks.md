# How to Add a Task
Alright, so you want to try using `jiant` on a new task.
But where do you begin?


Let us make an imaginary pair classification task with a new dataset SomeDataset, which is made up of two sentences and a binary label, and we want to score based on F1 and use mean square error loss.

First, we need to make sure that your files are in the right place.  Let us say that it is under your data/SomeDataset folder. Make sure your file is in a TSV file.

The first stop is [`jiant/tasks/tasks.py`.
](https://github.com/nyu-mll/jiant/blob/master/jiant/tasks/tasks.py)

We keep track of tasks via a registry, so you'll have to add a decorator to the beginning of your task. In the rel_path, you don't have to add the full path, but only what comes after your general data_dir path.

```python
@register_task(task_name, rel_path='SomeData')
```
There are a lot of tasks already supported in jiant (see list [here](https://jiant.info/documentation#/?id=running)), so inherit from one of the task types (unless your task is its own type). Let's say ours is a type of PairClassificationTask, so we can add that the our task class definition. Here's a skeleton!


```sh
@register_task(task_name, rel_path='SomeData')
class SomeDataClassificationTask(PairClassificationTask):
    def __init__(self, path, max_seq_len, name, **kw):
        super().__init__(name, n_classes=2, **kw)
    def load_data(self, path, max_seq_len):
        '''Process the dataset located at data_file.'''
    def _make_instance(input1, input2, label):
    	''' Make Allennlp instances from records '''
    def update_metrics(self, labels, logits, tagmask=None):
        '''Get metrics specific to the task'''
    def get_metrics(self, reset=False):
        '''Get metrics specific to the task'''
```



Let's walk through each of the major parts:

A lot of the following functions may already be written for your task type (especially if you're using a supported task type like PairClassificationTask). All functions listed with inheritable are those that are written for supported task types. You still may want to override them in your task for task-specific processing though!



1.  The `load_data` (inheritable) function is for loading your data. This function loads your TSV/JSONL/... to a format that can be made into AllenNLP iterators. In this function, you will want to call `load_tsv` from `jiant/utils/data_loaders.py`, which loads and tokenizes the data. Currently, only English tokenization is supported. You can specify the fields to load from as parameters to `load_tsv` (which right now is based on number-based indexing). See [here](https://github.com/jsalt18-sentence-repl/jiant/blob/master/jiant/utils/data_loaders.py) for more documentation on `load_tsv`.  An example is below

```python
    def load_data(self, path, max_seq_len):
        tr_data = load_tsv(self._tokenizer_name, os.path.join(path, "train.tsv"),
                           max_seq_len, s1_idx=3, s2_idx=4, targ_idx=5, skip_rows=1)
        val_data = load_tsv(self._tokenizer_name, os.path.join(path, "dev.tsv"),
                           max_seq_len,  s1_idx=3, s2_idx=4, targ_idx=5, skip_rows=1)
        te_data = load_tsv(self._tokenizer_name, os.path.join(path, 'test.tsv'),
                           max_seq_len, s1_idx=1, s2_idx=2, targ_idx=None, idx_idx=0, skip_rows=1)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data

  ```

2.  The `_make_instance` (inheritable) function, which is defined for PairClassificationTask but we may want to overwrite, indexes the data and converts our Python records into AllenNLP instances.
```python
    def _make_instance(input1, input2, label):
        ''' from multiple types in one column create multiple fields '''
        d = {}
        d["input1"] = sentence_to_text_field(input1, indexers)
        d["input2"] = sentence_to_text_field(input2, indexers)
        d["labels"] = LabelField(label, label_namespace="labels",
                                 skip_indexing=True)
        d['sent1_str'] = MetadataField(" ".join(input1[1:-1]))
        d['sent2_str'] = MetadataField(" ".join(input2[1:-1]))
        return Instance(d)
 ```
3. `update_metrics` (inheritable) is a function to update scorers, which are configerable scorers (mostly from AllenNLP) such as F1Measure or BooleanAccuracy that keeps track of task-specific scores. Let us say that we want to only update F1 and ignore accuracy. In that case, you can set self.scorers = [self.f1_scorer], and this will automatically set the inherited update_metrics function to only update the F1 scorer.

4. `get_metrics` (inheritable) is a function that returns the metrics from the updated scorers in dictionary form. Since we're only getting F1, we should set the get_metrics function to be:
```python
    def get_metrics(self, reset=False):
        '''Get metrics specific to the task'''
        f1 = self.f1_scorer.get_metric(reset)[2]
        return {'f1': f1}
```
5. `get_sentences` (inheritable) iterates over all text to index and returns an iterator of AllenNLP instances of your train and validation sets. Often, you can just set self.sentences to be a list of sentences to handle this, since `get_sentences` yields self.sentences in the `Task` class. Below, after loading the data and saving it into self.train_data_text and self.val_data_text in `load_data`, we set the below:
```python
self.sentences = self.train_data_text[0] + self.val_data_text[0]
```
6. `process_split` (inheritable) takes in a split of your data and produces an iterable of AllenNLP Instances. An Instance is a wrapper around a dictionary of (field_name, Field) pairs. Fields are objects to help with data processing (indexing, padding, etc.). This is handled for us here, since we inherit from `PairClassificationTask`, but if you're writing a task that inherits directly from `Task`, you should look to `PairClassificationTask` for an example of how to implement this method yourself.

7. `count_examples` (inheritable) sets `task.example_counts` (Dict[str:int]): the number of examples per split (train, val, test).

8. `val_metric` (inheritable) is a string variable that is the name of task-specific metric to track during training, e.g. F1 score.

9. `val_metric_decreases` (inheritable) is a boolean for whether or not the objective function should be minimized or maximized. The default is set to False.


Your finished task class may look something like this:


```python
@register_task(task_name, rel_path='SomeData')
class SomeDataClassificationTask(PairClassificationTask):
    def __init__(self, path, max_seq_len, name, **kw):
        super().__init__(name, n_classes=2, **kw)
        self.scorers = [self.f1_scorer]
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text[0] + self.val_data_text[0]

    def load_data(self, path, max_seq_len):
        tr_data = load_tsv(self._tokenizer_name, os.path.join(path, "train.tsv"),
                           max_seq_len, s1_idx=3, s2_idx=4, targ_idx=5, skip_rows=1)
        val_data = load_tsv(self._tokenizer_name, os.path.join(path, "dev.tsv"),
                           max_seq_len,  s1_idx=3, s2_idx=4, targ_idx=5, skip_rows=1)
        te_data = load_tsv(self._tokenizer_name, os.path.join(path, 'test.tsv'),
                           max_seq_len, s1_idx=1, s2_idx=2, targ_idx=None, idx_idx=0, skip_rows=1)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data

    def _make_instance(input1, input2, label):
         """ from multiple types in one column create multiple fields """
        d = {}
        d["input1"] = sentence_to_text_field(input1, indexers)
        d["input2"] = sentence_to_text_field(input2, indexers)
        d["labels"] = LabelField(label, label_namespace="labels",
                                 skip_indexing=True)
        d['sent1_str'] = MetadataField(" ".join(input1[1:-1]))
        d['sent2_str'] = MetadataField(" ".join(input2[1:-1]))
        return Instance(d)


    def get_metrics(self, reset=False):
        '''Get metrics specific to the task'''
        f1 = self.f1_scorer.get_metric(reset)[2]
        return {'f1': f1}
```

Phew! Now, you also have to add the models you're going to use for your task, which lives in [`jiant/models/py`](https://github.com/nyu-mll/jiant/blob/master/jiant/models.py).

Since our task type is PairClassificationTask, a well supported type of task, we can skip this step. However, if your task type is not well supported (or you want to try a different sort of model), in `jiant/models/models.py`, you will need to change the `build_task_specific_module` function to include a branch for your logic.
```python
	def build_task_specific_modules(
        task, model, d_sent, d_emb, vocab, embedder, args):
        task_params = model._get_task_params(task.name)
        if isinstance(task, NewTypeOfTask):
        	hid2voc = build_new_task_specific_module_type(task, d_sent, args)
        	setattr(model, '%s_mdl' % task.name, hid2voc)
```

You will also need to modify the `forward` function in `jiant/models.py` (again, not if you're inheriting from a well-supported task type!), which consist of passing your inputs into a sentence encoder and then through the task specific module you had coded up before. The model will receive the task class you created and a batch of data, where each batch is a dictionary with keys of the Instance objects you created in preprocessing, as well as a predict flag that indicates if your forward function should generate predictions or not.


```python
    elif isinstance(task, NewTypeOfTask):
        out = self._new_task_forward(batch, task, predict)
```

Of course, don't forget to define your task-specific module building function!

```python
    def _new_task_forward(self, batch, task, predict):
        sent1, mask1 = self.sent_encoder(batch['input1'], task)
        sent2, mask2 = self.sent_encoder(batch['input2'], task)
        classifier = self._get_classifier(task)
        logits = classifier(sent1, sent2, mask1, mask2)
        out['loss'] = F.mse_loss(logits, labels)
        tagmask = batch.get("tagmask", None)
        task.update_metrics(logits, batch["labels"], tagmask=tagmask)
        return out
```
Finally, all you have to do is add the task to either the `pretrain_tasks` or `target_tasks` parameter in the config file, and viola! Your task is added.

# Notes

## `boundary_token_fn`

This method applies boundary tokens (like SOS/EOS) to the edges of your text. It also, for BERT and XLNet, applies tokens like [SEP] that delimit the two halves of a two-part input sequence. So, if you'd like to feeding a two-part input into a BERT/XLNet model as a single sequence, create two token sequences, and feed them to `boundary_token_fn` as two arguments.
