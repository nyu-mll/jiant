# Adding a Task

In this tutorial we’ll show you how to add a new task to `jiant`.

We’ll use a real task for this example: Senteval’s Tense task. Senteval’s tense task is a single sentence classification task with labels “PAST” or “PRES”, and it uses accuracy as its evaluation metric.

Adding Senteval’s Tense task will require touching three files:
1. A (new) task file: `jiant/tasks/lib/senteval/tense.py`
2. The task retrieval library: `jiant/tasks/retrieval.py`
3. The evaluation library: `jiant/tasks/evaluate/core.py`

We’ll talk about these files in the following sections.

## 1. The Task file

Tasks files have 5 core components:

1. `Task` class
2. `Example` dataclass
3. `TokenizedExample` dataclass
4. `DataRow` dataclass
5. `Batch` dataclass

In the following sections we’ll explain these components (and build up a working example in code blocks).

#### 1. `Task` class

`Task` classes have attributes describing the task, and methods for producing `Example`s.
1. Attributes:
    1. `TASK_TYPE`: This attribute tells the model setup process what type of head the task requires. For our task the type is `CLASSIFICATION`.
    2. `LABELS` are used for mapping the label in the inputs files to input IDs.
2. `Example` getter methods instantiate and return `Examples`. When `jiant`’s core code calls the `get_train_examples()` method, this method must return an iterable of task `Example`s.

Here's an example of how we can define a `SentevalTenseTask` class:
```python
class SentevalTenseTask(Task):
    TASK_TYPE = TaskTypes.CLASSIFICATION

    LABELS = ["PAST", "PRES"]
    LABEL_TO_ID, ID_TO_LABEL = labels_to_bimap(LABELS)

    def get_train_examples(self):
        return self._create_examples(path=self.train_path, set_type="train")

    def get_val_examples(self):
        return self._create_examples(path=self.val_path, set_type="val")

    def get_test_examples(self):
        return self._create_examples(path=self.test_path, set_type="test")

    @classmethod
    def _create_examples(cls, path, set_type):
        examples = []
        df = pd.read_csv(path, index_col=0, names=["split", "label", "text", "unk_1", "unk_2"])
        for i, row in df.iterrows():
            examples.append(
                Example(
                    guid="%s-%s" % (set_type, i),
                    text=row.text,
                    label=row.label if set_type != "test" else cls.LABELS[-1],
                )
            )
        return examples
```

#### 2. `Example` dataclass
This dataclass must implement a `tokenize()` method. The `tokenize()` method is expected to take a tokenizer as an argument, use the tokenizer to tokenize the input text, translate label strings into label IDs, and create and return a `TokenizedExample`.

```python
@dataclass
class Example(BaseExample):
    guid: str
    text: str
    label: str

    def tokenize(self, tokenizer):
        return TokenizedExample(
            guid=self.guid,
            text=tokenizer.tokenize(self.text),
            label_id=SentevalTenseTask.LABEL_TO_ID[self.label],
        )
```

#### 3. `TokenizedExample` dataclass

The `TokenizedExample` dataclass must implement a `featurize()` method that takes a `Tokenizer` and `FeaturizationSpec`. It must return the example as a `DataRow` (including input IDs, input mask, and label ID and applying any model specific max sequence length and special tokens). The resulting `DataRow` can also optionally include metadata (e.g., guid and tokens) as well.

Note: the task we’re implementing has a common format (single-sentence classification) and we can use the `single_sentence_featurize()` function to form `DataRow` which has all the components expected by a Transformer model. For tasks with unusual featurization requirements, the featurize method can be home to additional featurization logic.

```python
@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    text: List
    label_id: int

    def featurize(self, tokenizer, feat_spec):
        return single_sentence_featurize(
            guid=self.guid,
            input_tokens=self.text,
            label_id=self.label_id,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
            data_row_class=DataRow,
        )
```

#### 4. `DataRow` dataclass
This dataclass’ fields contain the data that will be passed to the model after batching.

```python
@dataclass
class DataRow(BaseDataRow):
    guid: str
    input_ids: np.ndarray
    input_mask: np.ndarray
    segment_ids: np.ndarray
    label_id: int
    tokens: list
```

#### 5. `Batch` dataclass
This dataclass will contain examples batched for consumption by the model. Type annotations are used to cast the numpy data structures to the appropriate Torch Tensor type.

```python
@dataclass
class Batch(BatchMixin):
    input_ids: torch.LongTensor
    input_mask: torch.LongTensor
    segment_ids: torch.LongTensor
    label_id: torch.LongTensor
    tokens: list
```

Now that we've implemented required components of our SentevalTenseTask, we're ready to add it to the task retrieval library.

## 2. Task retrieval library
To register your task with the task retrieval library (in [`jiant/tasks/retrieval.py`](../../jiant/tasks/retrieval.py)) you simply associate your Task class with a short name in the `TASK_DICT`:

```python
TASK_DICT = {
...
	"senteval_tense": SentevalTenseTask,
	...
}

```

## 3. Evaluation library
Your task will need an evaluation scheme. Task evaluation schemes are specified in [`jiant/tasks/evaluate/core.py`](../../jiant/tasks/evaluate/core.py). Many common evaluation schemes are defined in this file. As a reminder, the `SentevalTenseTask` we’re adding in this guide is a two-class classification task which is evaluated using accuracy, so we’ll add it to the list of tasks that use the `SimpleAccuracyEvaluationScheme`:

```python
def get_evaluation_scheme_for_task(task) -> BaseEvaluationScheme:
    if isinstance(
        task,
        (
            # ...
            tasks.SentevalTenseTask,
	    # ...
        ),
    ):
        return SimpleAccuracyEvaluationScheme()
```

## 4. Add a data downloader
If the new task is publicly available, add the new task to the data downloader in [`jiant/scripts/download_data/runscript.py`](../../jiant/scripts/download_data/runscript.py). There are two flavors of supported task downloaders:

1) If the new task is supported by Hugging Face's [Datasets](https://huggingface.co/nlp/viewer/):
    - Add the task to `OTHER_HF_DATASETS_TASKS` in [`jiant/scripts/download_data/constants.py`](../../jiant/scripts/download_data/constants.py). Also add the task to `HF_DATASETS_CONVERSION_DICT` in [`jiant/scripts/download_data/dl_datasets/hf_datasets_tasks.py`](../../jiant/scripts/download_data/dl_datasets/hf_datasets_tasks.py). `HF_DATASETS_CONVERSION_DICT` is used to map field changes in Dataset's to the original dataset.

2) If the task is not supported by Hugging Face's Datasets, and the dataset is publicly available for download (i.e. downloadable directly via wget, and not behind an authentication wall such as Google Drive):
    - Add a function to directly download the task here: [`jiant/scripts/download_data/dl_datasets/files_tasks.py`](../../jiant/scripts/download_data/dl_datasets/files_tasks.py). The function signature should be similar to:

```python
def download_senteval_data_and_write_config(
    task_name: str, task_data_path: str, task_config_path: str
)
```

The direct download function for your task should generate output a config object to `task_config_path` based on the data required to complete the task. For Senteval, the task config object is:

```python
py_io.write_json(
    data={
        "task": "senteval",
        "paths": {
            "train": os.path.join(task_data_path, "train.jsonl"),
            "val": os.path.join(task_data_path, "valid.jsonl"),
            "test": os.path.join(task_data_path, "tests.jsonl"),
        },
        "name": "senteval",
    },
    path=task_config_path,
)
```

## 5. Update the supported tasks documentation
Add your task to [`guides/tasks/supported_tasks.md`](../../guides/tasks/supported_tasks.md).


## 6. (Recommended) Tag the creator in the pull request
Please @ the creator of the task in your PR to let them know it is part of `jiant`! This also gives them a chance to review the task.


## Congratulations!
And that’s it. You’ve made all the core code changes required to include the `SentevalTenseTask` in your `jiant` experiments.

What's next? To tokenize and cache your `SentevalTenseTask` (which you shortnamed `senteval_tense`) for an experiement, you'll need to provide a task config `json` file pointing to your task's data:

```json
{
  "task": "senteval_tense",
  "paths": {
    "train": "/data/Senteval/past_present/train.csv",
    "val": "/data/Senteval/past_present/val.csv",
    "test": "/data/Senteval/past_present/test.csv"
  },
  "name": "senteval_tense"
}
```

To learn more about running experiments with you new task, check out the examples [available here](../README.md).
