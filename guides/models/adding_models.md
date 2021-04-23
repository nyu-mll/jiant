 # Adding a model

`jiant` supports or can easily be extended to support Hugging Face's [Transformer models](https://huggingface.co/transformers/viewer/) since `jiant` utilizes [Auto Classes](https://huggingface.co/transformers/model_doc/auto.html) to determine the architecture of the model used based on the name of the [pretrained model](https://huggingface.co/models). Although `jiant` uses AutoModels to reolve model classes, the `jiant` pipeline requires additional information (such as matching the correct tokenizer for the models). Furthermore, there are subtle differences in the models that `jiant` must abstract and additional steps are required to add a Hugging Face model to `jiant`. To add a model not currently supported in `jiant`, follow the following steps:

## 1. Add to ModelArchitectures enum
Add the model to the ModelArchitectures enum in [`model_resolution.py`](../../jiant/tasks/model_resolution.py) as a member-string mapping. For example, adding the field DEBERTAV2 = "deberta-v2" would add Deberta V2 to the ModelArchitectures enum.

## 2. Add to the TOKENIZER_CLASS_DICT
Add the model to the TOKENIZER_CLASS_DICT in [`model_resolution.py`](../../jiant/tasks/model_resolution.py). This dictionary maps the ModelArchitectures to Hugging Face tokenizer classes.

## 3. Subclass JiantTransformersModel
Create a subclass of JiantTransformersModel in ['jiant/proj/main/modeling/primary.py'](../../jiant/proj/main/modeling/primary.py). The JiantTransformersModel is used to wrap Hugging Face Transformer models to abstract any inconsistencies in the model fields. JiantTransformersModel is an abstract class with several methods that must be implemented as well as several methods that can be optionally overridden.


```python
class JiantTransformersModel(metaclass=abc.ABCMeta):
    def __init__(self, baseObject):
        self.__class__ = type(
            baseObject.__class__.__name__, (self.__class__, baseObject.__class__), {}
        )
        self.__dict__ = baseObject.__dict__

    @classmethod
    @abc.abstractmethod
    def normalize_tokenizations(cls, tokenizer, space_tokenization, target_tokenization):
        """Abstract method to tag space_tokenization and process target_tokenization with
        the relevant tokenization method for the model."""
        pass

    @abc.abstractmethod
    def get_mlm_weights_dict(self, weights_dict):
        """Abstract method to get the pre-trained masked-language modeling head weights
        from the pretrained model from the weights_dict"""
        pass

    @abc.abstractmethod
    def get_feat_spec(self, weights_dict):
        """Abstract method that should return a FeaturizationSpec specifying the
        tokenization details used for the model"""
        pass

    def get_hidden_size(self):
        ...

    def get_hidden_dropout_prob(self):
        ...

    def encode(self, input_ids, segment_ids, input_mask, output_hidden_states=True):
        ...
```

`jiant` uses a dynamic registry for supported models. To add your model to the dynamic registry add a decorator to the new model class registering the new model class with the corresponding ModelArchitecture enum added in Step 1.


```python
@JiantTransformersModelFactory.register(ModelArchitectures.DEBERTAV2)
class JiantDebertaV2Model(JiantTransformersModel):
    def __init__(self, baseObject):
        super().__init__(baseObject)
```

## (Optional) 4. Add/Register additional task heads
Specific task heads may require model-specific implementation such as the MLM task heads. These model-specific task heads should be added and registered with their respective factory in (jiant/proj/main/modeling/heads.py)[../../jiant/proj/main/modeling/heads.py] if applicable. For example, MLM heads require a factory since their implementation differs across models:

```python
@JiantMLMHeadFactory.register([ModelArchitectures.DEBERTAV2])
class DebertaV2MLMHead(BaseMLMHead):
    ...
````

## 5. Fine-tune the model
You should now be able to use the new model with the following simple fine-tuning example (Deberta-V2 used as an example below):

```python
from jiant.proj.simple import runscript as run
import jiant.scripts.download_data.runscript as downloader

EXP_DIR = "/path/to/exp"

# Download the Data
downloader.download_data(["mrpc"], f"{EXP_DIR}/tasks")

# Set up the arguments for the Simple API
args = run.RunConfiguration(
   run_name="simple",
   exp_dir=EXP_DIR,
   data_dir=f"{EXP_DIR}/tasks",
   hf_pretrained_model_name_or_path="microsoft/deberta-v2-xlarge",
   tasks="mrpc",
   train_batch_size=16,
   num_train_epochs=3
)

# Run!
run.run_simple(args)
```
