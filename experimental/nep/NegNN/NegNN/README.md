# Code Re-use!

We use this repo to parse the provided datasets!

1. Install this folder as a package.

```
cd NegNN
python -m pip install -e .  
```

# NegNN (Neural Network for Automatic Negation Detection)

The code implements a feed-forward NN (```/feed_forward```) and a BiLSTM(```/bilstm_class``` or ```/bilstm```) to perform automatic negation scope detection. 

## Data
Training, test and development data can be found in the ```/data``` folder.
For training, development and initial testing, we used the data released for the [*SEM2012 shared task](http://www.clips.ua.ac.be/sem2012-st-neg/); please refer to the shared task and related papers for information regarding the annotation style.
Additional test data extracted from Simple Wikipedia is available in ```/data/test/simple_wiki```. The data was manually annotated following the guidelines released during the *SEM2012 shared task. Please refer to the .pdf file for ulterior information.
The python code in ```/reader``` used to read in the data is part of the code made available by Packard et al. (2014) (["Simple Negation Scope Resolution through Deep Parsing: A Semantic Solution to a Semantic Problem"](https://aclweb.org/anthology/P/P14/P14-1007.pdf)).

## Dependencies
- Tensorflow (tested on v. 0.7.1)
- scikit-learn (tested on v. 0.17.1) - for score report purposes only, feel free to use any other library instead -
- numpy (tested on v. 1.11.0)

## Train
To train the model, first go to the parent directory of the repository and run ```python NegNN/(bilstm_class|bilstm|feed_forward)/train.py```. ```/bilstm_class``` is a more elegant implementation that wraps the BiLSTM code inside a separate class so to avoid any repetition. There seems to be however problems with this implementation when run on MacOsX El capitan 10.11.5; if so, please run the less elegant implementation in the ```/bilstm``` folder
```train.py``` accepts the following flags
```
-h, --help            show this help message and exit
  --embedding_dim EMBEDDING_DIM
                        Dimensionality of character embedding (default: 50)
  --max_sent_length MAX_SENT_LENGTH
                        Maximum sentence length for padding (default:100)
  --num_hidden NUM_HIDDEN
                        Number of hidden units per layer (default:200)
  --num_classes NUM_CLASSES
                        Number of y classes (default:2)
  --num_epochs NUM_EPOCHS
                        Number of training epochs (default: 50)
  --learning_rate LEARNING_RATE
                        Learning rate(default: 1e-4)
  --scope_detection [SCOPE_DETECTION]
                        True if the task is scope detection or joined
                        scope/event detection
  --noscope_detection
  --event_detection [EVENT_DETECTION]
                        True is the task is event detection or joined
                        scope/event detection
  --noevent_detection
  --POS_emb POS_EMB     0: no POS embeddings; 1: normal POS; 2: universal POS
  --emb_update [EMB_UPDATE]
                        True if input embeddings should be updated (default:
                        False)
  --noemb_update
  --normalize_emb [NORMALIZE_EMB]
                        True to apply L2 regularization on input embeddings
                        (default: False)
  --nonormalize_emb
  --test_set TEST_SET   Path to the test filename (to use only in test mode
  --pre_training [PRE_TRAINING]
                        True to use pretrained embeddings
  --nopre_training
  --training_lang TRAINING_LANG
                        Language of the tranining data (default: en)
```
Please disregard the ```event detection``` flag for the moment; this might be part of future work.


