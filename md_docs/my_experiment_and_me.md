# My Experiment and Me

### Run outputs

After running an experiment, you will see your run folder populated with many files and folders. Here's a quick run-down of what they are:

* `args.json`: Saved a copy of your run arguments for future reference.
* `model.p`: Model weights at the end of training.
* `best_model.p`: The best version of the model weights based on validation-subset,
* `best_model.metadata.json`: Contains the metadata for the best-model-weights (e.g. what step of training they were from).  
* `checkpoint.p`: A checkpoint for the run that allows you to resume interrupted runs. Contains additional training state, such as the optimizer state, so it's at least 2x as large as model weights.
* `{log-timestamp}/loss_train.zlog`: JSONL log of training loss over training steps
* `{log-timestamp}/early_stopping.zlog`: JSONL log of early-stopping progress (e.g. steps since last best model)
* `{log-timestamp}/train_val.zlog`: JSONL log of validation-subset evaluation over the course of training (i.e. what's used for early stopping)
* `{log-timestamp}/train_val_best.zlog`: JSONL log of validation-subset evaluation, only recording the improving runs
