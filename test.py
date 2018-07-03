weighting_method = ...


if weighting_method == 'uniform':
    sample_weights = [1] * len(tasks)
elif weighting_method == 'proportional':
    sample_weights = [task_infos[task.name]['n_tr_batches'] for task in tasks]
    max_weight = max(sample_weights)
    min_weight = min(sample_weights)
elif weighting_method == 'proportional_log': # haven't written loss scaling
    sample_weights = [math.log(task_infos[task.name]['n_tr_batches']) for task in tasks]
    max_weight = max(sample_weights)
    min_weight = min(sample_weights)
samples = random.choices(tasks, weights=sample_weights, k=validation_interval)
