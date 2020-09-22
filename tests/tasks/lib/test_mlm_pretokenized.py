import transformers

import jiant.shared.model_resolution as model_resolution
import jiant.tasks as tasks


def test_tokenization_and_featurization():
    task = tasks.MLMPretokenizedTask(name="mlm_pretokenized", path_dict={})
    tokenizer = transformers.RobertaTokenizer.from_pretrained("roberta-base")
    example = task.Example(
        guid=None,
        tokenized_text=['Hi', ',', 'Ġmy', 'Ġname', 'Ġis', 'ĠBob', 'ĠRoberts', '.'],
        masked_spans=[[2, 3], [5, 6]],
    )
    tokenized_example = example.tokenize(tokenizer=tokenizer)
    assert tokenized_example.masked_tokens == \
        ['Hi', ',', 'Ġmy', 'Ġname', 'Ġis', 'ĠBob', 'ĠRoberts', '.']
    assert tokenized_example.label_tokens == \
        ['<pad>', '<pad>', 'Ġmy', '<pad>', '<pad>', 'ĠBob', '<pad>', '<pad>']

    data_row = tokenized_example.featurize(
        tokenizer=tokenizer,
        feat_spec=model_resolution.build_featurization_spec(
            model_type="roberta-base",
            max_seq_length=16,
        )
    )
    assert list(data_row.masked_input_ids) == \
        [0, 30086, 6, 127, 766, 16, 3045, 6274, 4, 2, 1, 1, 1, 1, 1, 1]
    assert list(data_row.masked_lm_labels) == \
        [-100, -100, -100, 127, -100, -100, 3045, -100, -100, -100, -100, -100, -100, -100, -100, -100]
