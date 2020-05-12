from jiant.shared.model_resolution import ModelArchitectures, resolve_tokenizer_class


def get_tokenizer(model_type, tokenizer_path):
    model_arch = ModelArchitectures.from_model_type(model_type)
    tokenizer_class = resolve_tokenizer_class(model_type)
    if model_arch in [ModelArchitectures.BERT]:
        if "-cased" in model_type:
            do_lower_case = False
        elif "-uncased" in model_type:
            do_lower_case = True
        else:
            raise RuntimeError(model_type)
    elif model_arch in [
        ModelArchitectures.XLM,
        ModelArchitectures.ROBERTA,
        ModelArchitectures.XLM_ROBERTA,
    ]:
        do_lower_case = False
    elif model_arch in [ModelArchitectures.ALBERT]:
        do_lower_case = True
    else:
        raise RuntimeError(str(tokenizer_class))
    tokenizer = tokenizer_class.from_pretrained(tokenizer_path, do_lower_case=do_lower_case,)
    return tokenizer
