import logging

from jiant.tasks.registry import REGISTRY


def test_instantiate_all_tasks():
    """
    All tasks should be able to be instantiated without needing to access actual data

    Test may change if task instantiation signature changes.
    """
    logger = logging.getLogger()
    logger.setLevel(level=logging.CRITICAL)
    for name, (cls, _, kw) in REGISTRY.items():
        cls(
            "dummy_path",
            max_seq_len=1,
            name="dummy_name",
            tokenizer_name="dummy_tokenizer_name",
            **kw,
        )
