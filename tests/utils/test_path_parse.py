import jiant.utils.path_parse as path_parse


def test_tags_to_regex():
    assert path_parse.tags_to_regex(
        "/path/to/experiments/{model}/{task}"
    ) == '/path/to/experiments/(?P<model>\\w+)/(?P<task>\\w+)'

    assert path_parse.tags_to_regex(
        "/path/to/experiments/{model}/{task}",
        default_format="(\\w|_)+"
    ) == '/path/to/experiments/(?P<model>(\\w|_)+)/(?P<task>(\\w|_)+)'

    assert path_parse.tags_to_regex(
        "/path/to/experiments/{model}/{task}",
        format_dict={"task": "(\\w|_)+"}
    ) == '/path/to/experiments/(?P<model>\\w+)/(?P<task>(\\w|_)+)'
