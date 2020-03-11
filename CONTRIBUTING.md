# Contributing to `jiant`

Thanks for contributing to `jiant`! :+1:

Please review the guidelines below before opening a PR.

### PR and review process guidelines:
1. Choose a descriptive PR title (“Adding SNLI Task” rather than “add task”).
2. In the PR description field provide a summary that explains the motivation for the changes, and link to any relevant GitHub issues related to the PR.
3. PRs should address only one issue (or a few very closely related issues).
4. While your PR is a work in progress (WIP), use GitHub’s [Draft PR feature](https://github.blog/2019-02-14-introducing-draft-pull-requests/) to provide visibility without requesting a review (publishing a draft PR while your changes are a WIP helps organize/invite comments on proposed changes, and avoids duplication of work).
5. Before requesting review, make sure your code passes tests:
    1. Run unit tests locally using `python -m unittest discover tests/`
    2. Run formatting tests locally with `black {source_file_or_dir}`, or, after installing all jiant requirements and activating the environment, you can run `pre-commit install` so that black checking and reformatting runs automatically before committing your staged changes.
    3. For involved changes, run a minimal experiment (e.g., with “max_vals=1 and val_interval=10”) to check that your changes don’t break at runtime.
6. Once your PR is ready for review, in your Draft PR press “Ready for review”. This will invite code owners to provide reviews, and makes the branch mergeable.
