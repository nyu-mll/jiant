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

### Test and documentation guidelines:
1. Tests: Test coverage and testing methods are expected to vary by PR. Test plans should be proposed/discussed early in the PR process. Except in special cases...
     * Bug fixes should be paired with a test and/or other validations demonstrating that the bug has been squashed.
     * Changes introducing a new feature should come with related unit tests.
     * Changes introducing a new task should come with performance benchmark tests demonstrating that the code can roughly reproduce published performance numbers for that task.
     * Changes to existing code are expected to come with some documented effort to test for related regressions. In cases where changes affect code that does not have high test coverage, this might involve designing validations and documenting them in the PR thread. If you’re unsure what validations are necessary, ping a code owner for guidance.
2. Documentation: Public functions and methods should be documented with [numpy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html). If you’re making a major change to a public function/method that does not already have a docstring, you should add one.

### Additional guidelines and helpful reminders for specific types of PRs:
* For PRs that change package dependencies, update both `environment.yml` (used for conda) and `setup.py` (used by pip, and in automatic CircleCI tests).
* For all PRs, make sure to update any existing config files, tutorials, and scripts to match your changes.
* For PRs that typical users will need to be aware of, make a matching PR to the [documentation](https://github.com/nyu-mll/jiant-site/edit/master/documentation/README.md). We will merge that documentation PR once the original PR is merged in and pushed out in a release. (Proposals for better ways to do this are welcome.)
* For PRs that add a new model or task, explain what types of sentence encoders are supported for that task.
* For PRs that could change performance across multiple tasks, run performance regression tests on at least one representative task from each “family” of tasks.
