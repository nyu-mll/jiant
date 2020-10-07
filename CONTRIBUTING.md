## Contributing to `jiant`

Thanks for considering contributing to `jiant`! :+1:

#### Guidelines for a successful PR review process:
1. Choose a descriptive PR title (“Adding SNLI Task” rather than “add task”).
2. In the PR description field provide a summary explaining the motivation for the changes, and link to any related issues.
3. PRs should address only one issue (or a few very closely related issues).
4. While your PR is a work in progress (WIP), use the [Draft PR feature](https://github.blog/2019-02-14-introducing-draft-pull-requests/) to provide visibility without requesting a review.
5. Once your PR is ready for review, in your Draft PR press “Ready for review”.

#### Requirements for pull requests (PR) into `jiant`'s master branch:
1. Requirements applied by the automated build system:
   1. black formatting check
   2. flake8 check for style and documentation
   3. pytest unit tests
2. Requirements for successful code reviews:
   1. Code changes must be paired with effective tests.
   2. PRs adding or modifying code must make appropriate changes to related documentation (using [google style](https://google.github.io/styleguide/pyguide.html)).

#### Setting up your local dev environment to run the validation steps applied to PRs by the build system:
```
pip install -r requirements-dev.txt
pre-commit install
```
