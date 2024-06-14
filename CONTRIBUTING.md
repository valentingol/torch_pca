# Contributing to My Worflow Template

Everyone can contribute to Pytorch PCA, and we value everyone’s contributions.
There are several ways to contribute, including:

- Raising [issue](https://github.com/valentingol/torch_pca/issues)
  on the Github repository

- Proposing [Pull requests](https://github.com/valentingol/torch_pca/pulls)
  to the Github repository

- Contact me by email (valentin.goldite@gmail.com)

- Create your own repository based on this one and cite it

## Pull request checklist

Before proposing a PR you must follow some rule:

- Pull requests typically comprise a **single git commit**. In preparing a pull
  request for review, you may need to squash together multiple commits.

- Code should work on Python 3.8-3.11

- Code should respect [PEP8](https://peps.python.org/pep-0008/)

- The format of the docstrings follows [Numpy guidline](https://numpydoc.readthedocs.io/en/latest/format.html)

- It is recommended to use all linters listed in `requirements-dev.txt`

Before submitting a PR you should run this pipeline:

```script
sh checks.sh
```

**Try to not decrease the global Pylint score** after a pull request. A minimum
of 7.0/10.0 is required but **we preferably want above 9.0/10.0.**

## Commit message

Commits should start with an emoji and directly followed by a descriptive and
precise message that starts with a capital letter and should be written in present
tense. E.g:

*✨: added configuration function* ❌ Bad

*✨ Add function to save configuration file* ✅ Good

Emojis not only look great but also makes you rethink what to add to a commit.
The goal is to dedicate each single kind of change to a single commit. Make many
but small commits!

Emojis of commit message follow mainly the [Gitmoji](https://gitmoji.dev/) guideline
(the different ones start with an asterisk *). The most useful are:

| Emoji                                 | Description                                             |
| ------------------------------------- | ------------------------------------------------------- |
| 🎉 `:tada:`                            | Initial commit                                          |
| ✨ `:sparkles:`                        | New cool feature                                        |
| ➕ `:heavy_plus_sign:` *               | Add file and/or folder                                  |
| 🔥 `:fire:`                            | Remove some code or file                                |
| 📝 `:memo:`                            | Add or improve readme, docstring or comments            |
| 🐛 `:bug:`                             | Fix a bug                                               |
| 🎨 `:art:`                             | Improve style, or format the code                       |
| ♻️ `:recycle:`                         | Refactor the code                                       |
| 🚚 `:truck:`                           | Rename and/or move files and folders                    |
| 🏗️ `:building_construction:`           | Change a part of the repository architecture            |
| ✏️  `:pencil2:`                        | Fix typo                                                |
| ⚙️  `:gear:` *                         | Add or update configuration file (config/exp.yaml, ...) |
| 🔧 `:wrench:`                          | Add or update tool configuration (pyproject.toml, ...)  |
| 🍱 `:bento:`                           | Add or update assets                                    |
| 🚀 `:rocket:` *                        | Improve performance                                     |
| ⚗️ `:alembic:`                         | Perform experiment                                      |
| 🚸 `:children_crossing:`               | Improve user experience                                 |
| 🆙 `:up:` * OR 🔖 `:bookmark:`          | Update the version/tag                                  |
| ⬆️  `:arrow_up:`                       | Upgrade dependency                                      |
| 🚧 `:construction:`                    | Work in progress                                        |
| 🔀 `:twisted_rightwards_arrows:`       | Merge a branch                                          |
| Check [Gitmoji](https://gitmoji.dev/) | *OTHER*                                                 |

Installing the [Gitmoji VSCode extension](https://marketplace.visualstudio.com/items?itemName=seatonjiang.gitmoji-vscode)
can be useful to get the emoji you want quickly.

## Version and tag numbers

Version/tag numbers will be assigned according to the [Semantic Versioning scheme](https://semver.org/).
This means, given a version number MAJOR.MINOR.PATCH, we will increment the:

- MAJOR version when we make incompatible API changes
- MINOR version when we add functionality in a backwards compatible manner
- PATCH version when we make backwards compatible bug fixes
