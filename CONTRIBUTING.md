# Contributing to Osculari

Thank you for considering to contribute to **osculari**.

Everyone is welcome to get involved with the project. There are different ways to contribute:

1. Report bugs through [GitHub issues](https://github.com/ArashAkbarinia/osculari/issues):
   - Do a quick search first to see whether others reported a similar issue.
   - In case you find an unreported bug, please open a new ticket.
   - Try to provide as much information as possible. Report using one of the available templates. Some tips:
     - Clear title and description of the issue.
     - Explain how to reproduce the error.
     - Report your package versions to facilitate the task.
     - Try to include a code sample/test that raises the error.
2. Fix a bug or develop a feature from the roadmap:
   - We will always have an open ticket showing the current roadmap.
   - Pick an unassigned feature (or potentially propose a new one) or an open bug ticket.
   - Check our coding conventions. See more details below.
   - Run the test framework locally and make sure all works as expected before sending a pull request.
   - Open a Pull Request, get the green light from the CI, and get your code merged.

# Coding Standards

This section provides general guidance for developing code for the project. The following rules will serve as a guide in
writing high-quality code that will allow us to scale the project and ensure that the code base remains readable and
maintainable.

- Use meaningful names for variables, functions, and classes.

- Write small incremental changes:

  - To have a linear and clean commits history, we recommend committing each small change that you do to the
    source code.
  - Clear commit messages will help to understand the progress of your work.
  - Please, avoid pushing large files.

- We give support to static type checker for Python >= 3.8

  - Please, read
    [MyPy cheatsheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html#type-hints-cheat-sheet-python-3) for
    Python 3.
  - It is recommended to use typing inside the function, **when** it would increase readability.
  - **Always** type function input and output

- Format your code:

  - We follow [PEP8 style guide](https://www.python.org/dev/peps/pep-0008).
  - Line length is 100 characters.
