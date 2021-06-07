# Project structure

```
- <lib name>: holds the code
  - __init__.py
- configs
- notebooks
- scripts: console scripts, if needed
- tests: various testing stuff, if needed
```

`__init__.py` must contain this line:
```python
__development__ = True
```
in order to work correctly with connectome's caching.

# Install

```
git clone <repo>
./<repo name>/install.sh
```

# Code of conduct

- List all required libraries in `requirements.txt`
