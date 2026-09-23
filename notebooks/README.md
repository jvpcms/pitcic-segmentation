# Notebooks

Figure generation only. No notebook is part of the pipeline: everything that
produces a number lives in `src/pitcic/`, and a notebook that trains or scores a
model is a copy of that logic waiting to drift out of sync with it.

Outputs are stripped before committing. Install the filter once:

```sh
pip install -e '.[figures]'
nbstripout --install
```

This is not cosmetic. In the working repository this replaces, one notebook
carried 37 MB of embedded output and the history reached 1,4 GB.
