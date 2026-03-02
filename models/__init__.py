"""Auto-discover and import all model sub-packages.

Any directory inside ``models/`` that does not start with ``_`` is treated as
a model plugin.  Its ``__init__.py`` is expected to call ``register_model``
so the model is available via the registry.

To add a new SLM:
    1. ``cp -r models/_template models/<your_name>``
    2. Implement ``model.py`` and update ``config.py`` / ``__init__.py``.
    3. Create ``configs/model/<your_name>_*.yaml`` with ``model_type: <your_name>``.

No other file needs to change.
"""

import importlib
import os

_models_dir = os.path.dirname(__file__)

for _name in sorted(os.listdir(_models_dir)):
    _path = os.path.join(_models_dir, _name)
    if _name.startswith("_") or not os.path.isdir(_path):
        continue
    importlib.import_module(f"models.{_name}")
