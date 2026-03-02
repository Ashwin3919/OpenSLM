"""Register MyModel with the plugin registry.

After copying this template:
    1. Change "my_model" to your model's unique name.
    2. Replace ``MyModel`` with your class name.
    3. Create ``configs/model/<your_name>_*.yaml`` with ``model_type: <your_name>``.
"""

from src.core.registry import register_model
from .model import MyModel

# Uncomment and rename once your model is ready:
# register_model("my_model")(MyModel)
