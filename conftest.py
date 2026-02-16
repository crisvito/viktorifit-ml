"""
conftest.py — Pytest Global Configuration
-------------------------------------------
Placed in the repo ROOT so pytest finds it automatically before any test runs.

WHY THIS FILE EXISTS:
  When pytest imports `app.app`, Python executes the top-level imports:

      from utils.common.predict_workout    import get_workout_plan_json
      from utils.common.predict_meal       import get_meal_plan_json
      from utils.common.predict_userprogress import get_progress_roadmap

  Each of those modules loads model files at import time:
      predict_workout.py      → opens models/model_workout.pickle
      predict_meal.py         → opens models/model_meal.pickle
      predict_userprogress.py → opens models/progress_encoders.pickle + all .onnx files

  During the CI 'test' job, model files are NOT downloaded (they live in the
  'validate-models' job which uses Git LFS). Without this file, every test
  would crash with FileNotFoundError before a single assertion runs.

THE FIX:
  We inject fake (MagicMock) modules into sys.modules BEFORE `app.app` is
  imported. Python sees "oh, utils.common.predict_workout is already loaded"
  and skips the real file entirely. The individual @patch decorators in
  test_models.py then control what each function returns per test.
"""

import sys
from unittest.mock import MagicMock

# Pre-register the entire utils.common package tree as mocks.
# Order matters: parent packages must be registered before children.
for module_path in [
    "utils",
    "utils.common",
    "utils.common.predict_workout",
    "utils.common.predict_meal",
    "utils.common.predict_userprogress",
]:
    sys.modules.setdefault(module_path, MagicMock())