"""
ViktoriFit ML — Unit Tests
tests/test_models.py

Patches the 3 utility functions at the point they are used in app.py:
  - app.app.get_workout_plan_json
  - app.app.get_meal_plan_json
  - app.app.get_progress_roadmap

Mock return shapes match the ACTUAL return values of each utility:
  - get_meal_plan_json    → {"status", "target_daily", "planned_total", "meal_plan"}
  - get_workout_plan_json → {"status", "workout_plan"}  (dict keyed by Day)
  - get_progress_roadmap  → {"status", "total_weeks", "roadmap"}

No real .pickle or .onnx files are needed — conftest.py handles module mocking.

Run locally:
    pytest tests/ -v
    pytest tests/ --cov=app --cov-report=term-missing
"""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.app import app  # conftest.py pre-mocks utils before this import runs

client = TestClient(app)


# ══════════════════════════════════════════════════════════════════════════════
# MOCK RETURN VALUES  — exact shape as the real utility functions return
# ══════════════════════════════════════════════════════════════════════════════

# --- Meal: get_meal_plan_json return shape ---
MOCK_MEAL_SUCCESS = {
    "status": "success",
    "target_daily": {
        "calories": 2200,
        "protein":  180.0,
        "carbs":    200.0,
        "fat":       70.0,
    },
    "planned_total": {
        "calories": 2145.0,
        "protein":  173.5,
        "carbs":    195.0,
        "fat":       68.0,
    },
    "meal_plan": [
        {
            "meal_order": 1,
            "menu_name":  "Chicken Breast",
            "portion":    2.0,
            "calories":   330.0,
            "protein":     62.0,
            "carbs":        0.0,
            "fat":          7.2,
        },
        {
            "meal_order": 2,
            "menu_name":  "Brown Rice",
            "portion":    1.5,
            "calories":   324.0,
            "protein":      6.75,
            "carbs":       67.5,
            "fat":          2.7,
        },
        {
            "meal_order": 3,
            "menu_name":  "Salmon Fillet",
            "portion":    1.5,
            "calories":   312.0,
            "protein":     30.0,
            "carbs":        0.0,
            "fat":         19.5,
        },
    ],
}

# --- Workout: get_workout_plan_json return shape ---
# workout_plan is a dict keyed by day name, each value is a list of exercises
MOCK_WORKOUT_SUCCESS = {
    "status": "success",
    "workout_plan": {
        "Monday": [
            {
                "muscle_group":     "Chest",
                "exercise_name":    "Push Up",
                "sets":             3,
                "reps":             "12",
                "calories_burned":  120.0,
                "duration_minutes": 30,
                "rest_minutes":     1,
                "equipment":        "None",
                "instructions":     "Keep your back straight.",
            }
        ],
        "Wednesday": [
            {
                "muscle_group":     "Back",
                "exercise_name":    "Pull Up",
                "sets":             3,
                "reps":             "10",
                "calories_burned":  130.0,
                "duration_minutes": 30,
                "rest_minutes":     1,
                "equipment":        "Pull-up Bar",
                "instructions":     "Full range of motion.",
            }
        ],
    },
}

# --- Progress: get_progress_roadmap return shape ---
MOCK_PROGRESS_SUCCESS = {
    "status":      "success",
    "total_weeks": 12,
    "roadmap": [
        {
            "week": 1,
            "physical": {
                "weight_kg":           60.5,
                "body_fat_percentage": 14.8,
            },
            "nutrition": {
                "calories":      2550,
                "water_ml":      2800,
                "sugar_limit_g": 45.0,
            },
            "macro": {
                "protein_g": 158.0,
                "carbs_g":   275.0,
                "fat_g":      65.0,
                "fiber_g":    28.0,
            },
        }
    ],
}

# --- Shared failure mock (same structure for all three) ---
MOCK_FAILED = {
    "status": "failed",
    "error":  "Internal model error during inference.",
}


# ══════════════════════════════════════════════════════════════════════════════
# REQUEST PAYLOADS  — match your Pydantic schemas exactly
# ══════════════════════════════════════════════════════════════════════════════

WORKOUT_PAYLOAD = {
    "Age": 25, "Gender": "Male",
    "Height_cm": 175.0, "Weight_kg": 70.0,
    "Body_Fat_Category": 2, "Body_Fat_Percentage": 15.0,
    "Goal": "Weight Loss", "Frequency": 4,
    "Duration": 60, "Level": "Beginner",
    "Environment": "Home",
    "Badminton": 0, "Football": 1,
    "Basketball": 0, "Volleyball": 0, "Swim": 0,
}

MEAL_PAYLOAD = {
    "Daily_Calories":   2200.0,
    "Target_Protein_g": 180.0,
    "Target_Carbs_g":   200.0,
    "Target_Fat_g":      70.0,
    "Frequency":          3,
}

PROGRESS_PAYLOAD = {
    "Age": 25, "Gender": "Male",
    "Height_cm": 175.0, "Initial_Weight_kg": 60.0,
    "Goal": "Muscle Gain", "Level": "Beginner",
    "Body_Fat_Category": 2, "Body_Fat_Percentage": 15.0,
    "Frequency": 4, "Duration": 60,
    "Badminton": 0, "Football": 1,
    "Basketball": 0, "Volleyball": 0, "Swim": 0,
}


# ══════════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ══════════════════════════════════════════════════════════════════════════════

def test_root_health_check():
    """GET / → 200, message field present."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


# ══════════════════════════════════════════════════════════════════════════════
# MEAL TESTS  —  POST /meal
# ══════════════════════════════════════════════════════════════════════════════

@patch("app.app.get_meal_plan_json", return_value=MOCK_MEAL_SUCCESS)
def test_meal_returns_200(mock_fn):
    assert client.post("/meal", json=MEAL_PAYLOAD).status_code == 200


@patch("app.app.get_meal_plan_json", return_value=MOCK_MEAL_SUCCESS)
def test_meal_response_has_meal_plan(mock_fn):
    """Key is 'meal_plan', not 'meals'."""
    data = client.post("/meal", json=MEAL_PAYLOAD).json()
    assert "meal_plan" in data
    assert isinstance(data["meal_plan"], list)
    assert len(data["meal_plan"]) == 3


@patch("app.app.get_meal_plan_json", return_value=MOCK_MEAL_SUCCESS)
def test_meal_response_has_daily_targets(mock_fn):
    data = client.post("/meal", json=MEAL_PAYLOAD).json()
    assert "target_daily" in data
    assert data["target_daily"]["calories"] == 2200


@patch("app.app.get_meal_plan_json", return_value=MOCK_MEAL_SUCCESS)
def test_meal_utility_called_once(mock_fn):
    client.post("/meal", json=MEAL_PAYLOAD)
    mock_fn.assert_called_once()


@patch("app.app.get_meal_plan_json", return_value=MOCK_FAILED)
def test_meal_failed_returns_400(mock_fn):
    assert client.post("/meal", json=MEAL_PAYLOAD).status_code == 400


def test_meal_missing_daily_calories_returns_422():
    bad = {k: v for k, v in MEAL_PAYLOAD.items() if k != "Daily_Calories"}
    assert client.post("/meal", json=bad).status_code == 422


def test_meal_frequency_has_default():
    """Frequency defaults to 3 — omitting it should still succeed."""
    payload = {k: v for k, v in MEAL_PAYLOAD.items() if k != "Frequency"}
    with patch("app.app.get_meal_plan_json", return_value=MOCK_MEAL_SUCCESS):
        assert client.post("/meal", json=payload).status_code == 200


# ══════════════════════════════════════════════════════════════════════════════
# WORKOUT TESTS  —  POST /workout
# ══════════════════════════════════════════════════════════════════════════════

@patch("app.app.get_workout_plan_json", return_value=MOCK_WORKOUT_SUCCESS)
def test_workout_returns_200(mock_fn):
    assert client.post("/workout", json=WORKOUT_PAYLOAD).status_code == 200


@patch("app.app.get_workout_plan_json", return_value=MOCK_WORKOUT_SUCCESS)
def test_workout_response_has_workout_plan(mock_fn):
    """Key is 'workout_plan' (dict by day), not 'schedule' (list)."""
    data = client.post("/workout", json=WORKOUT_PAYLOAD).json()
    assert "workout_plan" in data
    assert isinstance(data["workout_plan"], dict)


@patch("app.app.get_workout_plan_json", return_value=MOCK_WORKOUT_SUCCESS)
def test_workout_plan_contains_days(mock_fn):
    data = client.post("/workout", json=WORKOUT_PAYLOAD).json()
    days = list(data["workout_plan"].keys())
    assert len(days) > 0
    # Each day maps to a list of exercises
    assert isinstance(data["workout_plan"][days[0]], list)


@patch("app.app.get_workout_plan_json", return_value=MOCK_WORKOUT_SUCCESS)
def test_workout_utility_called_once(mock_fn):
    client.post("/workout", json=WORKOUT_PAYLOAD)
    mock_fn.assert_called_once()


@patch("app.app.get_workout_plan_json", return_value=MOCK_FAILED)
def test_workout_failed_returns_400(mock_fn):
    assert client.post("/workout", json=WORKOUT_PAYLOAD).status_code == 400


def test_workout_missing_goal_returns_422():
    bad = {k: v for k, v in WORKOUT_PAYLOAD.items() if k != "Goal"}
    assert client.post("/workout", json=bad).status_code == 422


def test_workout_wrong_type_for_age_returns_422():
    bad = {**WORKOUT_PAYLOAD, "Age": "twenty-five"}
    assert client.post("/workout", json=bad).status_code == 422


def test_workout_sport_flags_default_to_zero():
    """Sport flags are optional (default 0) — omitting them should still succeed."""
    sport_keys = {"Badminton", "Football", "Basketball", "Volleyball", "Swim"}
    payload = {k: v for k, v in WORKOUT_PAYLOAD.items() if k not in sport_keys}
    with patch("app.app.get_workout_plan_json", return_value=MOCK_WORKOUT_SUCCESS):
        assert client.post("/workout", json=payload).status_code == 200


# ══════════════════════════════════════════════════════════════════════════════
# PROGRESS TESTS  —  POST /userprogress
# ══════════════════════════════════════════════════════════════════════════════

@patch("app.app.get_progress_roadmap", return_value=MOCK_PROGRESS_SUCCESS)
def test_progress_returns_200(mock_fn):
    assert client.post("/userprogress", json=PROGRESS_PAYLOAD).status_code == 200


@patch("app.app.get_progress_roadmap", return_value=MOCK_PROGRESS_SUCCESS)
def test_progress_response_has_roadmap(mock_fn):
    data = client.post("/userprogress", json=PROGRESS_PAYLOAD).json()
    assert "roadmap" in data
    assert isinstance(data["roadmap"], list)


@patch("app.app.get_progress_roadmap", return_value=MOCK_PROGRESS_SUCCESS)
def test_progress_roadmap_week_structure(mock_fn):
    """Each roadmap item must have week, physical, nutrition, macro keys."""
    data = client.post("/userprogress", json=PROGRESS_PAYLOAD).json()
    week = data["roadmap"][0]
    assert "week"      in week
    assert "physical"  in week
    assert "nutrition" in week
    assert "macro"     in week


@patch("app.app.get_progress_roadmap", return_value=MOCK_PROGRESS_SUCCESS)
def test_progress_physical_has_weight(mock_fn):
    data  = client.post("/userprogress", json=PROGRESS_PAYLOAD).json()
    phys  = data["roadmap"][0]["physical"]
    assert "weight_kg" in phys


@patch("app.app.get_progress_roadmap", return_value=MOCK_PROGRESS_SUCCESS)
def test_progress_utility_called_once(mock_fn):
    client.post("/userprogress", json=PROGRESS_PAYLOAD)
    mock_fn.assert_called_once()


@patch("app.app.get_progress_roadmap", return_value=MOCK_FAILED)
def test_progress_failed_returns_400(mock_fn):
    assert client.post("/userprogress", json=PROGRESS_PAYLOAD).status_code == 400


def test_progress_missing_initial_weight_returns_422():
    bad = {k: v for k, v in PROGRESS_PAYLOAD.items() if k != "Initial_Weight_kg"}
    assert client.post("/userprogress", json=bad).status_code == 422


def test_progress_sport_flags_default_to_zero():
    sport_keys = {"Badminton", "Football", "Basketball", "Volleyball", "Swim"}
    payload = {k: v for k, v in PROGRESS_PAYLOAD.items() if k not in sport_keys}
    with patch("app.app.get_progress_roadmap", return_value=MOCK_PROGRESS_SUCCESS):
        assert client.post("/userprogress", json=payload).status_code == 200