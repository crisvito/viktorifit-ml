from locust import HttpUser, task, between

class TesPerforma(HttpUser):
    wait_time = between(1, 3)

    @task
    def test_meal_recommendation(self):
        payload = {
            "Daily_Calories": 2200,
            "Target_Protein_g": 150,
            "Target_Carbs_g": 200,
            "Frequency": 3
        }
        self.client.post("/meal", json=payload)

    @task
    def test_workout_recommendation(self):
        payload = {
            "Age": 25,
            "Gender": "Male",
            "Height_cm": 175,
            "Weight_kg": 90,
            "Body_Fat_Category": 2,
            "Body_Fat_Percentage": 15.0,
            "Goal": "Muscle Gain",
            "Frequency": 6,
            "Duration": 30,
            "Level": "Beginner",
            "Environment": "Home",
            "Badminton": 0,
            "Football": 1,
            "Basketball": 0,
            "Volleyball": 0,
            "Swim": 1
        }
        self.client.post("/workout", json=payload)

    @task
    def test_userprogress_recommendation(self):
        payload = {  
            "Age": 25, 
            "Gender": "Male", 
            "Height_cm": 175, 
            "Initial_Weight_kg": 60,
            "Goal": "Muscle Gain", 
            "Level": "Beginner",
            "Body_Fat_Category": 2, 
            "Body_Fat_Percentage": 15.0,
            "Frequency": 4, 
            "Duration": 60,
            "Badminton": 0, 
            "Football": 1, 
            "Basketball": 0, 
            "Volleyball": 0, 
            "Swim": 0
        }
        self.client.post("/userprogress", json=payload)