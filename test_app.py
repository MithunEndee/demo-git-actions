"""
Unit tests for the Flask API
"""

import unittest
import json
from app import app


class TestFlaskAPI(unittest.TestCase):
    def setUp(self):
        """Set up test client"""
        self.app = app.test_client()
        self.app.testing = True

    def test_home_endpoint(self):
        """Test the home endpoint"""
        response = self.app.get("/")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("message", data)
        self.assertIn("version", data)

    def test_health_endpoint(self):
        """Test the health check endpoint"""
        response = self.app.get("/health")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "healthy")

    def test_get_all_tasks(self):
        """Test getting all tasks"""
        response = self.app.get("/tasks")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("tasks", data)
        self.assertIsInstance(data["tasks"], list)

    def test_get_single_task(self):
        """Test getting a single task"""
        response = self.app.get("/tasks/1")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["id"], 1)

    def test_get_nonexistent_task(self):
        """Test getting a task that doesn't exist"""
        response = self.app.get("/tasks/9999")
        self.assertEqual(response.status_code, 404)
        data = json.loads(response.data)
        self.assertIn("error", data)

    def test_create_task(self):
        """Test creating a new task"""
        new_task = {"title": "Test Task", "completed": False}
        response = self.app.post(
            "/tasks", data=json.dumps(new_task), content_type="application/json"
        )
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data)
        self.assertEqual(data["title"], "Test Task")
        self.assertIn("id", data)

    def test_create_task_without_title(self):
        """Test creating a task without title (should fail)"""
        new_task = {"completed": False}
        response = self.app.post(
            "/tasks", data=json.dumps(new_task), content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn("error", data)

    def test_update_task(self):
        """Test updating a task"""
        update_data = {"title": "Updated Task", "completed": True}
        response = self.app.put(
            "/tasks/1", data=json.dumps(update_data), content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["title"], "Updated Task")
        self.assertEqual(data["completed"], True)

    def test_delete_task(self):
        """Test deleting a task"""
        response = self.app.delete("/tasks/2")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("message", data)

    def test_delete_nonexistent_task(self):
        """Test deleting a task that doesn't exist"""
        response = self.app.delete("/tasks/9999")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
