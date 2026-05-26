"""
Simple Flask API for CI/CD Demo
"""
from flask import Flask, jsonify, request

app = Flask(__name__)

# In-memory database (for demo purposes)
tasks = [
    {"id": 1, "title": "Learn CI/CD", "completed": False},
    {"id": 2, "title": "Build Pipeline", "completed": False}
]


@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        "message": "Welcome to CI/CD Demo API",
        "version": "1.0.0",
        "endpoints": {
            "GET /": "This help message",
            "GET /health": "Health check",
            "GET /tasks": "Get all tasks",
            "POST /tasks": "Create a new task",
            "GET /tasks/<id>": "Get a specific task",
            "PUT /tasks/<id>": "Update a task",
            "DELETE /tasks/<id>": "Delete a task"
        }
    })


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200


@app.route('/tasks', methods=['GET'])
def get_tasks():
    """Get all tasks"""
    return jsonify({"tasks": tasks}), 200


@app.route('/tasks/<int:task_id>', methods=['GET'])
def get_task(task_id):
    """Get a specific task"""
    task = next((t for t in tasks if t['id'] == task_id), None)
    if task:
        return jsonify(task), 200
    return jsonify({"error": "Task not found"}), 404


@app.route('/tasks', methods=['POST'])
def create_task():
    """Create a new task"""
    data = request.get_json()

    if not data or 'title' not in data:
        return jsonify({"error": "Title is required"}), 400

    new_task = {
        "id": max([t['id'] for t in tasks]) + 1 if tasks else 1,
        "title": data['title'],
        "completed": data.get('completed', False)
    }
    tasks.append(new_task)
    return jsonify(new_task), 201


@app.route('/tasks/<int:task_id>', methods=['PUT'])
def update_task(task_id):
    """Update a task"""
    task = next((t for t in tasks if t['id'] == task_id), None)
    if not task:
        return jsonify({"error": "Task not found"}), 404

    data = request.get_json()
    task['title'] = data.get('title', task['title'])
    task['completed'] = data.get('completed', task['completed'])
    return jsonify(task), 200


@app.route('/tasks/<int:task_id>', methods=['DELETE'])
def delete_task(task_id):
    """Delete a task"""
    global tasks
    task = next((t for t in tasks if t['id'] == task_id), None)
    if not task:
        return jsonify({"error": "Task not found"}), 404

    tasks = [t for t in tasks if t['id'] != task_id]
    return jsonify({"message": "Task deleted"}), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)