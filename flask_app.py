from flask import Flask, request, jsonify

app = Flask(__name__)

# Define a route to receive input
@app.route('/receive_input', methods=['POST'])
def receive_input():
    data = request.json  # Assuming input data is sent as JSON
    # Process the received data
    # For example, you can print it and return a response
    print("Received input:", data)
    # Perform processing here
    # Return a response
    response_data = {'message': 'Input received successfully'}
    return jsonify(response_data), 200

if __name__ == '__main__':
    app.run(debug=False)  # Run the Flask app without debug mode
