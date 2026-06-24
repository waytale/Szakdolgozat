import numpy as np
import tkinter as tk

# ==============================================================================
# 1. ACTIVATION FUNCTIONS
# ==============================================================================
def sigmoid(x):
    """
    Sigmoid activation function.
    Maps any real value into the range [0, 1].
    Useful for predicting probabilities and introducing non-linearity to the network,
    allowing it to learn complex patterns.
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """
    Derivative of the sigmoid function.
    Used during backpropagation to calculate how much we need to adjust the weights.
    Note: 'x' here assumes the input is ALREADY the output of the sigmoid function.
    """
    return x * (1 - x)


# ==============================================================================
# 2. THE NEURAL NETWORK CLASS
# ==============================================================================
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the neural network structure and random weights.
        """
        # --- Architecture Setup ---
        self.input_size = input_size     # Number of input features
        self.hidden_size = hidden_size   # Number of neurons in the hidden layer
        self.output_size = output_size   # Number of output neurons

        # --- Weight & Bias Initialization ---
        # Weights are initialized randomly, usually between -1 and 1.
        
        # 1. Connections between Input Layer and Hidden Layer
        # Shape: (input_features, hidden_neurons)
        self.weights_input_hidden = np.random.uniform(-1, 1, (self.input_size, self.hidden_size))
        # Each hidden neuron gets a bias (an extra trainable constant)
        self.bias_hidden = np.random.uniform(-1, 1, (1, self.hidden_size))

        # 2. Connections between Hidden Layer and Output Layer
        # Shape: (hidden_neurons, output_neurons)
        self.weights_hidden_output = np.random.uniform(-1, 1, (self.hidden_size, self.output_size))
        # Output neuron bias
        self.bias_output = np.random.uniform(-1, 1, (1, self.output_size))


    def feedforward(self, X):
        """
        The Forward Pass: Data flows from inputs, through hidden layers, to the output.
        """
        # Step 1: Calculate input to hidden layer (Dot product of inputs & weights + bias)
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        
        # Step 2: Apply activation function (Sigmoid) to get the hidden layer's real output
        self.hidden_output = sigmoid(self.hidden_input)

        # Step 3: Calculate input to the output layer
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        
        # Step 4: Apply activation function to get the final network prediction
        self.final_output = sigmoid(self.final_input)
        
        return self.final_output


    def backpropagation(self, X, y, learning_rate):
        """
        The Backward Pass: Calculate errors and update weights/biases (Gradient Descent).
        """
        # --- Error Calculation ---
        # Difference between actual target (y) and our prediction (final_output)
        error = y - self.final_output

        # --- Output Layer Gradients ---
        # How much did the output layer contribute to the error?
        # (Error multiplied by the slope/derivative of the output)
        d_output = error * sigmoid_derivative(self.final_output)

        # --- Hidden Layer Gradients ---
        # How much did the hidden layer contribute to the output error?
        # We find this by multiplying the output error by the weights connecting them.
        error_hidden_layer = d_output.dot(self.weights_hidden_output.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(self.hidden_output)

        # --- Update Weights & Biases ---
        # 1. Update Hidden-to-Output weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(d_output) * learning_rate
        self.bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate

        # 2. Update Input-to-Hidden weights and biases
        self.weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate
        self.bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate


    def train(self, X, y, epochs, learning_rate=0.1):
        """
        Trains the neural network by repeating the feedforward and backpropagation steps.
        """
        for epoch in range(epochs):
            # 1. Predict outputs based on current weights
            self.feedforward(X)
            
            # 2. Update weights to fix the prediction error
            self.backpropagation(X, y, learning_rate)

            # 3. Print progress every 1000 epochs to monitor the loss (error)
            if (epoch % 1000) == 0:
                loss = np.mean(np.square(y - self.final_output)) # Mean Squared Error (MSE)
                print(f"Epoch {epoch} - Error (Loss): {loss:.5f}")


# ==============================================================================
# 3. VISUAL GUI APP
# ==============================================================================
class NeuralNetVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Network XOR Visualizer")
        self.root.geometry("400x530")
        
        # --- Network Setup ---
        # 2 inputs, 4 hidden neurons, 1 output
        self.nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
        self.epoch_count = 0
        
        # XOR Dataset
        self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y = np.array([[0], [1], [1], [0]])

        # --- GUI Elements ---
        self.title_label = tk.Label(root, text="Network Decision Boundary MAP", font=("Helvetica", 14, "bold"))
        self.title_label.pack(pady=10)

        # The canvas where we will draw the network's spatial predictions
        self.canvas_size = 300
        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, bg="white", highlightthickness=1, highlightbackground="black")
        self.canvas.pack()

        self.info_label = tk.Label(root, text="Epochs: 0 | Loss: N/A", font=("Helvetica", 11))
        self.info_label.pack(pady=5)

        # Buttons Frame
        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack(pady=5)

        # Train button
        self.train_btn = tk.Button(self.btn_frame, text="Train 2000 Epochs", font=("Helvetica", 11), bg="#4CAF50", fg="white", command=self.train_step)
        self.train_btn.pack(side=tk.LEFT, padx=5)

        # Reset button
        self.reset_btn = tk.Button(self.btn_frame, text="Reset", font=("Helvetica", 11), bg="#f44336", fg="white", command=self.reset_network)
        self.reset_btn.pack(side=tk.LEFT, padx=5)

        # Draw the initial untrained state
        self.draw_decision_boundary()


    def train_step(self):
        """
        Runs when the Train button is clicked. 
        Trains the network for more epochs and updates the UI visualization.
        """
        epochs_to_train = 2000
        self.nn.train(self.X, self.y, epochs=epochs_to_train, learning_rate=0.2)
        self.epoch_count += epochs_to_train
        
        # Calculate current loss/error over the dataset
        predictions = self.nn.feedforward(self.X)
        loss = np.mean(np.square(self.y - predictions))
        
        # Update text label
        self.info_label.config(text=f"Epochs: {self.epoch_count} | Loss: {loss:.5f}")
        
        # Redraw the visual map
        self.draw_decision_boundary()


    def reset_network(self):
        """
        Resets the neural network to random weights to start over.
        """
        self.nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
        self.epoch_count = 0
        self.info_label.config(text="Epochs: 0 | Loss: N/A")
        self.draw_decision_boundary()


    def draw_decision_boundary(self):
        """
        Scans through coordinates from 0.0 to 1.0 (X and Y), gets the network's
        prediction, and turns it into a red/blue heatmap.
        """
        self.canvas.delete("all")
        step = 15 # Set square pixel size (resolution)
        
        # 1. Draw the background grid representing network predictions
        for i in range(0, self.canvas_size, step):
            for j in range(0, self.canvas_size, step):
                # Map canvas UI coordinates to input coordinates (0.0 to 1.0)
                # Flip the Y axis since UI Y goes down but math Y goes up
                x1 = i / self.canvas_size
                x2 = (self.canvas_size - j) / self.canvas_size
                
                # Make the network guess which color this coordinate should be
                pred = self.nn.feedforward(np.array([[x1, x2]]))[0][0]
                
                # Color mapping: Mostly 0 (Red) to mostly 1 (Blue)
                r = int((1 - pred) * 255)
                b = int(pred * 255)
                color = f'#{r:02x}40{b:02x}' # Small amount of green (40) added for a softer tone
                
                self.canvas.create_rectangle(i, j, i+step, j+step, fill=color, outline="")

        # 2. Draw the 4 XOR Target points on top to show what we are trying to achieve
        points = [
            (0, 0, "#ff4d4d"), # 0,0 -> 0 (Red)
            (0, 1, "#4d4dff"), # 0,1 -> 1 (Blue)
            (1, 0, "#4d4dff"), # 1,0 -> 1 (Blue)
            (1, 1, "#ff4d4d")  # 1,1 -> 0 (Red)
        ]
        
        radius = 8
        for px, py, p_color in points:
            cx = px * self.canvas_size
            cy = self.canvas_size - (py * self.canvas_size) # Flip Y
            
            # Keep points visually clamped inside canvas borders
            if cx == 0: cx += radius + 2
            if cy == 0: cy += radius + 2
            if cx == self.canvas_size: cx -= radius + 2
            if cy == self.canvas_size: cy -= radius + 2
            
            self.canvas.create_oval(cx-radius, cy-radius, cx+radius, cy+radius, fill=p_color, outline="white", width=2)


if __name__ == "__main__":
    # Launch standard Tkinter Window
    root = tk.Tk()
    app = NeuralNetVisualizer(root)
    root.mainloop()
