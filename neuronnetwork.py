import tkinter as tk
from tkinter import messagebox, scrolledtext
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SimpleMLP(nn.Module):
    def __init__(self, in_dim=784, hidden=64, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def build_model_report(model):
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    report_lines = [
        "SimpleMLP model info:",
        str(model),
        f"Total parameters: {parameter_count}",
    ]
    return "\n".join(report_lines)


def interval_linear(lower, upper, weight, bias):
    positive_weight = torch.clamp(weight, min=0)
    negative_weight = torch.clamp(weight, max=0)
    lower_out = torch.matmul(lower, positive_weight.t()) + torch.matmul(upper, negative_weight.t()) + bias
    upper_out = torch.matmul(upper, positive_weight.t()) + torch.matmul(lower, negative_weight.t()) + bias
    return lower_out, upper_out


def relu_interval(lower, upper):
    return torch.relu(lower), torch.relu(upper)


def interval_forward(model, lower, upper):
    lower, upper = interval_linear(lower, upper, model.fc1.weight, model.fc1.bias)
    lower, upper = relu_interval(lower, upper)

    lower, upper = interval_linear(lower, upper, model.fc2.weight, model.fc2.bias)
    lower, upper = relu_interval(lower, upper)

    lower, upper = interval_linear(lower, upper, model.fc3.weight, model.fc3.bias)
    return lower, upper


def build_interval_report(model, sample_image):
    flattened = sample_image.view(1, -1)
    lower = torch.clamp(flattened - 0.1, min=0.0, max=1.0)
    upper = torch.clamp(flattened + 0.1, min=0.0, max=1.0)
    interval_lower, interval_upper = interval_forward(model, lower, upper)

    report_lines = [
        "Interval arithmetic info:",
        "Input interval: x ± 0.1, clipped to [0, 1]",
        f"Lower bounds shape: {interval_lower.shape}",
        f"Upper bounds shape: {interval_upper.shape}",
        f"Output lower bounds: {interval_lower.squeeze().tolist()}",
        f"Output upper bounds: {interval_upper.squeeze().tolist()}",
    ]
    return "\n".join(report_lines)


def dataset():
    """Loads the MNIST dataset and returns data loaders for training and testing."""
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=128)
    return train_ds, test_ds, train_loader, test_loader


def build_dataset_report(train_ds, test_ds, train_loader, test_loader):
    report_lines = [
        "MNIST dataset info:",
        f"Train dataset size: {len(train_ds)}",
        f"Test dataset size: {len(test_ds)}",
        f"Train loader size: {len(train_loader)}",
        f"Test loader size: {len(test_loader)}",
        f"Input shape: {train_ds[0][0].shape}",
        f"Label shape: {train_ds[0][1]}",
        f"Input shape (flattened): {train_ds[0][0].view(-1).shape}",
    ]
    return "\n".join(report_lines)


def build_test_report():
    report_lines = [
        "Test info:",
        "test_neuronnetwork.py",
        "Run command: /home/waytale/Desktop/Uni/Szakdolgozat/venv/bin/python -m unittest discover -v",
        "Last validation: 6 tests passed",
    ]
    return "\n".join(report_lines)


def evaluate_model(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.view(images.size(0), -1)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100.0 * correct / total if total else 0.0


def train_model(model, train_loader, test_loader, epochs=1, learning_rate=0.001, progress_callback=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history_lines = []

    for epoch_index in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_index, (images, labels) in enumerate(train_loader, start=1):
            images = images.view(images.size(0), -1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs, 1)
            total += batch_size
            correct += (predicted == labels).sum().item()

            if progress_callback is not None and (batch_index % 50 == 0 or batch_index == len(train_loader)):
                progress_callback(epoch_index + 1, batch_index, len(train_loader), loss.item())

        train_loss = running_loss / total if total else 0.0
        train_accuracy = 100.0 * correct / total if total else 0.0
        test_accuracy = evaluate_model(model, test_loader)

        history_lines.extend(
            [
                f"Epoch {epoch_index + 1}/{epochs}",
                f"Train loss: {train_loss:.4f}",
                f"Train accuracy: {train_accuracy:.2f}%",
                f"Test accuracy: {test_accuracy:.2f}%",
            ]
        )

    return "\n".join(history_lines)


class NeuronNetworkApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Neuron Network")
        self.root.geometry("760x560")
        self.model = SimpleMLP()

        header = tk.Label(
            self.root,
            text="Neuron Network",
            font=("Arial", 18, "bold"),
        )
        header.pack(pady=(16, 8))

        self.status = tk.Label(
            self.root,
            text="Open the buttons to inspect the model and MNIST dataset.",
            anchor="w",
            justify="left",
        )
        self.status.pack(fill="x", padx=16, pady=(0, 8))

        button_row = tk.Frame(self.root)
        button_row.pack(fill="x", padx=16, pady=(0, 8))

        self.dataset_button = tk.Button(button_row, text="Load dataset info", command=self.show_dataset_info)
        self.dataset_button.pack(side="left")

        self.model_button = tk.Button(button_row, text="Show model info", command=self.show_model_info)
        self.model_button.pack(side="left", padx=(8, 0))

        self.train_button = tk.Button(button_row, text="Train model", command=self.train_network)
        self.train_button.pack(side="left", padx=(8, 0))

        self.interval_button = tk.Button(button_row, text="Interval arithmetic", command=self.show_interval_info)
        self.interval_button.pack(side="left", padx=(8, 0))

        self.tests_button = tk.Button(button_row, text="Show tests", command=self.show_test_info)
        self.tests_button.pack(side="left", padx=(8, 0))

        self.run_tests_button = tk.Button(button_row, text="Run tests", command=self.run_tests)
        self.run_tests_button.pack(side="left", padx=(8, 0))

        self.all_button = tk.Button(button_row, text="Show all info", command=self.show_all_info)
        self.all_button.pack(side="left", padx=(8, 0))

        self.output = scrolledtext.ScrolledText(self.root, wrap="word", height=22)
        self.output.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        self.root.after(100, self.show_welcome)

    def set_status(self, text):
        self.status.config(text=text)

    def write_output(self, text, clear=False):
        if clear:
            self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, text)
        self.output.see(tk.END)

    def show_welcome(self):
        self.write_output(
            "Neuron Network app is ready.\n\n"
            "Use the buttons above to load the dataset info or the model info.",
            clear=True,
        )

    def show_dataset_info(self):
        try:
            self.set_status("Loading MNIST dataset...")
            train_ds, test_ds, train_loader, test_loader = dataset()
            self.write_output(build_dataset_report(train_ds, test_ds, train_loader, test_loader), clear=True)
            self.set_status("MNIST dataset loaded.")
        except Exception as exc:
            messagebox.showerror("Dataset error", str(exc), parent=self.root)
            self.set_status("Dataset loading failed.")

    def show_model_info(self):
        self.set_status("Showing model info...")
        self.write_output(build_model_report(self.model), clear=True)
        self.set_status("Model info shown.")

    def train_network(self):
        try:
            self.set_status("Loading data for training...")
            train_ds, test_ds, train_loader, test_loader = dataset()
            self.write_output("Training SimpleMLP on MNIST...\n", clear=True)

            def progress_callback(epoch, batch_index, total_batches, loss_value):
                self.write_output(
                    f"Epoch {epoch}: batch {batch_index}/{total_batches}, loss {loss_value:.4f}\n"
                )
                self.root.update_idletasks()

            self.set_status("Training model...")
            training_report = train_model(
                self.model,
                train_loader,
                test_loader,
                epochs=1,
                progress_callback=progress_callback,
            )
            self.write_output("\n" + training_report + "\n", clear=False)
            self.set_status("Training finished.")
        except Exception as exc:
            messagebox.showerror("Training error", str(exc), parent=self.root)
            self.set_status("Training failed.")

    def show_interval_info(self):
        try:
            self.set_status("Loading sample for interval arithmetic...")
            train_ds, _, _, _ = dataset()
            sample_image, sample_label = train_ds[0]
            self.write_output(f"Using sample label: {sample_label}\n\n", clear=True)
            self.write_output(build_interval_report(self.model, sample_image), clear=False)
            self.set_status("Interval arithmetic shown.")
        except Exception as exc:
            messagebox.showerror("Interval arithmetic error", str(exc), parent=self.root)
            self.set_status("Interval arithmetic failed.")

    def show_test_info(self):
        self.set_status("Showing test info...")
        self.write_output(build_test_report(), clear=True)
        self.set_status("Test info shown.")

    def run_tests(self):
        try:
            self.set_status("Running tests...")
            self.write_output("Running unit tests...\n", clear=True)
            result = subprocess.run(
                [sys.executable, "-m", "unittest", "discover", "-v"],
                cwd=Path(__file__).resolve().parent,
                capture_output=True,
                text=True,
                check=False,
            )
            output = (result.stdout or "") + (result.stderr or "")
            self.write_output(output if output else "No test output returned.\n", clear=True)
            if result.returncode == 0:
                self.set_status("Tests finished successfully.")
            else:
                self.set_status("Tests finished with errors.")
        except Exception as exc:
            messagebox.showerror("Test run error", str(exc), parent=self.root)
            self.set_status("Test run failed.")

    def show_all_info(self):
        try:
            self.set_status("Loading model and dataset info...")
            train_ds, test_ds, train_loader, test_loader = dataset()
            combined_report = [
                build_model_report(self.model),
                "",
                build_dataset_report(train_ds, test_ds, train_loader, test_loader),
                "",
                build_interval_report(self.model, train_ds[0][0]),
                "",
                build_test_report(),
            ]
            self.write_output("\n".join(combined_report), clear=True)
            self.set_status("All info shown.")
        except Exception as exc:
            messagebox.showerror("Neuron network error", str(exc), parent=self.root)
            self.set_status("Failed to show all info.")

    def run(self):
        self.root.mainloop()


def main():
    app = NeuronNetworkApp()
    app.run()


if __name__ == "__main__":
    main()
