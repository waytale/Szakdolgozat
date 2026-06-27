import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from neuronnetwork import (
    SimpleMLP,
    build_dataset_report,
    build_model_report,
    evaluate_model,
    interval_forward,
    interval_linear,
    relu_interval,
)


class NeuronNetworkTests(unittest.TestCase):
    def test_build_model_report_includes_parameter_count(self):
        model = SimpleMLP(in_dim=4, hidden=3, out_dim=2)

        report = build_model_report(model)

        self.assertIn("SimpleMLP model info:", report)
        self.assertIn("Total parameters: 35", report)

    def test_interval_linear_computes_expected_bounds(self):
        lower = torch.tensor([[1.0, 2.0]])
        upper = torch.tensor([[3.0, 4.0]])
        weight = torch.tensor([[1.0, -2.0], [-1.0, 3.0]])
        bias = torch.tensor([0.5, -1.5])

        output_lower, output_upper = interval_linear(lower, upper, weight, bias)

        expected_lower = torch.tensor([[-6.5, 1.5]])
        expected_upper = torch.tensor([[-0.5, 9.5]])

        self.assertTrue(torch.allclose(output_lower, expected_lower))
        self.assertTrue(torch.allclose(output_upper, expected_upper))

    def test_relu_interval_clamps_negative_values(self):
        lower = torch.tensor([[-2.0, 1.0]])
        upper = torch.tensor([[3.0, 4.0]])

        relu_lower, relu_upper = relu_interval(lower, upper)

        self.assertTrue(torch.equal(relu_lower, torch.tensor([[0.0, 1.0]])))
        self.assertTrue(torch.equal(relu_upper, torch.tensor([[3.0, 4.0]])))

    def test_interval_forward_preserves_shape_order(self):
        model = SimpleMLP(in_dim=4, hidden=3, out_dim=2)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(0.25)

        lower = torch.zeros((1, 4))
        upper = torch.ones((1, 4))

        output_lower, output_upper = interval_forward(model, lower, upper)

        self.assertEqual(output_lower.shape, torch.Size([1, 2]))
        self.assertEqual(output_upper.shape, torch.Size([1, 2]))
        self.assertTrue(torch.all(output_lower <= output_upper))

    def test_build_dataset_report_uses_dataset_shapes(self):
        train_images = torch.zeros((5, 1, 28, 28))
        train_labels = torch.tensor([0, 1, 2, 3, 4])
        test_images = torch.zeros((2, 1, 28, 28))
        test_labels = torch.tensor([7, 8])

        train_ds = TensorDataset(train_images, train_labels)
        test_ds = TensorDataset(test_images, test_labels)
        train_loader = DataLoader(train_ds, batch_size=2)
        test_loader = DataLoader(test_ds, batch_size=1)

        report = build_dataset_report(train_ds, test_ds, train_loader, test_loader)

        self.assertIn("Train dataset size: 5", report)
        self.assertIn("Test dataset size: 2", report)
        self.assertIn("Train loader size: 3", report)
        self.assertIn("Test loader size: 2", report)
        self.assertIn("Input shape: torch.Size([1, 28, 28])", report)
        self.assertIn("Label shape: 0", report)

    def test_evaluate_model_returns_accuracy_for_constant_predictions(self):
        model = SimpleMLP(in_dim=4, hidden=3, out_dim=2)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()

        images = torch.zeros((4, 1, 2, 2))
        labels = torch.zeros(4, dtype=torch.long)
        loader = DataLoader(TensorDataset(images, labels), batch_size=2)

        accuracy = evaluate_model(model, loader)

        self.assertEqual(accuracy, 100.0)


if __name__ == "__main__":
    unittest.main()