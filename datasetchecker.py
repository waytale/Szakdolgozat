from pathlib import Path
import shutil
import tkinter as tk
from tkinter import messagebox, scrolledtext
import numpy as np


ROOT_DIR = Path("data/MNIST/raw")
MNIST_FILES = {
    "train_images": ROOT_DIR / "train-images-idx3-ubyte",
    "train_labels": ROOT_DIR / "train-labels-idx1-ubyte",
    "test_images": ROOT_DIR / "t10k-images-idx3-ubyte",
    "test_labels": ROOT_DIR / "t10k-labels-idx1-ubyte",
}

EXPECTED_REPORT = (
    "data/MNIST/raw/train-images-idx3-ubyte:\n"
    " magic: 2051\n"
    " count: 60000\n"
    " size: 28 x 28\n"
    "\n"
    "data/MNIST/raw/train-labels-idx1-ubyte:\n"
    " magic: 2049\n"
    " count: 60000\n"
    "\n"
    "data/MNIST/raw/t10k-images-idx3-ubyte:\n"
    " magic: 2051\n"
    " count: 10000\n"
    " size: 28 x 28\n"
    "\n"
    "data/MNIST/raw/t10k-labels-idx1-ubyte:\n"
    " magic: 2049\n"
    " count: 10000\n"
    "\n"
    "train pixel range: 0 .. 255"
)




def check_images(path):
    with open(path, "rb") as file_handle:
        magic, num, rows, cols = np.fromfile(file_handle, dtype=">i4", count=4)

    return (
        f"{path}:\n"
        f" magic: {magic}\n"
        f" count: {num}\n"
        f" size: {rows} x {cols}\n"
    )

def check_labels(path):
    with open(path, "rb") as file_handle:
        magic, num = np.fromfile(file_handle, dtype=">i4", count=2)

    return f"{path}:\n magic: {magic}\n count: {num}\n"

def dataset_exists():
    return all(path.exists() for path in MNIST_FILES.values())






def download_mnist():
    """Downloads the MNIST dataset using torchvision."""
    from torchvision import datasets, transforms

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    return train_dataset, test_dataset


def refresh_mnist():
    if ROOT_DIR.exists():
        shutil.rmtree(ROOT_DIR)
    return download_mnist()


def build_report():
    if not dataset_exists():
        return (
            "MNIST raw files are missing.\n"
            "Choose download to fetch the dataset again.\n"
        )

    images = np.fromfile(MNIST_FILES["train_images"], dtype=np.uint8)[16:]
    report_lines = [
        check_images(MNIST_FILES["train_images"]).rstrip(),
        check_labels(MNIST_FILES["train_labels"]).rstrip(),
        check_images(MNIST_FILES["test_images"]).rstrip(),
        check_labels(MNIST_FILES["test_labels"]).rstrip(),
        f"train pixel range: {int(images.min())} .. {int(images.max())}",
    ]
    return "\n\n".join(report_lines)




def dataset_matches_expected():
    return build_report() == EXPECTED_REPORT


class MnistCheckerApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("MNIST Checker")
        self.root.geometry("720x520")
        self.report_animation_step = 0
        self.report_animation_job = None
        self.report_finish_job = None

        header = tk.Label(
            self.root,
            text="MNIST Checker",
            font=("Arial", 18, "bold"),
        )
        header.pack(pady=(16, 8))

        self.status = tk.Label(
            self.root,
            text="Launch the app to inspect the MNIST dataset.",
            anchor="w",
            justify="left",
        )
        self.status.pack(fill="x", padx=16, pady=(0, 8))

        button_row = tk.Frame(self.root)
        button_row.pack(fill="x", padx=16, pady=(0, 8))

        self.download_button = tk.Button(button_row, text="Download MNIST again", command=self.download_again)
        self.download_button.pack(side="left")

        self.refresh_button = tk.Button(button_row, text="Check current MNIST", command=self.start_report_animation)
        self.refresh_button.pack(side="left", padx=(8, 0))

        self.output = scrolledtext.ScrolledText(self.root, wrap="word", height=20)
        self.output.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        self.root.after(100, self.show_report)

    def set_status(self, text):
        self.status.config(text=text)

    def write_output(self, text, clear=False):
        if clear:
            self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, text)
        self.output.see(tk.END)

    def cancel_report_animation(self):
        if self.report_animation_job is not None:
            self.root.after_cancel(self.report_animation_job)
            self.report_animation_job = None
        if self.report_finish_job is not None:
            self.root.after_cancel(self.report_finish_job)
            self.report_finish_job = None

    def set_refresh_controls(self, enabled):
        state = tk.NORMAL if enabled else tk.DISABLED
        self.refresh_button.config(state=state)
        self.download_button.config(state=state)

    def start_report_animation(self):
        self.cancel_report_animation()
        self.report_animation_step = 0
        self.set_refresh_controls(False)
        self.set_status("Checking current MNIST...")
        self.write_output("Checking current MNIST", clear=True)
        self.animate_report_text()
        self.report_finish_job = self.root.after(3000, self.finish_report_animation)

    def animate_report_text(self):
        dots = "." * (self.report_animation_step % 4)
        self.write_output(f"Checking current MNIST{dots}\n", clear=True)
        self.report_animation_step += 1
        self.report_animation_job = self.root.after(1000, self.animate_report_text)

    def finish_report_animation(self):
        self.cancel_report_animation()
        self.show_report()
        self.set_refresh_controls(True)

    def download_again(self):
        try:
            self.cancel_report_animation()
            self.set_status("Refreshing MNIST dataset...")
            self.write_output("Refreshing MNIST dataset...\n", clear=True)
            train_dataset, test_dataset = refresh_mnist()
            self.write_output(
                f"Downloaded train samples: {len(train_dataset)}\n"
                f"Downloaded test samples: {len(test_dataset)}\n\n"
            )
            self.write_output("Download finished successfully.\n\n")
            self.show_report()
            self.set_status("MNIST dataset is ready.")
        except Exception as exc:
            messagebox.showerror("MNIST download failed", str(exc), parent=self.root)
            self.set_status("MNIST download failed.")
        finally:
            self.set_refresh_controls(True)

    def show_report(self):
        self.set_status("Checking MNIST files...")
        report = build_report()
        if report == EXPECTED_REPORT:
            self.write_output(report + "\n\nSuccess: MNIST dataset matches the expected result.", clear=True)
        else:
            self.write_output(report + "Warning: the dataset is broken.\n\n", clear=True)
        self.set_status("MNIST check finished.")

    def run(self):
        self.root.mainloop()


def run_cli_fallback():
    print(build_report())


def main():
    try:
        app = MnistCheckerApp()
        app.run()
    except tk.TclError:
        run_cli_fallback()


if __name__ == "__main__":
    main()

