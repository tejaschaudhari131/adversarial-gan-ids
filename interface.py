import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import logging
import os

logger = logging.getLogger(__name__)


class UserInterface:
    def __init__(self, master):
        self.master = master
        master.title("Adversarial GAN for IDS")
        master.geometry("500x400")
        master.resizable(False, False)

        # Dataset path
        path_frame = tk.Frame(master, padx=10, pady=10)
        path_frame.pack(fill=tk.X)

        tk.Label(path_frame, text="Dataset:").pack(side=tk.LEFT)
        self.dataset_var = tk.StringVar()
        tk.Entry(path_frame, textvariable=self.dataset_var, width=40).pack(side=tk.LEFT, padx=5)
        tk.Button(path_frame, text="Browse", command=self._browse_dataset).pack(side=tk.LEFT)

        # Action buttons
        btn_frame = tk.Frame(master, padx=10, pady=5)
        btn_frame.pack(fill=tk.X)

        self.train_btn = tk.Button(btn_frame, text="Train IDS Model", width=20,
                                   command=self.train_ids)
        self.train_btn.pack(pady=5)

        self.train_gan_btn = tk.Button(btn_frame, text="Train GAN", width=20,
                                       command=self.train_gan)
        self.train_gan_btn.pack(pady=5)

        self.evaluate_btn = tk.Button(btn_frame, text="Evaluate Model", width=20,
                                      command=self.evaluate_model)
        self.evaluate_btn.pack(pady=5)

        # Status / log area
        tk.Label(master, text="Log:").pack(anchor=tk.W, padx=10)
        self.log_text = tk.Text(master, height=12, state=tk.DISABLED, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Progress bar
        self.progress = ttk.Progressbar(master, mode='indeterminate')
        self.progress.pack(fill=tk.X, padx=10, pady=(0, 10))

    def _browse_dataset(self):
        path = filedialog.askopenfilename(
            title="Select dataset CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.dataset_var.set(path)

    def _log(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _set_busy(self, busy):
        state = tk.DISABLED if busy else tk.NORMAL
        self.train_btn.config(state=state)
        self.train_gan_btn.config(state=state)
        self.evaluate_btn.config(state=state)
        if busy:
            self.progress.start()
        else:
            self.progress.stop()

    def _run_in_thread(self, target):
        dataset = self.dataset_var.get().strip()
        if not dataset:
            messagebox.showwarning("No dataset", "Please select a dataset file first.")
            return

        if not os.path.isfile(dataset):
            messagebox.showerror("File not found", f"Dataset not found:\n{dataset}")
            return

        self._set_busy(True)

        def wrapper():
            try:
                target(dataset)
            except Exception as e:
                self.master.after(0, lambda: messagebox.showerror("Error", str(e)))
                self.master.after(0, lambda: self._log(f"ERROR: {e}"))
            finally:
                self.master.after(0, lambda: self._set_busy(False))

        thread = threading.Thread(target=wrapper, daemon=True)
        thread.start()

    def train_ids(self):
        def _train(dataset):
            self.master.after(0, lambda: self._log("Starting advanced IDS training..."))
            from training.train_advanced_ids import train_ids_model
            results = train_ids_model(dataset)
            self.master.after(0, lambda: self._log(
                f"Training complete! Test results: {results['test_results']}"
            ))

        self._run_in_thread(_train)

    def train_gan(self):
        def _train(dataset):
            self.master.after(0, lambda: self._log("Starting GAN training..."))
            from training.train_gan import train_gan
            train_gan(dataset, epochs=1000, log_interval=100)
            self.master.after(0, lambda: self._log("GAN training complete!"))

        self._run_in_thread(_train)

    def evaluate_model(self):
        def _eval(dataset):
            self.master.after(0, lambda: self._log("Evaluating model..."))
            from preprocessing.data_preprocessing import prepare_dataset
            from evaluation.evaluate_model import evaluate_model_performance
            from tensorflow.keras.models import load_model

            model_path = "best_ids_model.h5"
            if not os.path.isfile(model_path):
                raise FileNotFoundError(
                    f"No trained model found at '{model_path}'. Train a model first."
                )

            prepared_data = prepare_dataset(dataset)
            model = load_model(model_path)
            results = evaluate_model_performance(
                model, prepared_data['X_test'], prepared_data['y_test'],
                prepared_data['label_encoder'],
            )
            msg = f"Evaluation complete!"
            if 'roc_auc' in results:
                msg += f" AUC: {results['roc_auc']:.4f}"
            self.master.after(0, lambda: self._log(msg))

        self._run_in_thread(_eval)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    root = tk.Tk()
    ui = UserInterface(root)
    root.mainloop()
