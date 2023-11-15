import tkinter as tk
from tkinter import filedialog
from threading import Thread
from TrainModel import train_gpt2_model
import os

class GPT2TrainerUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.txt_file_path = None
        self.title("GPT-2 Trainer")
        self.geometry("400x300")

        # Upload Button
        self.upload_button = tk.Button(self, text="Upload Text File", command=self.upload_file)
        self.upload_button.pack(pady=10)

        # Train Button
        self.train_button = tk.Button(self, text="Train Model", command=self.start_training, state=tk.DISABLED)
        self.train_button.pack(pady=10)

        # Status Label
        self.status_label = tk.Label(self, text="", font=("Helvetica", 14))
        self.status_label.pack()

    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            self.status_label.config(text=f"File {os.path.basename(file_path)} uploaded")
            self.txt_file_path = file_path
            self.train_button.config(state=tk.NORMAL)

    def start_training(self):
        self.status_label.config(text="Training started...")
        self.train_button.config(state=tk.DISABLED)
        self.upload_button.config(state=tk.DISABLED)

        # Start training in a separate thread
        Thread(target=self.train_model).start()

    def train_model(self):
        train_gpt2_model(self.txt_file_path)
        self.status_label.config(text="Training completed")
        self.upload_button.config(state=tk.NORMAL)

if __name__ == "__main__":
    app = GPT2TrainerUI()
    app.mainloop()
