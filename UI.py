import tkinter as tk
from tkinter import filedialog
from LoadModel import load_model
from TrainModel import train_gpt2_model
from threading import Thread
import os
from DownloadGPT_2 import download_gpt2
from Predict import predict

class GPT2App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GPT-2 Application")
        self.geometry("500x700")  # Increased height to accommodate new labels

        models_folder = "Models"
        if not os.path.exists(models_folder):
            os.makedirs(models_folder)
            download_gpt2()

        self.models_folder = models_folder
        self.model_names = os.listdir(models_folder)
        self.model = None
        self.init_model_selection()
        self.init_text_completion()
        self.init_model_training()

    def init_model_selection(self):
        tk.Label(self, text="Model Selection", font=("Helvetica", 14)).pack(pady=5)
        tk.Label(self, text="Choose a model and load it for text completion").pack()

        self.selected_model = tk.StringVar(self)
        self.selected_model.set(self.model_names[0])  # default value
        self.dropdown = tk.OptionMenu(self, self.selected_model, *self.model_names)
        self.dropdown.pack(pady=10)

        self.load_model_button = tk.Button(self, text="Load Model", command=self.load_model)
        self.load_model_button.pack(pady=10)

        self.model_status_label = tk.Label(self, text="", font=("Helvetica", 12))
        self.model_status_label.pack()

    def load_model(self):
        model_name = self.selected_model.get()
        self.model, self.tokenizer = load_model(model_name)
        self.model_status_label.config(text=f"Model {model_name} loaded!")

    def init_text_completion(self):
        self.method_names = ["Best", "Worst", "Coherent Worst"]
        tk.Label(self, text="Text Completion", font=("Helvetica", 14)).pack(pady=5)
        tk.Label(self, text="Select the completion method, enter text and predict the next word").pack()

        self.selected_method = tk.StringVar(self)
        self.selected_method.set(self.method_names[0])  # default value
        self.dropdown = tk.OptionMenu(self, self.selected_method, *self.method_names)
        self.dropdown.pack(pady=10)

        self.entry = tk.Text(self, width=50, height=5, wrap=tk.WORD)
        self.entry.pack(pady=20)

        self.predict_button = tk.Button(self, text="Predict The Next Word", command=self.predict_next_word)
        self.predict_button.pack()

        self.prediction_label = tk.Label(self, text="", font=("Helvetica", 12))
        self.prediction_label.pack()

    def predict_next_word(self):
        if not self.model:
            self.load_model()

        input_text = self.entry.get("1.0", "end-1c")
        if not input_text:
            next_word = "I am"
            self.entry.delete("1.0", tk.END)
            self.entry.insert(tk.END, input_text + next_word)
        else:
            if (input_text[-1] == " "):
                input_text = input_text[:-1]
            next_word = predict(input_text, self.model, self.tokenizer, self.selected_method.get())
            self.entry.delete("1.0", tk.END)
            self.entry.insert(tk.END, input_text + ' ' + next_word)

        self.prediction_label.config(text=f"Next word: {next_word}")


    def init_model_training(self):
        tk.Label(self, text="Model Training", font=("Helvetica", 14)).pack(pady=5)
        tk.Label(self, text="Upload a text file and train a new model").pack()

        self.upload_button = tk.Button(self, text="Upload Text File", command=self.upload_file)
        self.upload_button.pack(pady=10)

        self.train_button = tk.Button(self, text="Train Model", command=self.start_training, state=tk.DISABLED)
        self.train_button.pack(pady=10)

        self.training_status_label = tk.Label(self, text="", font=("Helvetica", 12))
        self.training_status_label.pack()

    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            self.txt_file_path = file_path
            self.train_button.config(state=tk.NORMAL)
            self.training_status_label.config(text=f"File {os.path.basename(file_path)} uploaded")

    def train_model(self):
        train_gpt2_model(self.txt_file_path)
        self.training_status_label.config(text="Training completed")
        self.upload_button.config(state=tk.NORMAL)

    def start_training(self):
        self.training_status_label.config(text="Training started...")
        self.train_button.config(state=tk.DISABLED)
        self.upload_button.config(state=tk.DISABLED)

        Thread(target=self.train_model).start()

if __name__ == "__main__":
    app = GPT2App()
    app.mainloop()
