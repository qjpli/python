import tkinter as tk
from tkinter import messagebox, ttk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    cohen_kappa_score, mean_absolute_error,
    mean_squared_error
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np

class ClassificationTab:
    def __init__(self, notebook, df):
        self.df = df
        self.create_tab(notebook)

    def create_tab(self, notebook):
        main_frame = tk.Frame(notebook)
        notebook.add(main_frame, text="Classification")

        canvas = tk.Canvas(main_frame)
        scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas)

        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        self.scrollable_window = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        title_label = tk.Label(scroll_frame, text="Classification Model", font=('helvetica', 16, 'bold'))
        title_label.pack(pady=15)

        self.feature_list = list(self.df.columns)
        features_label = tk.Label(scroll_frame, text="Select Features (comma-separated):", font=('helvetica', 12))
        features_label.pack(anchor="w", padx=10)

        self.features_entry = tk.Entry(scroll_frame, font=('helvetica', 12), width=50)
        self.features_entry.pack(fill="x", padx=10, pady=5)

        target_label = tk.Label(scroll_frame, text="Select Target Column:", font=('helvetica', 12))
        target_label.pack(anchor="w", padx=10)

        self.target_dropdown = ttk.Combobox(scroll_frame, values=self.feature_list, font=('helvetica', 12), width=50)
        self.target_dropdown.pack(fill="x", padx=10, pady=5)

        train_button = tk.Button(scroll_frame, text='Train Model', command=self.train_model, bg='blue', fg='white', font=('helvetica', 12, 'bold'))
        train_button.pack(pady=15)

        self.results_label = tk.Label(scroll_frame, text="", font=('helvetica', 12), wraplength=600, justify="left")
        self.results_label.pack(padx=10, pady=10)

        self.metrics_label = tk.Label(scroll_frame, text="", font=('helvetica', 12), wraplength=600, justify="left")
        self.metrics_label.pack(padx=10, pady=10)

        self.interpretation_label = tk.Label(scroll_frame, text="", font=('helvetica', 12), wraplength=600, justify="left")
        self.interpretation_label.pack(padx=10, pady=10)

        self.canvas_frame = tk.Frame(scroll_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        spacer = tk.Frame(scroll_frame, height=50)
        spacer.pack()

        canvas.bind("<Configure>", lambda e: canvas.itemconfig(self.scrollable_window, width=e.width))

    def train_model(self):
        try:
            features = [f.strip() for f in self.features_entry.get().split(",")]
            target = self.target_dropdown.get()

            if not features or target == "" or target not in self.df.columns:
                messagebox.showerror("Error", "Please select valid features and target.")
                return

            X = self.df[features].copy()
            y = self.df[target].copy()

            for col in X.columns:
                if X[col].dtype == 'object':
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))

            def categorize_production(value):
                if value < 5000:
                    return 'Low'
                elif value < 15000:
                    return 'Medium'
                else:
                    return 'High'

            y = y.apply(categorize_production)
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            accuracy = accuracy_score(y_test, predictions)
            report = classification_report(y_test, predictions)

            # Additional Metrics
            correct = np.sum(predictions == y_test)
            incorrect = np.sum(predictions != y_test)
            kappa = cohen_kappa_score(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            mean_actual = np.mean(y_test)
            rae = np.sum(np.abs(y_test - predictions)) / np.sum(np.abs(y_test - mean_actual))
            rrse = np.sqrt(np.sum((y_test - predictions) ** 2) / np.sum((y_test - mean_actual) ** 2))

            metrics_text = (
                f"Correctly Classified Instances: {correct}\n"
                f"Incorrectly Classified Instances: {incorrect}\n"
                f"Kappa Statistic: {kappa:.3f}\n"
                f"Mean Absolute Error: {mae:.3f}\n"
                f"Root Mean Squared Error: {rmse:.3f}\n"
                f"Relative Absolute Error: {rae * 100:.2f}%\n"
                f"Root Relative Squared Error: {rrse * 100:.2f}%\n"
            )

            # Feature Importances
            feature_importances = model.feature_importances_
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(features, feature_importances, color='teal')
            ax.set_xlabel("Features")
            ax.set_ylabel("Importance")
            ax.set_title("Feature Importances")
            plt.xticks(rotation=45, ha="right")
            fig.tight_layout()

            for widget in self.canvas_frame.winfo_children():
                widget.destroy()

            canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            self.results_label.config(text=f"Accuracy: {accuracy:.2f}\n\n{report}")
            self.metrics_label.config(text=metrics_text)

            if accuracy >= 0.9:
                comment = "Excellent classification accuracy! The model is performing very well."
            elif accuracy >= 0.7:
                comment = "Good classification accuracy. The model is performing well."
            elif accuracy >= 0.5:
                comment = "Moderate classification accuracy. The model's performance could be improved."
            else:
                comment = "Low classification accuracy. The model may not be suitable for this data."

            interpretation = f"Accuracy: {accuracy:.2f} → {comment}"
            self.interpretation_label.config(text=interpretation)

        except Exception as e:
            messagebox.showerror("Error", str(e))
