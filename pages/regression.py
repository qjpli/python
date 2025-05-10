import tkinter as tk
from tkinter import messagebox, ttk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class RegressionTab:
    def __init__(self, notebook, df):
        self.df = df
        self.create_tab(notebook)

    def create_tab(self, notebook):
        regression_frame = tk.Frame(notebook)
        notebook.add(regression_frame, text="Regression")

        label = tk.Label(regression_frame, text="Regression Model", font=('helvetica', 14))
        label.pack(pady=10)
        
        # Feature Selection
        self.feature_list = list(self.df.columns)
        features_label = tk.Label(regression_frame, text="Select Features (comma-separated):")
        features_label.pack()
        self.features_entry = tk.Entry(regression_frame)
        self.features_entry.pack(pady=5)

        # Target Selection
        target_label = tk.Label(regression_frame, text="Select Target Column:")
        target_label.pack()
        self.target_dropdown = ttk.Combobox(regression_frame, values=self.feature_list)
        self.target_dropdown.pack(pady=5)

        # Train Button
        train_button = tk.Button(regression_frame, text='Train Model', command=self.train_model, bg='green', fg='white', font=('helvetica', 10, 'bold'))
        train_button.pack(pady=10)

        # Results Label
        self.results_label = tk.Label(regression_frame, text="", font=('helvetica', 10))
        self.results_label.pack(pady=10)

        # Interpretation Label
        self.interpretation_label = tk.Label(regression_frame, text="", font=('helvetica', 10), wraplength=400, justify='left')
        self.interpretation_label.pack(pady=10)

        # Allow window to resize when content grows
        regression_frame.pack_propagate(False)  # Prevent resizing of frame to fit children
        self.canvas_frame = tk.Frame(regression_frame)  # Frame to contain the plot
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

    def train_model(self):
        try:
            # Extract features and target
            features = [f.strip() for f in self.features_entry.get().split(",")]
            target = self.target_dropdown.get()

            # Check if selections are valid
            if not features or target == "" or target not in self.df.columns:
                messagebox.showerror("Error", "Please select valid features and target.")
                return

            # Prepare data for training
            X = self.df[features]
            y = self.df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Train the model
            model = LinearRegression()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            # Calculate metrics
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            # Plot the results
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(y_test, predictions)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
            ax.set_xlabel("True Values")
            ax.set_ylabel("Predictions")
            ax.set_title("True vs Predicted")

            # Embed the plot in Tkinter
            for widget in self.canvas_frame.winfo_children():
                widget.destroy()  # Clear previous plot

            canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Display results
            self.results_label.config(text=f"MSE: {mse:.2f}\nR²: {r2:.2f}")

            # Interpretation Logic
            if r2 >= 0.9:
                comment = "This is an excellent fit. The model explains almost all variability in the data."
            elif r2 >= 0.7:
                comment = "This is a good fit. The model explains most of the variability."
            elif r2 >= 0.5:
                comment = "This is a moderate fit. The model explains some of the variability."
            else:
                comment = "This is a poor fit. Consider using more features, a different model, or data preprocessing."

            interpretation = f"R² Score: {r2:.2f} → {comment}"
            self.interpretation_label.config(text=interpretation)

        except Exception as e:
            messagebox.showerror("Error", str(e))
