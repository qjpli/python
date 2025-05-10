import tkinter as tk
from tkinter import messagebox, ttk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

class ClassificationTab:
    def __init__(self, notebook, df):
        self.df = df
        self.create_tab(notebook)

    def create_tab(self, notebook):
        classification_frame = tk.Frame(notebook)
        notebook.add(classification_frame, text="Classification")

        label = tk.Label(classification_frame, text="Classification Model", font=('helvetica', 14))
        label.pack(pady=10)

        # Feature Selection
        self.feature_list = list(self.df.columns)
        self.features_var = tk.StringVar(value=self.feature_list)
        features_label = tk.Label(classification_frame, text="Select Features (comma-separated):")
        features_label.pack()
        self.features_entry = tk.Entry(classification_frame)
        self.features_entry.pack(pady=5)

        # Target Selection
        target_label = tk.Label(classification_frame, text="Select Target Column:")
        target_label.pack()
        self.target_dropdown = ttk.Combobox(classification_frame, values=self.feature_list)
        self.target_dropdown.pack(pady=5)

        # Train Button
        train_button = tk.Button(classification_frame, text='Train Model', command=self.train_model, bg='blue', fg='white', font=('helvetica', 10, 'bold'))
        train_button.pack(pady=10)

        # Results Label
        self.results_label = tk.Label(classification_frame, text="", font=('helvetica', 10))
        self.results_label.pack(pady=10)

    def train_model(self):
        try:
            # Extract features and target
            features = [f.strip() for f in self.features_entry.get().split(",")]
            target = self.target_dropdown.get()

            # Check if selections are valid
            if not features or target == "" or target not in self.df.columns:
                messagebox.showerror("Error", "Please select valid features and target.")
                return

            # Prepare data
            X = self.df[features].copy()
            y = self.df[target].copy()

            # Encode categorical features
            for col in X.columns:
                if X[col].dtype == 'object':
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))

            # Encode target if it's categorical
            if y.dtype == 'object':
                target_encoder = LabelEncoder()
                y = target_encoder.fit_transform(y.astype(str))

            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            # Calculate accuracy
            accuracy = accuracy_score(y_test, predictions)
            report = classification_report(y_test, predictions)

            # Display results
            self.results_label.config(text=f"Accuracy: {accuracy:.2f}\n\n{report}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
