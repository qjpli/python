import tkinter as tk
from tkinter import messagebox
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

class AssociationTab:
    def __init__(self, notebook, df):
        self.df = df
        self.create_tab(notebook)

    def create_tab(self, notebook):
        association_frame = tk.Frame(notebook)
        notebook.add(association_frame, text="Association")

        label = tk.Label(association_frame, text="Association Analysis", font=('helvetica', 14))
        label.pack(pady=10)
        
        # Minimum Support
        support_label = tk.Label(association_frame, text="Enter Minimum Support (0.0 - 1.0):")
        support_label.pack()
        self.support_entry = tk.Entry(association_frame)
        self.support_entry.pack(pady=5)

        # Generate Rules Button
        generate_button = tk.Button(association_frame, text='Generate Rules', command=self.generate_rules, bg='purple', fg='white', font=('helvetica', 10, 'bold'))
        generate_button.pack(pady=10)

        # Results Label
        self.results_label = tk.Label(association_frame, text="", font=('helvetica', 10))
        self.results_label.pack(pady=10)

    def generate_rules(self):
        try:
            # Get minimum support
            min_support = float(self.support_entry.get())
            if not (0.0 <= min_support <= 1.0):
                messagebox.showerror("Error", "Support must be between 0.0 and 1.0.")
                return

            # Convert to one-hot if necessary
            df_encoded = self.df.copy()

            # Convert non-binary data to one-hot (if categorical) or binary (if numeric)
            if not all(df_encoded.dtypes == 'bool'):
                for col in df_encoded.columns:
                    if df_encoded[col].dtype == object:
                        df_encoded = pd.get_dummies(df_encoded, columns=[col])
                    else:
                        # Numeric columns: binarize (presence > 0)
                        df_encoded[col] = df_encoded[col].apply(lambda x: 1 if x > 0 else 0)

            # Ensure it's all bool
            df_encoded = df_encoded.astype(bool)

            # Generate frequent itemsets
            frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

            # Display results
            if rules.empty:
                self.results_label.config(text="No rules found with the given support.")
            else:
                self.results_label.config(text=rules[['antecedents', 'consequents', 'support', 'confidence']].to_string(index=False))

        except Exception as e:
            messagebox.showerror("Error", str(e))
