import tkinter as tk
from tkinter import messagebox, ttk
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

class AssociationTab:
    def __init__(self, notebook, raw_df):
        self.original_df = raw_df
        self.df = None
        self.create_tab(notebook)

    def preprocess_data(self):
        try:
            def bucketize(val):
                if val < 5000:
                    return 'Low'
                elif val < 15000:
                    return 'Med'
                else:
                    return 'High'

            species_encoded = pd.get_dummies(
                self.original_df['Species'].str.replace(r"[^\w]", "_", regex=True),
                prefix="Species"
            )

            annual_cols = [col for col in self.original_df.columns if 'Annual' in col]
            bucketized = self.original_df[annual_cols].applymap(bucketize)

            renamed = bucketized.copy()
            for col in bucketized.columns:
                year = col.split()[0]
                renamed.rename(columns={col: f"{year}_Annual"}, inplace=True)

            annual_encoded = pd.get_dummies(renamed)

            self.df = pd.concat([species_encoded, annual_encoded], axis=1).astype(bool)
            messagebox.showinfo("Success", f"Preprocessing complete. {self.df.shape[1]} features ready.")
            print(self.df)
        except Exception as e:
            messagebox.showerror("Preprocessing Error", str(e))

    def create_tab(self, notebook):
        main_frame = tk.Frame(notebook)
        notebook.add(main_frame, text="Association")

        canvas = tk.Canvas(main_frame)
        scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas)

        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        self.scrollable_window = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        title_label = tk.Label(scroll_frame, text="Association Rule Mining", font=('helvetica', 16, 'bold'))
        title_label.pack(pady=10)

        preprocess_btn = tk.Button(
            scroll_frame, text="Preprocess Raw Data", bg="green", fg="white",
            font=('helvetica', 12, 'bold'), command=self.preprocess_data
        )
        preprocess_btn.pack(pady=5)

        tk.Label(scroll_frame, text="Select Features (comma-separated):", font=('helvetica', 12)).pack(anchor="w", padx=10)
        self.features_entry = tk.Entry(scroll_frame, font=('helvetica', 12), width=50)
        self.features_entry.pack(fill="x", padx=10, pady=5)

        tk.Label(scroll_frame, text="Enter Minimum Support (0.1 - 1.0):", font=('helvetica', 12)).pack(anchor="w", padx=10)
        self.support_entry = tk.Entry(scroll_frame, font=('helvetica', 12), width=50)
        self.support_entry.insert(0, "0.3")
        self.support_entry.pack(fill="x", padx=10, pady=5)

        generate_button = tk.Button(
            scroll_frame, text="Generate Rules", bg="purple", fg="white",
            font=('helvetica', 12, 'bold'), command=self.generate_rules
        )
        generate_button.pack(pady=10)

        self.results_label = tk.Label(scroll_frame, text="Results will appear below:", font=('helvetica', 12), wraplength=600, justify="left")
        self.results_label.pack(pady=5)

        # Updated Treeview with additional columns
        self.tree = ttk.Treeview(scroll_frame, columns=('Antecedents', 'Consequents', 'Support', 'Confidence', 'Lift', 'Conviction'), show='headings')
        self.tree.heading('Antecedents', text='Antecedents')
        self.tree.heading('Consequents', text='Consequents')
        self.tree.heading('Support', text='Support')
        self.tree.heading('Confidence', text='Confidence')
        self.tree.heading('Lift', text='Lift')
        self.tree.heading('Conviction', text='Conviction')
        self.tree.column('Antecedents', width=150)
        self.tree.column('Consequents', width=150)
        self.tree.column('Support', width=80)
        self.tree.column('Confidence', width=90)
        self.tree.column('Lift', width=70)
        self.tree.column('Conviction', width=90)
        self.tree.pack(fill="both", expand=True)

        # Summary Label
        self.summary_label = tk.Label(scroll_frame, text="", font=('helvetica', 12, 'italic'), wraplength=600, justify="left")
        self.summary_label.pack(pady=10)

        tk.Frame(scroll_frame, height=40).pack()
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(self.scrollable_window, width=e.width))

    def generate_rules(self):
        try:
            if self.df is None:
                messagebox.showerror("Error", "Please preprocess the data first.")
                return

            self.results_label.config(text="Generating rules... Please wait.")
            self.tree.delete(*self.tree.get_children())
            self.summary_label.config(text="")

            raw_input = self.features_entry.get()
            input_cols = [col.strip() for col in raw_input.split(',')]
            valid_cols = [col for col in input_cols if col in self.df.columns]

            if not valid_cols:
                messagebox.showerror("Error", "None of the selected features matched available columns. Please check spelling.")
                return

            min_support = float(self.support_entry.get())
            if not (0.1 <= min_support <= 1.0):
                messagebox.showerror("Error", "Support must be between 0.1 and 1.0.")
                return

            if len(valid_cols) > 30:
                messagebox.showwarning("Warning", f"Too many features selected ({len(valid_cols)}). Try selecting fewer.")
                return

            df_subset = self.df[valid_cols].copy()
            frequent_itemsets = apriori(df_subset, min_support=min_support, use_colnames=True)
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

            if rules.empty:
                self.results_label.config(text="No rules found with the given support and features.")
                return

            for _, rule in rules.iterrows():
                self.tree.insert('', 'end', values=(
                    ', '.join(list(rule['antecedents'])),
                    ', '.join(list(rule['consequents'])),
                    f"{rule['support']:.3f}",
                    f"{rule['confidence']:.3f}",
                    f"{rule['lift']:.3f}",
                    f"{rule['conviction']:.3f}"
                ))

            # Display summary
            self.results_label.config(text=f"Found {len(rules)} rules.")
            self.summary_label.config(text=(
                f"Summary:\n"
                f"- Number of Rules: {len(rules)}\n"
                f"- Minimum Confidence: 0.5\n"
                f"- Support Threshold: {min_support}"
            ))

        except Exception as e:
            messagebox.showerror("Error", str(e))
