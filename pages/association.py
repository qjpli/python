import tkinter as tk
from tkinter import messagebox, ttk
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

class AssociationTab:
    def __init__(self, notebook, df):
        self.df = df
        self.create_tab(notebook)

    def create_tab(self, notebook):
        # Main frame for the tab
        main_frame = tk.Frame(notebook)
        notebook.add(main_frame, text="Association")

        # Scrollable frame setup
        canvas = tk.Canvas(main_frame)
        scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas)

        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        self.scrollable_window = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Pack the canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Add Widgets
        title_label = tk.Label(scroll_frame, text="Association Analysis", font=('helvetica', 16, 'bold'))
        title_label.pack(pady=15)

        # Minimum Support
        support_label = tk.Label(scroll_frame, text="Enter Minimum Support (0.0 - 1.0):", font=('helvetica', 12))
        support_label.pack(anchor="w", padx=10)
        
        self.support_entry = tk.Entry(scroll_frame, font=('helvetica', 12), width=50)
        self.support_entry.pack(fill="x", padx=10, pady=5)

        # Generate Rules Button
        generate_button = tk.Button(scroll_frame, text='Generate Rules', command=self.generate_rules, bg='purple', fg='white', font=('helvetica', 12, 'bold'))
        generate_button.pack(pady=15)

        # Results Label
        self.results_label = tk.Label(scroll_frame, text="Results will appear below:", font=('helvetica', 12), wraplength=600, justify="left")
        self.results_label.pack(padx=10, pady=10)

        # Treeview for displaying results in a table
        self.tree = ttk.Treeview(scroll_frame, columns=('Antecedents', 'Consequents', 'Support', 'Confidence'), show='headings')
        self.tree.heading('Antecedents', text='Antecedents')
        self.tree.heading('Consequents', text='Consequents')
        self.tree.heading('Support', text='Support')
        self.tree.heading('Confidence', text='Confidence')

        # Set column widths
        self.tree.column('Antecedents', width=150)
        self.tree.column('Consequents', width=150)
        self.tree.column('Support', width=100)
        self.tree.column('Confidence', width=100)

        self.tree.pack(fill="both", expand=True)

        # Spacer for better scrolling experience
        spacer = tk.Frame(scroll_frame, height=50)
        spacer.pack()

        # Handle dynamic resizing
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(self.scrollable_window, width=e.width))

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

            # Clear the table before displaying new results
            for row in self.tree.get_children():
                self.tree.delete(row)

            # Display results
            if rules.empty:
                self.results_label.config(text="No rules found with the given support.")
            else:
                for _, rule in rules.iterrows():
                    antecedents = ', '.join(list(rule['antecedents']))
                    consequents = ', '.join(list(rule['consequents']))
                    support = rule['support']
                    confidence = rule['confidence']

                    # Insert the rule into the table
                    self.tree.insert('', 'end', values=(antecedents, consequents, f'{support:.3f}', f'{confidence:.3f}'))

        except Exception as e:
            messagebox.showerror("Error", str(e))
