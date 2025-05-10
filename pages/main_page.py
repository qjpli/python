import tkinter as tk
from tkinter import Button, filedialog, ttk, messagebox
import os
import pandas as pd
from pages.clustering import ClusteringTab
from pages.regression import RegressionTab
from pages.association import AssociationTab
from pages.classification import ClassificationTab


class MainPage:
    def __init__(self, root):
        self.root = root
        self.df = None
        self.create_main_page()

    def create_main_page(self):
        self.root.title("Machine Learning Dashboard")

        # Get screen width and height
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Set the window size to the maximum available size, minus a small margin (for window borders, etc.)
        self.root.geometry(f"{screen_width}x{screen_height}+0+0")

        label = tk.Label(self.root, text="Upload Your Data File", font=('helvetica', 16))
        label.pack(pady=20)
        
        upload_button = Button(self.root, text="Upload File", command=self.upload_file, bg='green', fg='white', font=('helvetica', 12, 'bold'))
        upload_button.pack(pady=10)

        # Allow the root window to resize based on the content
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

    def upload_file(self):
        import_file_path = filedialog.askopenfilename()
        ext = os.path.splitext(import_file_path)[-1].lower()

        if ext == '.csv':
            self.df = pd.read_csv(import_file_path)
        elif ext in ['.xls', '.xlsx']:
            self.df = pd.read_excel(import_file_path, engine='openpyxl')
        else:
            messagebox.showerror("Invalid File", "Please upload a CSV or Excel file.")
            return
        
        self.open_dashboard()

    def open_dashboard(self):
        # Clear the existing widgets in the root window before displaying the dashboard
        for widget in self.root.winfo_children():
            widget.destroy()

        # Create a notebook widget (Tab Container)
        notebook = ttk.Notebook(self.root)
        notebook.grid(row=0, column=0, sticky="nsew")  # Grid with expansion in all directions
        
        # Add tabs to the notebook
        ClassificationTab(notebook, self.df)
        RegressionTab(notebook, self.df)
        AssociationTab(notebook, self.df)
        ClusteringTab(notebook, self.df)
        
        # Allow the window to expand when the notebook content is updated
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
