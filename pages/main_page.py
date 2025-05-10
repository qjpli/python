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
        # Set the window title and maximize size
        self.root.title("Machine Learning Dashboard")
        # self.root.state("zoomed")  # Start in maximized mode
        self.root.geometry("1200x700")  # Set initial window size to 1200x700

        # Main Frame (Center the upload section)
        main_frame = tk.Frame(self.root)
        main_frame.grid(row=0, column=0, sticky="nsew")

        # Center the main frame
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Set the main frame to expand fully
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        # Upload Section
        upload_section = tk.Frame(main_frame)
        upload_section.grid(row=0, column=0)

        # Center the upload section
        upload_section.grid_rowconfigure(0, weight=1)
        upload_section.grid_columnconfigure(0, weight=1)

        # Upload Label
        label = tk.Label(upload_section, text="Upload Your Data File", font=('helvetica', 18, 'bold'))
        label.pack(pady=20)

        # Upload Button
        upload_button = Button(
            upload_section, 
            text="Upload File", 
            command=self.upload_file, 
            bg='green', 
            fg='white', 
            font=('helvetica', 14, 'bold'),
            width=15,
            height=2
        )
        upload_button.pack(pady=20)

    def upload_file(self):
        # Open file dialog for selecting the data file
        import_file_path = filedialog.askopenfilename(
            title="Select a Data File",
            filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xls *.xlsx")]
        )

        # Check if a file was selected
        if not import_file_path:
            return

        try:
            # Determine file type and read accordingly
            ext = os.path.splitext(import_file_path)[-1].lower()

            if ext == '.csv':
                self.df = pd.read_csv(import_file_path)
            elif ext in ['.xls', '.xlsx']:
                self.df = pd.read_excel(import_file_path, engine='openpyxl')
            else:
                raise ValueError("Invalid file format. Please upload a CSV or Excel file.")

            # Open the dashboard if file is valid
            self.open_dashboard()

        except Exception as e:
            messagebox.showerror("File Error", f"Failed to load the file. Error: {e}")

    def open_dashboard(self):
        # Clear the root window before displaying the dashboard
        for widget in self.root.winfo_children():
            widget.destroy()

        # Create a notebook widget (Tab Container)
        notebook = ttk.Notebook(self.root)
        notebook.grid(row=0, column=0, sticky="nsew")
        
        # Add tabs to the notebook
        ClassificationTab(notebook, self.df)
        RegressionTab(notebook, self.df)
        AssociationTab(notebook, self.df)
        ClusteringTab(notebook, self.df)
        
        # Allow the notebook to expand fully
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
