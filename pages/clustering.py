import tkinter as tk
from tkinter import Button, messagebox, ttk
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ClusteringTab:
    def __init__(self, notebook, df):
        self.df = df
        self.canvas = None  # To keep track of the existing canvas
        self.create_tab(notebook)

    def create_tab(self, notebook):
        # Main frame for the tab
        main_frame = tk.Frame(notebook)
        notebook.add(main_frame, text="Clustering")

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

        # Add Widgets to the scrollable frame
        label = tk.Label(scroll_frame, text="K-Means Clustering", font=('helvetica', 16, 'bold'))
        label.pack(pady=20)

        # Input for Number of Clusters
        entry_label = tk.Label(scroll_frame, text="Number of Clusters:", font=('helvetica', 12))
        entry_label.pack(anchor="w", padx=10)
        
        self.cluster_entry = tk.Entry(scroll_frame, font=('helvetica', 12), width=50)
        self.cluster_entry.pack(fill="x", padx=10, pady=5)

        # Button to run the clustering
        process_button = Button(scroll_frame, text='Run K-Means', command=self.run_kmeans, bg='brown', fg='white', font=('helvetica', 12, 'bold'))
        process_button.pack(pady=20)

        # Results and Interpretation Labels
        self.results_label = tk.Label(scroll_frame, text="", font=('helvetica', 12))
        self.results_label.pack(pady=10)

        self.interpretation_label = tk.Label(scroll_frame, text="", font=('helvetica', 12), wraplength=400, justify='left')
        self.interpretation_label.pack(pady=10)

        # Frame to contain the plot
        self.canvas_frame = tk.Frame(scroll_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        # Spacer for better scrolling experience
        spacer = tk.Frame(scroll_frame, height=50)
        spacer.pack()

        # Handle dynamic resizing
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(self.scrollable_window, width=e.width))

    def run_kmeans(self):
        try:
            # Automatically detect the first two numeric columns for clustering
            numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_cols) < 2:
                messagebox.showerror("Error", "The file must have at least two numeric columns for clustering.")
                return
            
            # Get the number of clusters
            k = int(self.cluster_entry.get())
            if k < 2:
                messagebox.showerror("Error", "Please enter a number greater than 1 for clusters.")
                return

            # Run KMeans
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42).fit(self.df[numeric_cols[:2]])
            centroids = kmeans.cluster_centers_
            labels = kmeans.labels_

            # Calculate silhouette score for clustering quality
            silhouette_avg = silhouette_score(self.df[numeric_cols[:2]], labels)
            
            # Clear previous plot if it exists
            if self.canvas is not None:
                self.canvas.get_tk_widget().destroy()
                self.canvas = None

            # Plot the new clustering result
            fig, ax = plt.subplots(figsize=(6, 4))
            scatter = ax.scatter(self.df[numeric_cols[0]], self.df[numeric_cols[1]], c=labels, cmap='viridis', s=50, alpha=0.6)
            ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, marker='X')
            ax.set_title(f'K-Means Clustering (k={k})')
            ax.set_xlabel(numeric_cols[0])
            ax.set_ylabel(numeric_cols[1])
            fig.colorbar(scatter, ax=ax)

            # Embed the plot in Tkinter
            for widget in self.canvas_frame.winfo_children():
                widget.destroy()  # Clear previous plot

            canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Display results
            self.results_label.config(text=f"Silhouette Score: {silhouette_avg:.2f}")

            # Interpretation Logic based on Silhouette Score
            if silhouette_avg >= 0.7:
                comment = "Good clustering. The clusters are well separated."
            elif silhouette_avg >= 0.5:
                comment = "Fair clustering. There is some overlap between clusters."
            else:
                comment = "Poor clustering. Consider adjusting the number of clusters or features."

            interpretation = f"Silhouette Score: {silhouette_avg:.2f} → {comment}"
            self.interpretation_label.config(text=interpretation)

        except Exception as e:
            messagebox.showerror("Error", str(e))
