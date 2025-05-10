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
        cluster_frame = tk.Frame(notebook)
        notebook.add(cluster_frame, text="Clustering")

        label = tk.Label(cluster_frame, text="K-Means Clustering", font=('helvetica', 14))
        label.pack(pady=10)

        # Input for Number of Clusters
        entry_label = tk.Label(cluster_frame, text="Number of Clusters:")
        entry_label.pack()

        self.cluster_entry = tk.Entry(cluster_frame)
        self.cluster_entry.pack(pady=5)

        # Button to run the clustering
        process_button = Button(cluster_frame, text='Run K-Means', command=self.run_kmeans, bg='brown', fg='white', font=('helvetica', 10, 'bold'))
        process_button.pack(pady=10)

        # Results and Interpretation Labels
        self.results_label = tk.Label(cluster_frame, text="", font=('helvetica', 10))
        self.results_label.pack(pady=10)

        self.interpretation_label = tk.Label(cluster_frame, text="", font=('helvetica', 10), wraplength=400, justify='left')
        self.interpretation_label.pack(pady=10)

        # Frame to contain the plot
        self.canvas_frame = tk.Frame(cluster_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

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
