import tkinter as tk
from tkinter import Button, messagebox, ttk
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ClusteringTab:
    def __init__(self, notebook, df):
        self.df = df.copy()
        self.create_tab(notebook)

    def create_tab(self, notebook):
        main_frame = tk.Frame(notebook)
        notebook.add(main_frame, text="Clustering")

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

        label = tk.Label(scroll_frame, text="K-Means Clustering with Feature Selection", font=('helvetica', 16, 'bold'))
        label.pack(pady=20)

        feature_label = tk.Label(scroll_frame, text="Select Features (comma-separated):", font=('helvetica', 12))
        feature_label.pack(anchor="w", padx=10)

        self.features_entry = tk.Entry(scroll_frame, font=('helvetica', 12), width=50)
        self.features_entry.pack(fill="x", padx=10, pady=5)

        cluster_label = tk.Label(scroll_frame, text="Number of Clusters:", font=('helvetica', 12))
        cluster_label.pack(anchor="w", padx=10)

        self.cluster_entry = tk.Entry(scroll_frame, font=('helvetica', 12), width=50)
        self.cluster_entry.pack(fill="x", padx=10, pady=5)

        process_button = Button(scroll_frame, text='Run K-Means', command=self.run_kmeans, bg='brown', fg='white', font=('helvetica', 12, 'bold'))
        process_button.pack(pady=20)

        self.results_label = tk.Label(scroll_frame, text="", font=('helvetica', 12), justify="left", wraplength=600)
        self.results_label.pack(pady=10)

        self.interpretation_label = tk.Label(scroll_frame, text="", font=('helvetica', 12), wraplength=600, justify='left')
        self.interpretation_label.pack(pady=10)

        self.canvas_frame = tk.Frame(scroll_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        spacer = tk.Frame(scroll_frame, height=50)
        spacer.pack()

        canvas.bind("<Configure>", lambda e: canvas.itemconfig(self.scrollable_window, width=e.width))

    def run_kmeans(self):
        try:
            selected_features = [col.strip() for col in self.features_entry.get().split(',')]
            if not selected_features:
                messagebox.showerror("Error", "Please enter at least two features for clustering.")
                return

            k = int(self.cluster_entry.get())
            if k < 2:
                messagebox.showerror("Error", "Please enter a number greater than 1 for clusters.")
                return

            df_selected = self.df[selected_features].copy()

            for col in df_selected.columns:
                if df_selected[col].dtype == 'object':
                    le = LabelEncoder()
                    df_selected[col] = le.fit_transform(df_selected[col].astype(str))

            if df_selected.shape[1] < 2:
                messagebox.showerror("Error", "Need at least two numeric features for clustering.")
                return

            scaler = StandardScaler()
            X = scaler.fit_transform(df_selected)

            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = kmeans.fit_predict(X)
            centroids = kmeans.cluster_centers_
            silhouette_avg = silhouette_score(X, labels)
            inertia = kmeans.inertia_
            n_iter = kmeans.n_iter_
            cluster_sizes = [list(labels).count(i) for i in range(k)]

            for widget in self.canvas_frame.winfo_children():
                widget.destroy()

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
            ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, marker='X')
            ax.set_title(f'K-Means Clustering (k={k})')
            ax.set_xlabel(selected_features[0])
            ax.set_ylabel(selected_features[1])

            canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Display Results
            result_text = (
                f"Silhouette Score: {silhouette_avg:.2f}\n"
                f"Within Cluster Sum of Squares (WCSS): {inertia:.2f}\n"
                f"Number of Iterations: {n_iter}\n"
                f"Cluster Size Distribution: {cluster_sizes}"
            )
            self.results_label.config(text=result_text)

            # Centroid Details
            centroids_text = "Cluster Centroids (Standardized Values):\n"
            for i, center in enumerate(centroids):
                centroid_values = ', '.join(f"{val:.2f}" for val in center)
                centroids_text += f"Cluster {i + 1}: [{centroid_values}]\n"

            # Interpretation
            if silhouette_avg >= 0.7:
                comment = "Good clustering. The clusters are well separated."
            elif silhouette_avg >= 0.5:
                comment = "Fair clustering. Some overlap between clusters."
            else:
                comment = "Poor clustering. Try adjusting cluster count or features."

            self.interpretation_label.config(text=centroids_text + "\n" + comment)

        except Exception as e:
            messagebox.showerror("Error", str(e))