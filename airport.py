# ============================================================
# AIRPORT SECURITY – FINAL DATA ANALYST DESKTOP APPLICATION
# ACP – CLUSTERING – AFC – CYBERSECURITY (ENGLISH VERSION)
# ============================================================

# ===================== SYSTEM FIXES =====================
import os
import warnings
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

# ===================== IMPORTS =====================
import tkinter as tk
from tkinter import ttk, font
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
import matplotlib

# Set matplotlib to use Agg backend for better performance
matplotlib.use('Agg')

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import chi2_contingency
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# ===================== DATA GENERATION =====================
np.random.seed(42)

# English column names
data = pd.DataFrame({
    "Passengers/hour": np.random.randint(10, 20000, 100),
    "Control time (min)": np.random.randint(1, 60, 100),
    "Detections": np.random.randint(0, 500, 100),
    "Agents": np.random.randint(1, 500, 100),
    "Cameras": np.random.randint(1, 2000, 100),
    "Incidents": np.random.randint(0, 100, 100),
    "Area (m²)": np.random.randint(100, 50000, 100),
    "Satisfaction (%)": np.random.randint(0, 100, 100)
})

# ===================== PREPROCESSING =====================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# ===================== CLUSTERING & AI =====================
kmeans = KMeans(n_clusters=4, random_state=0, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

rf = RandomForestClassifier(random_state=0)
rf.fit(X_scaled, clusters)

# ===================== MODERN UI STYLING =====================
class ModernUI:
    # Color scheme
    PRIMARY = "#2C3E50"
    SECONDARY = "#34495E"
    ACCENT = "#2980B9"  # Changed to requested blue
    SUCCESS = "#27AE60"
    WARNING = "#E74C3C"
    LIGHT = "#ECF0F1"
    DARK = "#2C3E50"
    BACKGROUND = "#F5F7FA"
    
    # Fonts
    TITLE_FONT = ("Segoe UI", 24, "bold")
    HEADING_FONT = ("Segoe UI", 18, "bold")
    SUBHEADING_FONT = ("Segoe UI", 14, "bold")
    BODY_FONT = ("Segoe UI", 11)
    BUTTON_FONT = ("Segoe UI", 11, "bold")
    
    @staticmethod
    def configure_styles():
        style = ttk.Style()
        
        # Configure main styles
        style.theme_use('clam')
        
        # Configure button styles
        style.configure('Primary.TButton',
                       font=ModernUI.BUTTON_FONT,
                       padding=12,
                       background=ModernUI.ACCENT,
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none')
        style.map('Primary.TButton',
                 background=[('active', '#2471A3'), ('pressed', '#1F618D')])
        
        style.configure('Secondary.TButton',
                       font=ModernUI.BUTTON_FONT,
                       padding=10,
                       background=ModernUI.SECONDARY,
                       foreground='white',
                       borderwidth=0)
        style.map('Secondary.TButton',
                 background=[('active', '#2C3E50'), ('pressed', '#1A252F')])

# ===================== UI UTILITIES =====================
def show_table(df, title):
    win = tk.Toplevel(app)
    win.title(title)
    win.geometry("1400x800")
    win.configure(bg=ModernUI.BACKGROUND)
    
    # Header
    header = tk.Frame(win, bg=ModernUI.PRIMARY, height=80)
    header.pack(fill="x")
    header.pack_propagate(False)
    
    tk.Label(header, text=title, font=ModernUI.HEADING_FONT,
             fg='white', bg=ModernUI.PRIMARY).pack(pady=20)
    
    # Main content
    main_frame = tk.Frame(win, bg=ModernUI.BACKGROUND)
    main_frame.pack(fill="both", expand=True, padx=20, pady=20)
    
    # Create treeview with scrollbars
    tree_frame = tk.Frame(main_frame, bg=ModernUI.BACKGROUND)
    tree_frame.pack(fill="both", expand=True)
    
    # Create treeview
    tree = ttk.Treeview(tree_frame, columns=list(df.columns), show="headings", height=25)
    
    # Add scrollbars
    tree_vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=tree_vsb.set)
    
    tree_hsb = ttk.Scrollbar(win, orient="horizontal", command=tree.xview)
    tree.configure(xscrollcommand=tree_hsb.set)
    
    # Grid layout
    tree.grid(row=0, column=0, sticky="nsew")
    tree_vsb.grid(row=0, column=1, sticky="ns")
    tree_hsb.pack(side="bottom", fill="x")
    
    # Configure grid weights
    tree_frame.grid_rowconfigure(0, weight=1)
    tree_frame.grid_columnconfigure(0, weight=1)
    
    # Configure columns
    for col in df.columns:
        tree.heading(col, text=col, anchor="center")
        col_width = min(max(len(str(col)) * 10, 120), 200)
        tree.column(col, width=col_width, anchor="center", minwidth=100)
    
    # Insert data
    for index, row in df.iterrows():
        tree.insert("", "end", values=list(row))
    
    # Footer with stats
    footer = tk.Frame(main_frame, bg=ModernUI.BACKGROUND)
    footer.pack(fill="x", pady=(10, 0))
    
    stats_text = f"{len(df)} rows × {len(df.columns)} columns"
    tk.Label(footer, text=stats_text, font=ModernUI.BODY_FONT,
             fg=ModernUI.DARK, bg=ModernUI.BACKGROUND).pack()

def show_message(title, text):
    win = tk.Toplevel(app)
    win.title(title)
    win.geometry("800x600")
    win.configure(bg=ModernUI.BACKGROUND)
    
    # Header
    header = tk.Frame(win, bg=ModernUI.PRIMARY, height=70)
    header.pack(fill="x")
    header.pack_propagate(False)
    
    tk.Label(header, text=title, font=ModernUI.HEADING_FONT,
             fg='white', bg=ModernUI.PRIMARY).pack(pady=15)
    
    # Content with scrollbar
    content_frame = tk.Frame(win, bg=ModernUI.BACKGROUND)
    content_frame.pack(fill="both", expand=True, padx=30, pady=20)
    
    # Text widget with scrollbar
    text_frame = tk.Frame(content_frame, bg=ModernUI.BACKGROUND)
    text_frame.pack(fill="both", expand=True)
    
    text_widget = tk.Text(text_frame,
                         wrap="word",
                         fg=ModernUI.DARK,
                         bg='white',
                         font=ModernUI.BODY_FONT,
                         padx=15,
                         pady=15,
                         relief='flat',
                         borderwidth=1)
    
    scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
    text_widget.configure(yscrollcommand=scrollbar.set)
    
    text_widget.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    text_widget.insert("1.0", text)
    text_widget.config(state="disabled")

def show_plot(fig):
    win = tk.Toplevel(app)
    win.title("Graphical Visualization")
    win.geometry("1200x800")
    win.configure(bg=ModernUI.BACKGROUND)
    
    # Header
    header = tk.Frame(win, bg=ModernUI.PRIMARY, height=70)
    header.pack(fill="x")
    header.pack_propagate(False)
    
    tk.Label(header, text="Analysis Graph", font=ModernUI.HEADING_FONT,
             fg='white', bg=ModernUI.PRIMARY).pack(pady=15)
    
    # Content frame
    content_frame = tk.Frame(win, bg=ModernUI.BACKGROUND)
    content_frame.pack(fill="both", expand=True, padx=20, pady=20)
    
    # Create canvas for matplotlib figure
    canvas = FigureCanvasTkAgg(fig, content_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
    
    # Add matplotlib toolbar
    toolbar = NavigationToolbar2Tk(canvas, content_frame)
    toolbar.update()
    canvas.get_tk_widget().pack(fill="both", expand=True)

# ===================== MAIN WINDOW =====================
app = tk.Tk()
app.title("Airport Security Analysis Platform")
app.geometry("1600x900")
app.configure(bg=ModernUI.BACKGROUND)

# Configure styles
ModernUI.configure_styles()

# ===================== MODERN LAYOUT =====================
# Create main container
main_container = tk.Frame(app, bg=ModernUI.BACKGROUND)
main_container.pack(fill="both", expand=True)

# Sidebar with modern design
sidebar = tk.Frame(main_container, bg=ModernUI.PRIMARY, width=280)
sidebar.pack(side="left", fill="y")

# Logo/Title area
logo_frame = tk.Frame(sidebar, bg=ModernUI.PRIMARY)
logo_frame.pack(fill="x", pady=(40, 30))

tk.Label(logo_frame, text="Airport", font=ModernUI.TITLE_FONT, 
         bg=ModernUI.PRIMARY, fg='white').pack()
tk.Label(logo_frame, text="SECURITY", font=ModernUI.TITLE_FONT, 
         bg=ModernUI.PRIMARY, fg=ModernUI.LIGHT).pack()

# Navigation menu
nav_frame = tk.Frame(sidebar, bg=ModernUI.PRIMARY)
nav_frame.pack(fill="x", padx=20, pady=20)

# Navigation buttons
nav_buttons = []

def create_nav_button(text, command):
    btn_frame = tk.Frame(nav_frame, bg=ModernUI.PRIMARY)
    btn_frame.pack(fill="x", pady=8)
    
    btn = tk.Button(btn_frame,
                    text=text,
                    font=ModernUI.BUTTON_FONT,
                    bg=ModernUI.SECONDARY,
                    fg='white',
                    activebackground=ModernUI.ACCENT,
                    activeforeground='white',
                    relief='flat',
                    cursor='hand2',
                    anchor='w',
                    padx=20,
                    pady=15,
                    command=command)
    btn.pack(fill="x")
    
    # Add hover effect
    def on_enter(e):
        btn.configure(bg=ModernUI.ACCENT)
    
    def on_leave(e):
        btn.configure(bg=ModernUI.SECONDARY)
    
    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)
    
    return btn

# Create navigation buttons
nav_acp = create_nav_button("Principal Component Analysis", lambda: show_section("ACP"))
nav_cluster = create_nav_button("Clustering", lambda: show_section("Clustering"))
nav_afc = create_nav_button("Correspondence Analysis", lambda: show_section("AFC"))
nav_cyber = create_nav_button("Cybersecurity", lambda: show_section("Cybersecurity"))

# Main content area
content = tk.Frame(main_container, bg=ModernUI.BACKGROUND)
content.pack(side="right", fill="both", expand=True, padx=40, pady=30)

# Welcome screen
def show_welcome():
    clear_content()
    
    # Welcome header
    welcome_frame = tk.Frame(content, bg=ModernUI.BACKGROUND)
    welcome_frame.pack(fill="x", pady=(0, 40))
    
    tk.Label(welcome_frame, text="DATA Analysis, AI and Cybersecurity Project", 
             font=ModernUI.TITLE_FONT, fg=ModernUI.PRIMARY, 
             bg=ModernUI.BACKGROUND).pack()
    tk.Label(welcome_frame, text="Airport Security", 
             font=("Segoe UI", 18), fg=ModernUI.SECONDARY, 
             bg=ModernUI.BACKGROUND).pack(pady=10)
    
    # Stats cards
    stats_frame = tk.Frame(content, bg=ModernUI.BACKGROUND)
    stats_frame.pack(fill="x", pady=(0, 40))
    
    stats = [
        ("100", "Analyzed zones", ModernUI.ACCENT),
        ("8", "Security variables", ModernUI.SUCCESS),
        ("5", "Identified clusters", ModernUI.WARNING)
    ]
    
    for i, (value, label, color) in enumerate(stats):
        card = tk.Frame(stats_frame, bg='white', relief='flat', 
                       borderwidth=1, highlightbackground='#E0E0E0',
                       highlightthickness=1)
        card.grid(row=0, column=i, padx=10, sticky='nsew')
        stats_frame.grid_columnconfigure(i, weight=1)
        
        tk.Label(card, text=value, font=("Segoe UI", 32, "bold"), 
                 fg=color, bg='white').pack(pady=(20, 5))
        tk.Label(card, text=label, font=ModernUI.BODY_FONT, 
                 fg=ModernUI.DARK, bg='white').pack(pady=(0, 20))
    
    # Features section
    features_frame = tk.Frame(content, bg=ModernUI.BACKGROUND)
    features_frame.pack(fill="both", expand=True)
    
    tk.Label(features_frame, text="Available Features", 
             font=ModernUI.HEADING_FONT, fg=ModernUI.PRIMARY, 
             bg=ModernUI.BACKGROUND).pack(anchor='w', pady=(0, 20))
    
    features = [
        ("Principal Component Analysis", "Principal Component Analysis with complete visualization"),
        ("Clustering", "Grouping by K-Means and prediction by Random Forest"),
        ("Correspondence Analysis", "Correspondence Factor Analysis with Chi-square test"),
        ("Cybersecurity", "Anomaly detection with Isolation Forest and LOF")
    ]
    
    for feature_text, description in features:
        feature_item = tk.Frame(features_frame, bg=ModernUI.BACKGROUND)
        feature_item.pack(fill="x", pady=8)
        
        tk.Label(feature_item, text=feature_text, font=("Segoe UI", 14, "bold"),
                 fg=ModernUI.ACCENT, bg=ModernUI.BACKGROUND, 
                 anchor='w').pack(side='left', padx=(0, 20))
        tk.Label(feature_item, text=description, font=ModernUI.BODY_FONT,
                 fg=ModernUI.DARK, bg=ModernUI.BACKGROUND,
                 wraplength=800, justify='left').pack(side='left')

def clear_content():
    for w in content.winfo_children():
        w.destroy()

def show_section(section_name):
    clear_content()
    
    # Section header
    header_frame = tk.Frame(content, bg=ModernUI.BACKGROUND)
    header_frame.pack(fill="x", pady=(0, 30))
    
    section_titles = {
        "ACP": "Principal Component Analysis",
        "Clustering": "Clustering & Artificial Intelligence",
        "AFC": "Correspondence Factor Analysis",
        "Cybersecurity": "Cybersecurity - Anomaly Detection"
    }
    
    tk.Label(header_frame, text=section_titles.get(section_name, section_name),
             font=ModernUI.HEADING_FONT, fg=ModernUI.PRIMARY,
             bg=ModernUI.BACKGROUND).pack(anchor='w')
    
    # Show appropriate section
    if section_name == "ACP":
        show_acp_content()
    elif section_name == "Clustering":
        show_clustering_content()
    elif section_name == "AFC":
        show_afc_content()
    elif section_name == "Cybersecurity":
        show_cybersecurity_content()

# ===================== ACP CONTENT =====================
def show_acp_content():
    # Create container for buttons with responsive grid
    buttons_container = tk.Frame(content, bg=ModernUI.BACKGROUND)
    buttons_container.pack(fill="x", pady=(20, 0))
    
    # Create 2x4 grid of buttons (8 buttons total)
    button_grid = tk.Frame(buttons_container, bg=ModernUI.BACKGROUND)
    button_grid.pack()
    
    def create_action_button(text, command, row, col):
        btn = tk.Button(button_grid,
                       text=text,
                       font=("Segoe UI", 12, "bold"),
                       bg=ModernUI.ACCENT,
                       fg='white',
                       activebackground='#2471A3',
                       activeforeground='white',
                       relief='flat',
                       cursor='hand2',
                       width=20,
                       height=2,
                       command=command)
        btn.grid(row=row, column=col, padx=10, pady=10, sticky='nsew')
        return btn
    
    def stats():
        desc = pd.DataFrame({
            "Mean": data.mean(),
            "Standard Deviation": data.std()
        }).round(2)
        show_table(desc, "Descriptive Statistics")

    def mcr():
        df_mcr = pd.DataFrame(X_scaled, columns=data.columns).round(3)
        df_mcr.index = [f"Zone {i+1}" for i in range(len(df_mcr))]
        show_table(df_mcr, "Centered-Reduced Matrix")

    def corr():
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f", center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title("Correlation Matrix", fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        show_plot(fig)

    def inertia():
        eig = pca.explained_variance_ratio_ * 100
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Bar plot for eigenvalues
        bars = ax.bar(range(1, len(eig)+1), eig, alpha=0.7, color=ModernUI.ACCENT,
                     label='Inertia per axis')
        
        # Cumulative line
        ax2 = ax.twinx()
        ax2.plot(range(1, len(eig)+1), np.cumsum(eig), 'ro-', linewidth=3,
                markersize=8, label='Cumulative inertia')
        
        ax.set_xlabel("Factor Axes", fontsize=12, fontweight='bold')
        ax.set_ylabel("Explained Variance (%)", fontsize=12, fontweight='bold')
        ax2.set_ylabel("Cumulative Variance (%)", fontsize=12, fontweight='bold')
        ax.set_title("Eigenvalue Scree Plot", fontsize=16, fontweight='bold', pad=20)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, eig)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
        
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, len(eig)+1))
        plt.tight_layout()
        show_plot(fig)

    def plan_ind():
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Create scatter plot
        scatter = ax.scatter(X_pca[:,0], X_pca[:,1], alpha=0.7, s=60,
                           color=ModernUI.ACCENT, edgecolors='white', linewidth=0.5)
        
        ax.set_xlabel(f"Axis 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", 
                     fontsize=12, fontweight='bold')
        ax.set_ylabel(f"Axis 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", 
                     fontsize=12, fontweight='bold')
        ax.set_title("Factor Plane of Individuals - 100 Airport Zones", 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Add labels for ALL zones as "Zone X"
        for i in range(len(X_pca)):
            ax.text(X_pca[i,0] + 0.02, X_pca[i,1] + 0.02, 
                   f"Zone {i+1}", fontsize=8, alpha=0.7, 
                   color=ModernUI.DARK)
        
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.legend(['Zones'], loc='upper right')
        
        plt.tight_layout()
        show_plot(fig)

    def cercle():
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Draw unit circle
        circle = plt.Circle((0, 0), 1, color='gray', fill=False, 
                        linestyle='--', linewidth=1.5, alpha=0.5)
        ax.add_artist(circle)
        
        # Define a color palette for variables
        colors = cm.tab10(np.linspace(0, 1, len(data.columns)))
        
        # Draw arrows with different colors and add labels
        for i, var in enumerate(data.columns):
            # Draw arrow
            arrow = ax.arrow(0, 0, 
                            pca.components_[0, i] * 0.85,
                            pca.components_[1, i] * 0.85, 
                            color=colors[i], 
                            alpha=0.9, 
                            head_width=0.05, 
                            head_length=0.05,
                            length_includes_head=True,
                            linewidth=2.5,
                            label=var)
            
            # Add variable name at arrow end with background
            arrow_end_x = pca.components_[0, i] * 0.95
            arrow_end_y = pca.components_[1, i] * 0.95
            
            # Calculate text offset based on quadrant
            text_offset = 0.03
            ha = 'center'
            va = 'center'
            
            if arrow_end_x > 0:
                text_x = arrow_end_x + text_offset
                ha = 'left'
            else:
                text_x = arrow_end_x - text_offset
                ha = 'right'
                
            if arrow_end_y > 0:
                text_y = arrow_end_y + text_offset
                va = 'bottom'
            else:
                text_y = arrow_end_y - text_offset
                va = 'top'
            
            # Add variable label with colored background
            ax.text(text_x, text_y, var,
                    color=colors[i],
                    fontsize=11,
                    fontweight='bold',
                    ha=ha,
                    va=va,
                    bbox=dict(boxstyle='round,pad=0.3', 
                            facecolor='white', 
                            alpha=0.8,
                            edgecolor=colors[i],
                            linewidth=1.5))
        
        # Add grid lines at major positions
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Add diagonal grid lines
        for angle in [45, 135, 225, 315]:
            angle_rad = np.deg2rad(angle)
            x_end = np.cos(angle_rad)
            y_end = np.sin(angle_rad)
            ax.plot([0, x_end], [0, y_end], 'gray', linestyle='--', linewidth=0.5, alpha=0.2)
        
        # Set axis limits
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        
        # Add axis lines through origin
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        # Add axis labels with variance percentage
        ax.set_xlabel(f"Axis 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", 
                    fontsize=12, fontweight='bold')
        ax.set_ylabel(f"Axis 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", 
                    fontsize=12, fontweight='bold')
        
        # Add title
        ax.set_title("Correlation Circle - Principal Component Analysis", 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Add subtle grid
        ax.grid(True, alpha=0.2, linestyle='--')
        
        # Add legend with variable names and their colors
        legend_elements = []
        for i, var in enumerate(data.columns):
            legend_elements.append(plt.Line2D([0], [0], 
                                            color=colors[i], 
                                            lw=3, 
                                            label=var))
        
        # Position legend outside the plot
        ax.legend(handles=legend_elements, 
                loc='upper left', 
                bbox_to_anchor=(1.02, 1),
                fontsize=10,
                title="Variables",
                title_fontsize=11)
        
        # Ensure equal aspect ratio
        ax.set_aspect('equal', 'box')
        
        plt.tight_layout()
        show_plot(fig)

    def qualite():
        cos2 = np.zeros((len(X_pca), 2))
        for i in range(len(X_pca)):
            norm = np.sum(X_pca[i,:]**2)
            if norm > 0:
                cos2[i,0] = X_pca[i,0]**2 / norm
                cos2[i,1] = X_pca[i,1]**2 / norm
        
        df_qual = pd.DataFrame({
            "Zone": [f"Zone {i+1}" for i in range(len(X_pca))],
            "Cos² Axis 1": cos2[:,0],
            "Cos² Axis 2": cos2[:,1],
            "Total Quality": cos2[:,0] + cos2[:,1]
        }).round(3)
        show_table(df_qual, "Quality of Representation of Individuals")

    def contribution():
        contrib1 = (X_pca[:,0]**2) / np.sum(X_pca[:,0]**2) * 100
        contrib2 = (X_pca[:,1]**2) / np.sum(X_pca[:,1]**2) * 100
        
        df_contrib = pd.DataFrame({
            "Zone": [f"Zone {i+1}" for i in range(len(X_pca))],
            "Contribution Axis 1 (%)": contrib1,
            "Contribution Axis 2 (%)": contrib2,
            "Total Contribution (%)": contrib1 + contrib2
        }).round(2)
        show_table(df_contrib, "Contribution of Individuals to Axes")

    # Create action buttons in 2x4 grid
    create_action_button("Statistics", stats, 0, 0)
    create_action_button("Centered Matrix", mcr, 0, 1)
    create_action_button("Correlation", corr, 0, 2)
    create_action_button("Eigenvalues", inertia, 0, 3)
    create_action_button("Factor Plane", plan_ind, 1, 0)
    create_action_button("Correlation Circle", cercle, 1, 1)
    create_action_button("Quality", qualite, 1, 2)
    create_action_button("Contributions", contribution, 1, 3)
    
    # Configure grid weights for responsiveness
    for i in range(4):
        button_grid.columnconfigure(i, weight=1)
    for i in range(2):
        button_grid.rowconfigure(i, weight=1)

# ===================== CLUSTERING CONTENT =====================
def show_clustering_content():
    clear_content()
    tk.Label(content, text="AI – Clustering & Artificial Intelligence",
             font=("Segoe UI", 22, "bold"), fg=ModernUI.PRIMARY, 
             bg=ModernUI.BACKGROUND).pack(pady=(0, 30))
    
    # Generate clustering results for k=3 to k=7
    clustering_results = {}
    silhouette_scores = {}
    inertia_values = {}
    
    for k in range(3, 8):  # k = 3, 4, 5, 6, 7
        kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters_k = kmeans_model.fit_predict(X_scaled)
        clustering_results[k] = clusters_k
        inertia_values[k] = kmeans_model.inertia_
        
        # Calculate silhouette score
        from sklearn.metrics import silhouette_score
        if len(set(clusters_k)) > 1:
            silhouette_scores[k] = silhouette_score(X_scaled, clusters_k)
        else:
            silhouette_scores[k] = 0
    
    # Train Random Forest on each clustering
    rf_models = {}
    rf_accuracies = {}
    for k in range(3, 8):
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_scaled, clustering_results[k])
        rf_models[k] = rf
        rf_accuracies[k] = rf.score(X_scaled, clustering_results[k])
    
    # ===================== BUTTON 1: Show 5 Clusters =====================
    def show_five_clusters():
        cluster_window = tk.Toplevel(app)
        cluster_window.title("Visualization of 5 Clustering (k=3 to 7)")
        cluster_window.geometry("1400x900")
        cluster_window.configure(bg=ModernUI.BACKGROUND)
        
        # Store current k value as an attribute
        cluster_window.current_k = tk.IntVar(value=3)
        
        # Header
        header = tk.Frame(cluster_window, bg=ModernUI.PRIMARY, height=80)
        header.pack(fill="x")
        header.pack_propagate(False)
        
        title_label = tk.Label(header, 
                              text=f"Clustering k=3 - Cluster View",
                              font=ModernUI.HEADING_FONT,
                              fg='white', 
                              bg=ModernUI.PRIMARY)
        title_label.pack(pady=20)
        
        # Navigation buttons frame
        nav_frame = tk.Frame(cluster_window, bg=ModernUI.BACKGROUND)
        nav_frame.pack(fill="x", padx=30, pady=15)
        
        # Previous button
        def prev_cluster():
            current = cluster_window.current_k.get()
            if current > 3:
                cluster_window.current_k.set(current - 1)
                update_cluster_display()
        
        prev_btn = tk.Button(nav_frame, text="Previous (k-1)",
                            font=ModernUI.BUTTON_FONT,
                            bg=ModernUI.SECONDARY,
                            fg='white',
                            command=prev_cluster,
                            padx=20, pady=10)
        prev_btn.pack(side="left", padx=10)
        
        # K selector
        k_frame = tk.Frame(nav_frame, bg=ModernUI.BACKGROUND)
        k_frame.pack(side="left", expand=True)
        
        tk.Label(k_frame, text="Number of Clusters (k):",
                font=ModernUI.BODY_FONT,
                fg=ModernUI.DARK,
                bg=ModernUI.BACKGROUND).pack(side="left", padx=10)
        
        k_value_label = tk.Label(k_frame, 
                                text="3",
                                font=("Segoe UI", 16, "bold"),
                                fg=ModernUI.ACCENT,
                                bg=ModernUI.BACKGROUND)
        k_value_label.pack(side="left")
        
        # Next button
        def next_cluster():
            current = cluster_window.current_k.get()
            if current < 7:
                cluster_window.current_k.set(current + 1)
                update_cluster_display()
        
        next_btn = tk.Button(nav_frame, text="Next (k+1)",
                            font=ModernUI.BUTTON_FONT,
                            bg=ModernUI.SECONDARY,
                            fg='white',
                            command=next_cluster,
                            padx=20, pady=10)
        next_btn.pack(side="right", padx=10)
        
        # Main content frame
        content_frame = tk.Frame(cluster_window, bg=ModernUI.BACKGROUND)
        content_frame.pack(fill="both", expand=True, padx=30, pady=20)
        
        # Create a PanedWindow for resizable split
        paned = tk.PanedWindow(content_frame, orient=tk.HORIZONTAL, bg=ModernUI.BACKGROUND)
        paned.pack(fill="both", expand=True)
        
        # Left frame for plot
        left_frame = tk.Frame(paned, bg=ModernUI.BACKGROUND)
        
        # Right frame for cluster info
        right_frame = tk.Frame(paned, bg=ModernUI.BACKGROUND)
        
        # Add frames to paned window
        paned.add(left_frame, width=1000)
        paned.add(right_frame, width=400)
        
        # Plot frame
        plot_frame = tk.Frame(left_frame, bg='white', relief='solid', borderwidth=1)
        plot_frame.pack(fill="both", expand=True)
        
        # Info frame
        info_frame = tk.Frame(right_frame, bg='white', relief='solid', borderwidth=1)
        info_frame.pack(fill="both", expand=True)
        
        # Create a container frame for info content
        info_container = tk.Frame(info_frame, bg='white')
        info_container.pack(fill="both", expand=True)
        
        # Add scrollbar to info container
        info_canvas = tk.Canvas(info_container, bg='white', highlightthickness=0)
        info_scrollbar = ttk.Scrollbar(info_container, orient="vertical", command=info_canvas.yview)
        info_content = tk.Frame(info_canvas, bg='white')
        
        info_content.bind(
            "<Configure>",
            lambda e: info_canvas.configure(scrollregion=info_canvas.bbox("all"))
        )
        
        info_canvas.create_window((0, 0), window=info_content, anchor="nw")
        info_canvas.configure(yscrollcommand=info_scrollbar.set)
        
        info_canvas.pack(side="left", fill="both", expand=True)
        info_scrollbar.pack(side="right", fill="y")
        
        def update_cluster_display():
            current_k = cluster_window.current_k.get()
            
            # Update title
            title_label.config(text=f"Clustering k={current_k} - Cluster View")
            k_value_label.config(text=str(current_k))
            
            # Clear previous plot
            for widget in plot_frame.winfo_children():
                widget.destroy()
            
            # Clear previous info
            for widget in info_content.winfo_children():
                widget.destroy()
            
            # Create new plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Get current clustering
            clusters = clustering_results[current_k]
            
            # Define colors for clusters
            colors = cm.tab10(np.linspace(0, 1, current_k))
            
            # Plot each cluster
            for cluster_num in range(current_k):
                mask = clusters == cluster_num
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                          color=colors[cluster_num],
                          s=80, alpha=0.8, edgecolors='white',
                          linewidth=0.5, label=f'Cluster {cluster_num}')
                
                # Label ALL zones in this cluster
                cluster_indices = np.where(mask)[0]
                for idx in cluster_indices:
                    ax.text(X_pca[idx, 0] + 0.02, X_pca[idx, 1] + 0.02,
                           f"Zone {idx+1}", fontsize=7, alpha=0.7,
                           color='black')
            
            ax.set_xlabel(f"Axis 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                         fontsize=11, fontweight='bold')
            ax.set_ylabel(f"Axis 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
                         fontsize=11, fontweight='bold')
            ax.set_title(f'K-Means with k={current_k} Clusters',
                        fontsize=14, fontweight='bold', pad=15)
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
            ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
            
            plt.tight_layout()
            
            # Embed plot
            canvas = FigureCanvasTkAgg(fig, plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
            
            # Add toolbar
            toolbar = NavigationToolbar2Tk(canvas, plot_frame)
            toolbar.update()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
            # Update info panel
            # Metrics
            metrics_frame = tk.Frame(info_content, bg='white')
            metrics_frame.pack(fill="x", pady=(0, 20))
            
            tk.Label(metrics_frame, text=f"Metrics for k={current_k}",
                    font=("Segoe UI", 14, "bold"),
                    fg=ModernUI.PRIMARY,
                    bg='white').pack(anchor='w', pady=(0, 10))
            
            metrics = [
                ("Silhouette Score:", f"{silhouette_scores[current_k]:.4f}"),
                ("Intra-cluster Inertia:", f"{inertia_values[current_k]:.0f}"),
                ("RF Accuracy:", f"{rf_accuracies[current_k]*100:.2f}%")
            ]
            
            for metric_name, metric_value in metrics:
                metric_row = tk.Frame(metrics_frame, bg='white')
                metric_row.pack(fill="x", pady=3)
                
                tk.Label(metric_row, text=metric_name,
                        font=ModernUI.BODY_FONT,
                        fg=ModernUI.DARK,
                        bg='white',
                        width=20,
                        anchor='w').pack(side='left')
                
                tk.Label(metric_row, text=metric_value,
                        font=("Segoe UI", 11, "bold"),
                        fg=ModernUI.ACCENT,
                        bg='white').pack(side='left')
            
            # Cluster details
            details_frame = tk.Frame(info_content, bg='white')
            details_frame.pack(fill="both", expand=True)
            
            tk.Label(details_frame, text="Cluster Details:",
                    font=("Segoe UI", 14, "bold"),
                    fg=ModernUI.PRIMARY,
                    bg='white').pack(anchor='w', pady=(0, 10))
            
            # Calculate cluster statistics
            for cluster_num in range(current_k):
                mask = clusters == cluster_num
                cluster_data = data[mask]
                cluster_size = np.sum(mask)
                
                cluster_card = tk.Frame(details_frame, bg='#F8F9FA',
                                       relief='solid', borderwidth=1)
                cluster_card.pack(fill="x", pady=5, padx=5)
                
                # Cluster header
                header_frame = tk.Frame(cluster_card, bg='#E9ECEF')
                header_frame.pack(fill="x", pady=8)
                
                tk.Label(header_frame, text=f"Cluster {cluster_num}",
                        font=("Segoe UI", 12, "bold"),
                        fg=ModernUI.PRIMARY,
                        bg='#E9ECEF').pack(side='left', padx=10)
                
                tk.Label(header_frame, text=f"{cluster_size} zones ({cluster_size/100*100:.1f}%)",
                        font=ModernUI.BODY_FONT,
                        fg=ModernUI.DARK,
                        bg='#E9ECEF').pack(side='right', padx=10)
                
                # Zones list
                zones_frame = tk.Frame(cluster_card, bg='#F8F9FA')
                zones_frame.pack(fill="x", padx=10, pady=5)
                
                cluster_indices = np.where(mask)[0]
                zones_text = ", ".join([f"Zone {idx+1}" for idx in cluster_indices[:10]])
                if len(cluster_indices) > 10:
                    zones_text += f"... (+{len(cluster_indices)-10} more)"
                
                tk.Label(zones_frame, text="Zones:",
                        font=("Segoe UI", 10, "bold"),
                        fg=ModernUI.DARK,
                        bg='#F8F9FA').pack(anchor='w')
                
                tk.Label(zones_frame, text=zones_text,
                        font=ModernUI.BODY_FONT,
                        fg=ModernUI.DARK,
                        bg='#F8F9FA',
                        wraplength=350,
                        justify='left').pack(anchor='w', pady=(0, 5))
                
                # Statistics
                stats_text = f"• Avg Passengers/hour: {cluster_data['Passengers/hour'].mean():.0f}\n"
                stats_text += f"• Avg Detections: {cluster_data['Detections'].mean():.0f}\n"
                stats_text += f"• Avg Satisfaction: {cluster_data['Satisfaction (%)'].mean():.0f}%"
                
                tk.Label(zones_frame, text=stats_text,
                        font=ModernUI.BODY_FONT,
                        fg=ModernUI.DARK,
                        bg='#F8F9FA',
                        justify='left').pack(anchor='w')
        
        # Initial display
        update_cluster_display()
    
    # ===================== BUTTON 2: Percentage of each cluster =====================
    def show_cluster_percentages():
        percent_window = tk.Toplevel(app)
        percent_window.title("Percentage of each Cluster on Data")
        percent_window.geometry("1200x800")
        percent_window.configure(bg=ModernUI.BACKGROUND)
        
        # Store current k value
        percent_window.current_k = tk.IntVar(value=3)
        
        # Header
        header = tk.Frame(percent_window, bg=ModernUI.PRIMARY, height=80)
        header.pack(fill="x")
        header.pack_propagate(False)
        
        title_label = tk.Label(header, 
                              text=f"Cluster Percentages - k=3",
                              font=ModernUI.HEADING_FONT,
                              fg='white', 
                              bg=ModernUI.PRIMARY)
        title_label.pack(pady=20)
        
        # Navigation frame
        nav_frame = tk.Frame(percent_window, bg=ModernUI.BACKGROUND)
        nav_frame.pack(fill="x", padx=30, pady=15)
        
        def prev_percent():
            current = percent_window.current_k.get()
            if current > 3:
                percent_window.current_k.set(current - 1)
                update_percent_display()
        
        def next_percent():
            current = percent_window.current_k.get()
            if current < 7:
                percent_window.current_k.set(current + 1)
                update_percent_display()
        
        prev_btn = tk.Button(nav_frame, text="Previous k",
                            font=ModernUI.BUTTON_FONT,
                            bg=ModernUI.SECONDARY,
                            fg='white',
                            command=prev_percent,
                            padx=20, pady=10)
        prev_btn.pack(side="left", padx=10)
        
        k_frame = tk.Frame(nav_frame, bg=ModernUI.BACKGROUND)
        k_frame.pack(side="left", expand=True)
        
        tk.Label(k_frame, text="Number of Clusters:",
                font=ModernUI.BODY_FONT,
                fg=ModernUI.DARK,
                bg=ModernUI.BACKGROUND).pack(side="left", padx=10)
        
        k_value_label = tk.Label(k_frame,
                                text="k=3",
                                font=("Segoe UI", 16, "bold"),
                                fg=ModernUI.ACCENT,
                                bg=ModernUI.BACKGROUND)
        k_value_label.pack(side="left")
        
        next_btn = tk.Button(nav_frame, text="Next k",
                            font=ModernUI.BUTTON_FONT,
                            bg=ModernUI.SECONDARY,
                            fg='white',
                            command=next_percent,
                            padx=20, pady=10)
        next_btn.pack(side="right", padx=10)
        
        # Content frame
        content_frame = tk.Frame(percent_window, bg=ModernUI.BACKGROUND)
        content_frame.pack(fill="both", expand=True, padx=30, pady=20)
        
        def update_percent_display():
            # Clear previous content
            for widget in content_frame.winfo_children():
                widget.destroy()
            
            current_k = percent_window.current_k.get()
            title_label.config(text=f"Cluster Percentages - k={current_k}")
            k_value_label.config(text=f"k={current_k}")
            
            clusters = clustering_results[current_k]
            counts = np.bincount(clusters)
            percentages = (counts / len(clusters) * 100).round(2)
            
            # Create main frame with PanedWindow for resizable split
            paned = tk.PanedWindow(content_frame, orient=tk.HORIZONTAL, bg=ModernUI.BACKGROUND)
            paned.pack(fill="both", expand=True)
            
            # Left - Pie chart
            left_frame = tk.Frame(paned, bg=ModernUI.BACKGROUND)
            
            # Right - Detailed table
            right_frame = tk.Frame(paned, bg=ModernUI.BACKGROUND)
            
            # Add frames to paned window
            paned.add(left_frame, width=700)
            paned.add(right_frame, width=400)
            
            # Chart frame
            chart_frame = tk.Frame(left_frame, bg='white', relief='solid', borderwidth=1)
            chart_frame.pack(fill="both", expand=True)
            
            # Create pie chart
            fig, ax = plt.subplots(figsize=(8, 8))
            colors = cm.tab10(np.linspace(0, 1, current_k))
            
            wedges, texts, autotexts = ax.pie(percentages,
                                             labels=[f'Cluster {i}' for i in range(current_k)],
                                             autopct='%1.1f%%',
                                             colors=colors,
                                             startangle=90,
                                             pctdistance=0.85)
            
            # Make autotexts larger
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(11)
                autotext.set_fontweight('bold')
            
            # Draw circle for donut effect
            centre_circle = plt.Circle((0, 0), 0.70, fc='white')
            ax.add_artist(centre_circle)
            
            ax.set_title(f'Cluster Distribution (k={current_k})', fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            
            # Embed chart
            canvas = FigureCanvasTkAgg(fig, chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
            
            # Right frame content
            table_frame = tk.Frame(right_frame, bg='white', relief='solid', borderwidth=1)
            table_frame.pack(fill="both", expand=True)
            
            # Create table with scrollbar
            table_container = tk.Frame(table_frame, bg='white')
            table_container.pack(fill="both", expand=True, padx=20, pady=20)
            
            tk.Label(table_container, text="Cluster Details",
                    font=("Segoe UI", 14, "bold"),
                    fg=ModernUI.PRIMARY,
                    bg='white').pack(anchor='w', pady=(0, 15))
            
            # Create treeview for table
            tree_container = tk.Frame(table_container, bg='white')
            tree_container.pack(fill="both", expand=True)
            
            tree = ttk.Treeview(tree_container, columns=('Cluster', 'Count', 'Percentage', 'Zones'), 
                               show='headings', height=current_k)
            
            # Configure columns
            tree.heading('Cluster', text='Cluster')
            tree.heading('Count', text='Count')
            tree.heading('Percentage', text='%')
            tree.heading('Zones', text='Example Zones')
            
            tree.column('Cluster', width=80, anchor='center')
            tree.column('Count', width=80, anchor='center')
            tree.column('Percentage', width=80, anchor='center')
            tree.column('Zones', width=150, anchor='w')
            
            # Add scrollbar
            scrollbar = ttk.Scrollbar(tree_container, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=scrollbar.set)
            
            tree.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            # Insert data
            for cluster_num in range(current_k):
                cluster_indices = np.where(clusters == cluster_num)[0]
                sample_zones = [f"Zone {idx+1}" for idx in cluster_indices[:3]]
                zones_text = ", ".join(sample_zones)
                if len(cluster_indices) > 3:
                    zones_text += f"..."
                
                tree.insert('', 'end', values=(
                    f'C{cluster_num}',
                    counts[cluster_num],
                    f'{percentages[cluster_num]}%',
                    zones_text
                ))
            
            # Statistics frame
            stats_frame = tk.Frame(table_container, bg='white')
            stats_frame.pack(fill="x", pady=(20, 0))
            
            tk.Label(stats_frame, text="Global Statistics:",
                    font=("Segoe UI", 12, "bold"),
                    fg=ModernUI.PRIMARY,
                    bg='white').pack(anchor='w', pady=(0, 10))
            
            stats_text = f"• Total zones: {len(clusters)}\n"
            stats_text += f"• Clusters: {current_k}\n"
            stats_text += f"• Silhouette score: {silhouette_scores[current_k]:.3f}\n"
            stats_text += f"• Largest cluster: {max(percentages):.1f}%\n"
            stats_text += f"• Smallest cluster: {min(percentages):.1f}%"
            
            tk.Label(stats_frame, text=stats_text,
                    font=ModernUI.BODY_FONT,
                    fg=ModernUI.DARK,
                    bg='white',
                    justify='left').pack(anchor='w')
        
        # Initial display
        update_percent_display()
    
    # ===================== BUTTON 3: Training Metrics for Random Forest =====================
    def show_rf_metrics():
        metrics_window = tk.Toplevel(app)
        metrics_window.title("Random Forest Training Metrics")
        metrics_window.geometry("1400x900")
        metrics_window.configure(bg=ModernUI.BACKGROUND)
        
        # Header
        header = tk.Frame(metrics_window, bg=ModernUI.PRIMARY, height=80)
        header.pack(fill="x")
        header.pack_propagate(False)
        
        tk.Label(header, text="Random Forest Training Metrics (k=3 to 7)",
                font=ModernUI.HEADING_FONT,
                fg='white', 
                bg=ModernUI.PRIMARY).pack(pady=20)
        
        # Content frame
        content_frame = tk.Frame(metrics_window, bg=ModernUI.BACKGROUND)
        content_frame.pack(fill="both", expand=True, padx=30, pady=20)
        
        # Create tabs for different k values
        notebook = ttk.Notebook(content_frame)
        notebook.pack(fill="both", expand=True)
        
        # Create a tab for each k value
        for k in range(3, 8):
            tab_frame = tk.Frame(notebook, bg=ModernUI.BACKGROUND)
            notebook.add(tab_frame, text=f"k={k}")
            
            # Main content for this tab
            main_tab_frame = tk.Frame(tab_frame, bg=ModernUI.BACKGROUND)
            main_tab_frame.pack(fill="both", expand=True, padx=10, pady=10)
            
            # Top metrics
            metrics_top = tk.Frame(main_tab_frame, bg='white', relief='solid', borderwidth=1)
            metrics_top.pack(fill="x", pady=(0, 15))
            
            metrics_content = tk.Frame(metrics_top, bg='white')
            metrics_content.pack(fill="both", expand=True, padx=20, pady=20)
            
            tk.Label(metrics_content, text=f"Random Forest Performance for k={k}",
                    font=("Segoe UI", 16, "bold"),
                    fg=ModernUI.PRIMARY,
                    bg='white').pack(anchor='w', pady=(0, 15))
            
            # Key metrics in grid
            metrics_grid = tk.Frame(metrics_content, bg='white')
            metrics_grid.pack(fill="x")
            
            rf = rf_models[k]
            accuracy = rf_accuracies[k]
            
            metrics_data = [
                ("Training data accuracy:", f"{accuracy*100:.2f}%"),
                ("Number of trees:", "100"),
                ("Average depth:", f"{np.mean([estimator.tree_.max_depth for estimator in rf.estimators_]):.1f}"),
                ("Distinct clusters:", f"{k}")
            ]
            
            for i, (label, value) in enumerate(metrics_data):
                row = i % 2
                col = i // 2
                
                metric_frame = tk.Frame(metrics_grid, bg='white')
                metric_frame.grid(row=row, column=col, sticky='w', padx=10, pady=5)
                
                tk.Label(metric_frame, text=label,
                        font=ModernUI.BODY_FONT,
                        fg=ModernUI.DARK,
                        bg='white').pack(side='left')
                
                tk.Label(metric_frame, text=value,
                        font=("Segoe UI", 11, "bold"),
                        fg=ModernUI.ACCENT,
                        bg='white').pack(side='left')
            
            # Feature importance visualization
            imp_frame = tk.Frame(main_tab_frame, bg='white', relief='solid', borderwidth=1)
            imp_frame.pack(fill="both", expand=True)
            
            imp_content = tk.Frame(imp_frame, bg='white')
            imp_content.pack(fill="both", expand=True, padx=20, pady=20)
            
            tk.Label(imp_content, text="Variable Importance",
                    font=("Segoe UI", 14, "bold"),
                    fg=ModernUI.PRIMARY,
                    bg='white').pack(anchor='w', pady=(0, 10))
            
            # Create plot frame
            plot_container = tk.Frame(imp_content, bg='white')
            plot_container.pack(fill="both", expand=True)
            
            # Calculate feature importances
            importances = rf.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            bars = ax.barh(range(len(importances)), importances[indices], 
                          color=ModernUI.ACCENT, alpha=0.8)
            
            ax.set_yticks(range(len(importances)))
            ax.set_yticklabels([data.columns[i] for i in indices])
            ax.set_xlabel('Importance', fontsize=11, fontweight='bold')
            ax.set_title(f'Relative Importance of Variables (k={k})', 
                        fontsize=13, fontweight='bold', pad=15)
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, (bar, imp) in enumerate(zip(bars, importances[indices])):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{imp:.3f}', va='center', fontsize=10)
            
            plt.tight_layout()
            
            # Embed plot
            canvas = FigureCanvasTkAgg(fig, plot_container)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
            
            # Add toolbar
            toolbar = NavigationToolbar2Tk(canvas, plot_container)
            toolbar.update()
            canvas.get_tk_widget().pack(fill="both", expand=True)
    
    # ===================== BUTTON 4: Prediction of New Individuals =====================
    def predict_new_individual():
        pred_window = tk.Toplevel(app)
        pred_window.title("Prediction of New Individuals with Random Forest")
        pred_window.geometry("1200x800")
        pred_window.configure(bg=ModernUI.BACKGROUND)
        
        # Header
        header = tk.Frame(pred_window, bg=ModernUI.PRIMARY, height=80)
        header.pack(fill="x")
        header.pack_propagate(False)
        
        tk.Label(header, text="Prediction of New Airport Zone",
                font=ModernUI.HEADING_FONT,
                fg='white', 
                bg=ModernUI.PRIMARY).pack(pady=20)
        
        # Main content
        main_frame = tk.Frame(pred_window, bg=ModernUI.BACKGROUND)
        main_frame.pack(fill="both", expand=True, padx=30, pady=20)
        
        # Create PanedWindow for resizable split
        paned = tk.PanedWindow(main_frame, orient=tk.HORIZONTAL, bg=ModernUI.BACKGROUND)
        paned.pack(fill="both", expand=True)
        
        # Left column: Input Form
        left_column = tk.Frame(paned, bg=ModernUI.BACKGROUND)
        
        # Right column: Results
        right_column = tk.Frame(paned, bg=ModernUI.BACKGROUND)
        
        # Add to paned window
        paned.add(left_column, width=600)
        paned.add(right_column, width=500)
        
        # ===== LEFT COLUMN: Input Form =====
        input_frame = tk.Frame(left_column, bg='white', relief='solid', borderwidth=1)
        input_frame.pack(fill="both", expand=True)
        
        input_content = tk.Frame(input_frame, bg='white')
        input_content.pack(fill="both", expand=True, padx=25, pady=25)
        
        tk.Label(input_content, text="New Zone Data Entry",
                font=("Segoe UI", 16, "bold"),
                fg=ModernUI.PRIMARY,
                bg='white').pack(anchor='w', pady=(0, 20))
        
        # Create input fields
        input_vars = {}
        ranges = {
            "Passengers/hour": (10, 20000),
            "Control time (min)": (1, 60),
            "Detections": (0, 500),
            "Agents": (1, 500),
            "Cameras": (1, 2000),
            "Incidents": (0, 100),
            "Area (m²)": (100, 50000),
            "Satisfaction (%)": (0, 100)
        }
        
        # Create scrollable input area
        input_canvas = tk.Canvas(input_content, bg='white', highlightthickness=0)
        input_scrollbar = ttk.Scrollbar(input_content, orient="vertical", command=input_canvas.yview)
        scrollable_input = tk.Frame(input_canvas, bg='white')
        
        scrollable_input.bind(
            "<Configure>",
            lambda e: input_canvas.configure(scrollregion=input_canvas.bbox("all"))
        )
        
        input_canvas.create_window((0, 0), window=scrollable_input, anchor="nw")
        input_canvas.configure(yscrollcommand=input_scrollbar.set)
        
        # Create input fields in scrollable area
        for i, var in enumerate(data.columns):
            row_frame = tk.Frame(scrollable_input, bg='white')
            row_frame.pack(fill="x", pady=8)
            
            # Label
            tk.Label(row_frame, text=var,
                    font=ModernUI.BODY_FONT,
                    fg=ModernUI.DARK,
                    bg='white',
                    width=25,
                    anchor='w').pack(side='left')
            
            # Entry with validation
            min_val, max_val = ranges[var]
            default = str(int(data[var].mean()))
            
            entry_frame = tk.Frame(row_frame, bg='white')
            entry_frame.pack(side='right', fill='x', expand=True)
            
            entry = tk.Entry(entry_frame,
                           font=ModernUI.BODY_FONT,
                           bg='#F8F9FA',
                           fg=ModernUI.DARK,
                           relief='solid',
                           borderwidth=1,
                           justify='right')
            entry.insert(0, default)
            entry.pack(side='left', fill='x', expand=True)
            
            # Range label
            tk.Label(entry_frame, text=f"[{min_val}-{max_val}]",
                    font=("Segoe UI", 9),
                    fg=ModernUI.SECONDARY,
                    bg='white',
                    padx=10).pack(side='right')
            
            input_vars[var] = entry
        
        # Add some padding at the bottom
        tk.Frame(scrollable_input, bg='white', height=20).pack()
        
        input_canvas.pack(side="left", fill="both", expand=True)
        input_scrollbar.pack(side="right", fill="y")
        
        # ===== SINGLE BUTTON AT BOTTOM OF INPUT SECTION =====
        # Create a separate frame for the single button at the bottom
        button_frame = tk.Frame(input_content, bg='white')
        button_frame.pack(fill="x", pady=(20, 0))
        
        def fill_random():
            for var, entry in input_vars.items():
                min_val, max_val = ranges[var]
                rand_val = str(np.random.randint(min_val, max_val + 1))
                entry.delete(0, 'end')
                entry.insert(0, rand_val)
        
        # Create the single "Random" button
        btn_random = tk.Button(button_frame, text="Random",
                 font=ModernUI.BUTTON_FONT,
                 bg=ModernUI.ACCENT,
                 fg='white',
                 command=fill_random,
                 padx=30,
                 pady=12)
        btn_random.pack(fill='x', expand=True)
        
        # ===== RIGHT COLUMN: Prediction Results =====
        result_frame = tk.Frame(right_column, bg='white', relief='solid', borderwidth=1)
        result_frame.pack(fill="both", expand=True)
        
        result_content = tk.Frame(result_frame, bg='white')
        result_content.pack(fill="both", expand=True, padx=25, pady=25)
        
        tk.Label(result_content, text="Prediction Results",
                font=("Segoe UI", 16, "bold"),
                fg=ModernUI.PRIMARY,
                bg='white').pack(anchor='w', pady=(0, 20))
        
        # Results display area
        results_display = tk.Frame(result_content, bg='#F8F9FA', relief='solid', borderwidth=1)
        results_display.pack(fill="both", expand=True)
        
        # Create Text widget with scrollbar
        results_container = tk.Frame(results_display, bg='#F8F9FA')
        results_container.pack(fill="both", expand=True, padx=5, pady=5)
        
        results_text = tk.Text(results_container,
                              wrap="word",
                              font=ModernUI.BODY_FONT,
                              bg='#F8F9FA',
                              fg=ModernUI.DARK,
                              relief='flat')
        
        results_scrollbar = ttk.Scrollbar(results_container, orient="vertical", command=results_text.yview)
        results_text.configure(yscrollcommand=results_scrollbar.set)
        
        results_text.pack(side="left", fill="both", expand=True)
        results_scrollbar.pack(side="right", fill="y")
        
        results_text.insert("1.0", "Enter data and click 'Predict' to see results.")
        results_text.config(state="disabled")
        
        # Prediction button
        predict_btn = tk.Button(result_content,
                               text="Run Prediction",
                               font=ModernUI.BUTTON_FONT,
                               bg=ModernUI.SUCCESS,
                               fg='white',
                               padx=30,
                               pady=15)
        predict_btn.pack(pady=20)
        
        # Prediction function
        def make_prediction():
            try:
                # Gather input values
                input_data = {}
                errors = []
                
                for var, entry in input_vars.items():
                    try:
                        value = float(entry.get())
                        min_val, max_val = ranges[var]
                        
                        if value < min_val or value > max_val:
                            errors.append(f"{var}: must be between {min_val} and {max_val}")
                        else:
                            input_data[var] = value
                    except ValueError:
                        errors.append(f"{var}: numeric value required")
                
                if errors:
                    error_msg = "Input errors:\n" + "\n".join([f"• {e}" for e in errors])
                    results_text.config(state="normal")
                    results_text.delete("1.0", "end")
                    results_text.insert("1.0", error_msg)
                    results_text.config(state="disabled")
                    return
                
                # Create DataFrame
                new_df = pd.DataFrame([input_data])
                new_scaled = scaler.transform(new_df)
                
                # Generate prediction results
                results_output = "RANDOM FOREST PREDICTIONS\n"
                results_output += "=" * 50 + "\n\n"
                
                results_output += "RESULTS BY NUMBER OF CLUSTERS:\n\n"
                
                for k in range(3, 8):
                    rf = rf_models[k]
                    pred = rf.predict(new_scaled)[0]
                    proba = rf.predict_proba(new_scaled)[0]
                    
                    results_output += f"For k={k} clusters:\n"
                    results_output += f"  • Predicted cluster: {pred}\n"
                    results_output += f"  • Probabilities: {np.round(proba, 3)}\n"
                    results_output += f"  • Certainty: {max(proba)*100:.1f}%\n"
                    
                    # Add interpretation
                    if max(proba) > 0.8:
                        results_output += f"  • High confidence\n"
                    elif max(proba) > 0.6:
                        results_output += f"  • Medium confidence\n"
                    else:
                        results_output += f"  • Low confidence\n"
                    
                    results_output += "\n"
                
                results_output += "ENTERED VALUES:\n"
                for var, val in input_data.items():
                    results_output += f"  • {var}: {val:.0f}\n"
                
                results_output += "\nRECOMMENDATIONS:\n"
                results_output += "• Analyze consistency between predictions for different k\n"
                results_output += "• Consider k=4 as operational reference\n"
                results_output += "• Monitor zones with uncertain predictions\n"
                
                # Update results display
                results_text.config(state="normal")
                results_text.delete("1.0", "end")
                results_text.insert("1.0", results_output)
                results_text.config(state="disabled")
                
            except Exception as e:
                error_msg = f"Technical error:\n{str(e)}"
                results_text.config(state="normal")
                results_text.delete("1.0", "end")
                results_text.insert("1.0", error_msg)
                results_text.config(state="disabled")
        
        # Connect button to prediction function
        predict_btn.config(command=make_prediction)
    
    # ===================== MAIN BUTTONS =====================
    buttons_container = tk.Frame(content, bg=ModernUI.BACKGROUND)
    buttons_container.pack(fill="x", pady=20)
    
    # Create 2x2 grid of buttons
    button_grid = tk.Frame(buttons_container, bg=ModernUI.BACKGROUND)
    button_grid.pack()
    
    # Button 1: Show 5 Clusters
    btn1 = tk.Button(button_grid,
                    text="Visualize the 5 Clusters\n(k=3 to k=7)",
                    font=("Segoe UI", 13, "bold"),
                    bg=ModernUI.ACCENT,
                    fg='white',
                    command=show_five_clusters,
                    width=25,
                    height=4,
                    cursor='hand2',
                    relief='flat',
                    padx=20)
    btn1.grid(row=0, column=0, padx=15, pady=15, sticky='nsew')
    
    # Button 2: Percentage of each cluster
    btn2 = tk.Button(button_grid,
                    text="Percentage of each Cluster\non Data",
                    font=("Segoe UI", 13, "bold"),
                    bg=ModernUI.ACCENT,
                    fg='white',
                    command=show_cluster_percentages,
                    width=25,
                    height=4,
                    cursor='hand2',
                    relief='flat',
                    padx=20)
    btn2.grid(row=0, column=1, padx=15, pady=15, sticky='nsew')
    
    # Button 3: Random Forest Training Metrics
    btn3 = tk.Button(button_grid,
                    text="Random Forest Training\nMetrics Display",
                    font=("Segoe UI", 13, "bold"),
                    bg=ModernUI.ACCENT,
                    fg='white',
                    command=show_rf_metrics,
                    width=25,
                    height=4,
                    cursor='hand2',
                    relief='flat',
                    padx=20)
    btn3.grid(row=1, column=0, padx=15, pady=15, sticky='nsew')
    
    # Button 4: Prediction of New Individuals
    btn4 = tk.Button(button_grid,
                    text="Prediction of New Individuals\nwith Random Forest",
                    font=("Segoe UI", 13, "bold"),
                    bg=ModernUI.ACCENT,
                    fg='white',
                    command=predict_new_individual,
                    width=25,
                    height=4,
                    cursor='hand2',
                    relief='flat',
                    padx=20)
    btn4.grid(row=1, column=1, padx=15, pady=15, sticky='nsew')
    
    # Configure grid weights
    button_grid.grid_columnconfigure(0, weight=1)
    button_grid.grid_columnconfigure(1, weight=1)
    button_grid.grid_rowconfigure(0, weight=1)
    button_grid.grid_rowconfigure(1, weight=1)
    
    # Description frame
    desc_frame = tk.Frame(content, bg=ModernUI.BACKGROUND)
    desc_frame.pack(fill="x", pady=(30, 0))

# ===================== AFC CONTENT =====================
def show_afc_content():
    # Create container for buttons with responsive grid
    buttons_container = tk.Frame(content, bg=ModernUI.BACKGROUND)
    buttons_container.pack(fill="x", pady=(20, 0))
    
    # Create 2x4 grid of buttons (8 buttons total)
    button_grid = tk.Frame(buttons_container, bg=ModernUI.BACKGROUND)
    button_grid.pack()
    
    def create_action_button(text, command, row, col):
        btn = tk.Button(button_grid,
                       text=text,
                       font=("Segoe UI", 12, "bold"),
                       bg=ModernUI.ACCENT,
                       fg='white',
                       activebackground='#2471A3',
                       activeforeground='white',
                       relief='flat',
                       cursor='hand2',
                       width=20,
                       height=2,
                       command=command)
        btn.grid(row=row, column=col, padx=10, pady=10, sticky='nsew')
        return btn
    
    # Create 8×8 contingency table (8 zones × 8 variables) with values 1-10
    np.random.seed(42)
    zones = [f"Zone {i+1}" for i in range(8)]
    variables = list(data.columns)
    
    contingency = pd.DataFrame(
        np.random.randint(1, 11, (8, 8)),
        index=zones,
        columns=variables
    )

    # AFC calculations as per TP4
    def calculate_afc():
        # 1. Frequency matrix
        n = contingency.values.sum()
        P = contingency / n
        
        # 2. Row and column masses
        r = P.sum(axis=1).values.reshape(-1, 1)  # row masses
        c = P.sum(axis=0).values.reshape(1, -1)  # column masses
        
        # 3. Independence deviation matrix (χ² distance)
        S = (P - r @ c) / np.sqrt(r @ c)
        
        # 4. SVD decomposition
        U, s, Vt = np.linalg.svd(S, full_matrices=False)
        
        # 5. Eigenvalues and inertia
        eigenvalues = s**2
        inertie_totale = eigenvalues.sum()
        explained_inertia = eigenvalues / inertie_totale
        
        # 6. Chi-square test
        chi2_calc = n * inertie_totale
        chi2_scipy, p_value, ddl, _ = chi2_contingency(contingency)
        
        # 7. Factorial coordinates (F1, F2)
        F = U[:, :2] @ np.diag(s[:2])  # rows
        G = Vt.T[:, :2] @ np.diag(s[:2])  # columns
        
        coords_lignes = pd.DataFrame(F, index=zones, columns=['F1', 'F2'])
        coords_colonnes = pd.DataFrame(G, index=variables, columns=['F1', 'F2'])
        
        return {
            'contingency': contingency,
            'frequences': P,
            'masses_lignes': r,
            'masses_colonnes': c,
            'ecarts': S,
            'eigenvalues': eigenvalues,
            'inertie_totale': inertie_totale,
            'explained_inertia': explained_inertia,
            'chi2_calc': chi2_calc,
            'chi2_scipy': chi2_scipy,
            'p_value': p_value,
            'ddl': ddl,
            'coords_lignes': coords_lignes,
            'coords_colonnes': coords_colonnes
        }
    
    afc_results = calculate_afc()

    def cont():
        show_table(afc_results['contingency'], "8×8 Contingency Table")

    def freq():
        show_table(afc_results['frequences'].round(4), "Frequency Matrix")

    def masses():
        df_masses = pd.DataFrame({
            'Zone': zones,
            'Row Mass': afc_results['masses_lignes'].flatten().round(4)
        })
        show_table(df_masses, "Row Masses")

    def ecarts():
        df_ecarts = pd.DataFrame(afc_results['ecarts'], 
                               index=zones, 
                               columns=variables).round(4)
        show_table(df_ecarts, "Independence Deviation Matrix")

    def inertie():
        df_inertie = pd.DataFrame({
            'Axis': [f'F{i+1}' for i in range(len(afc_results['eigenvalues']))],
            'Eigenvalue': afc_results['eigenvalues'].round(4),
            'Inertia (%)': (afc_results['explained_inertia'] * 100).round(2),
            'Cumulative Inertia (%)': (np.cumsum(afc_results['explained_inertia']) * 100).round(2)
        })
        show_table(df_inertie, "Eigenvalues & Inertia")

    def chi2():
        chi2_text = f"CHI-SQUARE TEST (χ²)\n"
        chi2_text += "=" * 50 + "\n\n"
        chi2_text += f"Calculated Chi²: {afc_results['chi2_calc']:.4f}\n"
        chi2_text += f"Chi² (scipy): {afc_results['chi2_scipy']:.4f}\n"
        chi2_text += f"p-value: {afc_results['p_value']:.6f}\n"
        chi2_text += f"Degrees of freedom: {afc_results['ddl']}\n"
        chi2_text += f"Total inertia: {afc_results['inertie_totale']:.4f}\n\n"
        
        chi2_text += "INTERPRETATION:\n"
        if afc_results['p_value'] < 0.05:
            chi2_text += "• p-value < 0.05 → SIGNIFICANT dependence\n"
            chi2_text += "• Zones and variables are STATISTICALLY ASSOCIATED\n"
        else:
            chi2_text += "• p-value ≥ 0.05 → NON-significant dependence\n"
            chi2_text += "• Zones and variables are INDEPENDENT\n"
        
        chi2_text += f"\nINDICATORS:\n"
        chi2_text += f"• High Chi² ({afc_results['chi2_calc']:.1f}) → Strong association\n"
        chi2_text += f"• Inertia ({afc_results['inertie_totale']:.3f}) → Data structure\n"
        
        chi2_text += "\nAIRPORT IMPLICATIONS:\n"
        chi2_text += "• Security variables are linked to specific zones\n"
        chi2_text += "• Certain zones present distinct security profiles\n"
        chi2_text += "• Possibility to categorize zones by risk type"
        
        show_message("Chi-square Test & Interpretation", chi2_text)

    def plan():
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Plot zones
        ax.scatter(afc_results['coords_lignes']["F1"], 
                  afc_results['coords_lignes']["F2"], 
                  color='blue', s=200, alpha=0.8, edgecolors='k', 
                  linewidth=2, label='Zones')
        
        # Label ALL zones as "Zone X"
        for i, zone in enumerate(zones):
            ax.text(afc_results['coords_lignes']["F1"][i] + 0.02, 
                   afc_results['coords_lignes']["F2"][i] + 0.02, 
                   zone, color='blue', fontsize=12, weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.5))
        
        # Plot variables
        ax.scatter(afc_results['coords_colonnes']["F1"], 
                  afc_results['coords_colonnes']["F2"], 
                  color='red', marker='s', s=200, alpha=0.8, 
                  edgecolors='k', linewidth=2, label='Variables')
        
        # Label ALL variables
        for i, var in enumerate(variables):
            ax.text(afc_results['coords_colonnes']["F1"][i] + 0.02, 
                   afc_results['coords_colonnes']["F2"][i] + 0.02, 
                   var, color='red', fontsize=11,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='mistyrose', alpha=0.5))
        
        ax.axhline(0, color='gray', linewidth=1, linestyle='--', alpha=0.5)
        ax.axvline(0, color='gray', linewidth=1, linestyle='--', alpha=0.5)
        ax.set_xlabel(f'F1 ({afc_results["explained_inertia"][0]*100:.1f}%)', 
                     fontsize=12, fontweight='bold')
        ax.set_ylabel(f'F2 ({afc_results["explained_inertia"][1]*100:.1f}%)', 
                     fontsize=12, fontweight='bold')
        ax.set_title('Factor Plane AFC - Zones vs Variables', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        show_plot(fig)

    def coordonnees():
        show_table(afc_results['coords_lignes'].round(4), 
                  "Factorial Coordinates of Zones")

    # Create action buttons in 2x4 grid
    create_action_button("Contingency Table", cont, 0, 0)
    create_action_button("Frequency Matrix", freq, 0, 1)
    create_action_button("Row Masses", masses, 0, 2)
    create_action_button("Deviation Matrix", ecarts, 0, 3)
    create_action_button("Eigenvalues", inertie, 1, 0)
    create_action_button("Chi-square Test", chi2, 1, 1)
    create_action_button("Factor Plane", plan, 1, 2)
    create_action_button("Coordinates", coordonnees, 1, 3)
    
    # Configure grid weights for responsiveness
    for i in range(4):
        button_grid.columnconfigure(i, weight=1)
    for i in range(2):
        button_grid.rowconfigure(i, weight=1)

# ===================== CYBERSECURITY CONTENT =====================
def show_cybersecurity_content():
    # Create container for buttons with responsive grid
    buttons_container = tk.Frame(content, bg=ModernUI.BACKGROUND)
    buttons_container.pack(fill="x", pady=(20, 0))
    
    # Create 2x3 grid of buttons (6 buttons total)
    button_grid = tk.Frame(buttons_container, bg=ModernUI.BACKGROUND)
    button_grid.pack()
    
    def create_action_button(text, command, row, col):
        btn = tk.Button(button_grid,
                       text=text,
                       font=("Segoe UI", 12, "bold"),
                       bg=ModernUI.ACCENT,
                       fg='white',
                       activebackground='#2471A3',
                       activeforeground='white',
                       relief='flat',
                       cursor='hand2',
                       width=20,
                       height=2,
                       command=command)
        btn.grid(row=row, column=col, padx=10, pady=10, sticky='nsew')
        return btn
    
    # Use airport security data methods
    cyber_df = pd.DataFrame({
        "Zone": [f"Zone {i+1}" for i in range(100)],
        # Map airport variables to cybersecurity concepts
        "Connected devices": data["Cameras"] + data["Agents"],
        "Suspicious attempts": data["Detections"],
        "Response time (min)": data["Control time (min)"],
        "Vulnerability score": (data["Incidents"] * 0.5 + 
                               (100 - data["Satisfaction (%)"]) * 0.3 +
                               data["Detections"] * 0.2) / 100
    })
    
    # Prepare features for anomaly detection
    features = ["Connected devices", "Suspicious attempts", 
                "Response time (min)", "Vulnerability score"]
    
    scaler_cyber = MinMaxScaler()
    X_cyber_scaled = scaler_cyber.fit_transform(cyber_df[features])

    # Isolation Forest (method)
    iso = IsolationForest(contamination=0.15, random_state=42)
    iso_pred = iso.fit_predict(X_cyber_scaled)
    
    # Local Outlier Factor (method)
    lof = LocalOutlierFactor(n_neighbors=10, contamination=0.15)
    lof_pred = lof.fit_predict(X_cyber_scaled)
    
    # Add predictions to dataframe
    cyber_df['IF_Anomaly'] = iso_pred
    cyber_df['LOF_Anomaly'] = lof_pred
    cyber_df['Risk_IF'] = np.where(iso_pred == -1, 'RISK', 'NORMAL')
    cyber_df['Risk_LOF'] = np.where(lof_pred == -1, 'RISK', 'NORMAL')

    def show_data():
        display_df = cyber_df[['Zone', 'Connected devices', 'Suspicious attempts', 
                             'Response time (min)', 'Vulnerability score', 
                             'Risk_IF', 'Risk_LOF']].copy()
        display_df['Vulnerability score'] = display_df['Vulnerability score'].round(3)
        show_table(display_df, "Airport Cybersecurity Data")

    def iso_table():
        display_df = cyber_df[['Zone', 'Connected devices', 'Suspicious attempts', 
                             'Response time (min)', 'Vulnerability score', 'Risk_IF']].copy()
        display_df['Vulnerability score'] = display_df['Vulnerability score'].round(3)
        show_table(display_df, "Results - Isolation Forest")

    def lof_table():
        display_df = cyber_df[['Zone', 'Connected devices', 'Suspicious attempts', 
                             'Response time (min)', 'Vulnerability score', 'Risk_LOF']].copy()
        display_df['Vulnerability score'] = display_df['Vulnerability score'].round(3)
        show_table(display_df, "Results - LOF")

    def iso_graph():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Graph 1: Main view
        colors = ['red' if pred == -1 else 'green' for pred in iso_pred]
        
        # Plot ALL points
        scatter1 = ax1.scatter(cyber_df['Connected devices'], 
                              cyber_df['Suspicious attempts'], 
                              c=colors, s=100, alpha=0.7, edgecolors='k', linewidth=0.5)
        
        # Label ALL points as "Zone X"
        for i, row in cyber_df.iterrows():
            ax1.text(row['Connected devices'] + 5, 
                    row['Suspicious attempts'] + 2, 
                    row['Zone'], fontsize=8, alpha=0.7, color=ModernUI.DARK)
        
        ax1.set_xlabel("Connected devices", fontsize=12, fontweight='bold')
        ax1.set_ylabel("Suspicious attempts", fontsize=12, fontweight='bold')
        ax1.set_title("Isolation Forest - Main View", fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Graph 2: Secondary view
        scatter2 = ax2.scatter(cyber_df['Response time (min)'], 
                              cyber_df['Vulnerability score'], 
                              c=colors, s=100, alpha=0.7, edgecolors='k', linewidth=0.5)
        
        # Label ALL points as "Zone X"
        for i, row in cyber_df.iterrows():
            ax2.text(row['Response time (min)'] + 0.3, 
                    row['Vulnerability score'] + 0.005, 
                    row['Zone'], fontsize=8, alpha=0.7, color=ModernUI.DARK)
        
        ax2.set_xlabel("Response time (min)", fontsize=12, fontweight='bold')
        ax2.set_ylabel("Vulnerability score", fontsize=12, fontweight='bold')
        ax2.set_title("Isolation Forest - Secondary View", fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Create custom legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', alpha=0.7, label='Risk zone'),
                          Patch(facecolor='green', alpha=0.7, label='Normal zone')]
        
        fig.suptitle("Anomaly Detection - Isolation Forest Method", 
                    fontsize=16, fontweight='bold', y=1.02)
        fig.legend(handles=legend_elements, loc='upper center', 
                  bbox_to_anchor=(0.5, 0.98), ncol=2, fontsize=11)
        
        plt.tight_layout()
        show_plot(fig)

    def lof_graph():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Graph 1: Main view
        colors = ['red' if pred == -1 else 'green' for pred in lof_pred]
        
        # Plot ALL points
        scatter1 = ax1.scatter(cyber_df['Connected devices'], 
                              cyber_df['Suspicious attempts'], 
                              c=colors, s=100, alpha=0.7, edgecolors='k', linewidth=0.5)
        
        # Label ALL points as "Zone X"
        for i, row in cyber_df.iterrows():
            ax1.text(row['Connected devices'] + 5, 
                    row['Suspicious attempts'] + 2, 
                    row['Zone'], fontsize=8, alpha=0.7, color=ModernUI.DARK)
        
        ax1.set_xlabel("Connected devices", fontsize=12, fontweight='bold')
        ax1.set_ylabel("Suspicious attempts", fontsize=12, fontweight='bold')
        ax1.set_title("LOF - Main View", fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Graph 2: Secondary view
        scatter2 = ax2.scatter(cyber_df['Response time (min)'], 
                              cyber_df['Vulnerability score'], 
                              c=colors, s=100, alpha=0.7, edgecolors='k', linewidth=0.5)
        
        # Label ALL points as "Zone X"
        for i, row in cyber_df.iterrows():
            ax2.text(row['Response time (min)'] + 0.3, 
                    row['Vulnerability score'] + 0.005, 
                    row['Zone'], fontsize=8, alpha=0.7, color=ModernUI.DARK)
        
        ax2.set_xlabel("Response time (min)", fontsize=12, fontweight='bold')
        ax2.set_ylabel("Vulnerability score", fontsize=12, fontweight='bold')
        ax2.set_title("LOF - Secondary View", fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Create custom legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', alpha=0.7, label='Risk zone'),
                          Patch(facecolor='green', alpha=0.7, label='Normal zone')]
        
        fig.suptitle("Anomaly Detection - LOF Method", 
                    fontsize=16, fontweight='bold', y=1.02)
        fig.legend(handles=legend_elements, loc='upper center', 
                  bbox_to_anchor=(0.5, 0.98), ncol=2, fontsize=11)
        
        plt.tight_layout()
        show_plot(fig)

    def interpretation():
        risky_zones_if = cyber_df[cyber_df['IF_Anomaly'] == -1]['Zone'].tolist()
        risky_zones_lof = cyber_df[cyber_df['LOF_Anomaly'] == -1]['Zone'].tolist()
        common_risky = list(set(risky_zones_if) & set(risky_zones_lof))
        
        analysis = "AIRPORT CYBERSECURITY REPORT\n"
        analysis += "=" * 60 + "\n\n"
        
        analysis += "ANOMALY DETECTION RESULTS:\n"
        analysis += f"• Isolation Forest: {len(risky_zones_if)} risk zones\n"
        analysis += f"• LOF: {len(risky_zones_lof)} risk zones\n\n"
        
        if common_risky:
            analysis += "HIGH RISK ZONES (detected by both methods):\n"
            analysis += f"   {', '.join(common_risky)}\n\n"
        
        analysis += "RISK ZONE CHARACTERISTICS:\n"
        risky_if = cyber_df[cyber_df['IF_Anomaly'] == -1]
        risky_lof = cyber_df[cyber_df['LOF_Anomaly'] == -1]
        
        if len(risky_if) > 0:
            analysis += "• Isolation Forest detects:\n"
            analysis += f"  - Average devices: {risky_if['Connected devices'].mean():.0f}\n"
            analysis += f"  - Average attempts: {risky_if['Suspicious attempts'].mean():.0f}\n"
            analysis += f"  - Average vulnerability: {risky_if['Vulnerability score'].mean():.3f}\n\n"
        
        if len(risky_lof) > 0:
            analysis += "• LOF detects:\n"
            analysis += f"  - Average devices: {risky_lof['Connected devices'].mean():.0f}\n"
            analysis += f"  - Average attempts: {risky_lof['Suspicious attempts'].mean():.0f}\n"
            analysis += f"  - Average vulnerability: {risky_lof['Vulnerability score'].mean():.3f}\n\n"
        
        analysis += "SPECIFIC RECOMMENDATIONS:\n"
        analysis += "1. Complete audit of identified risk zones\n"
        analysis += "2. Reinforcement of video and human surveillance\n"
        analysis += "3. Staff cybersecurity training\n"
        analysis += "4. Update of detection systems and firewalls\n"
        analysis += "5. Cybersecurity incident response plan\n"
        analysis += "6. Continuous 24/7 monitoring of critical zones\n"
        analysis += "7. Regular penetration and vulnerability tests\n\n"
        
        analysis += "APPLIED METHODOLOGY:\n"
        analysis += "• Isolation Forest: Detects anomalies by random isolation\n"
        analysis += "• LOF: Compares local density to identify outliers\n"
        analysis += "• Contamination: 15% of data considered as anomalies\n"
        analysis += "• Variables: 4 key airport security indicators"
        
        show_message("Cybersecurity Report", analysis)

    # Create action buttons in 2x3 grid
    create_action_button("Airport Data", show_data, 0, 0)
    create_action_button("IF Results", iso_table, 0, 1)
    create_action_button("IF Graph", iso_graph, 0, 2)
    create_action_button("LOF Results", lof_table, 1, 0)
    create_action_button("LOF Graph", lof_graph, 1, 1)
    create_action_button("Report & Interpretation", interpretation, 1, 2)
    
    # Configure grid weights for responsiveness
    for i in range(3):
        button_grid.columnconfigure(i, weight=1)
    for i in range(2):
        button_grid.rowconfigure(i, weight=1)

# ===================== INITIALIZE APPLICATION =====================
# Show welcome screen initially
show_welcome()

# Start the application
app.mainloop()