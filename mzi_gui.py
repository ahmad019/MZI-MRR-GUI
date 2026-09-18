import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import sympy as sp
import pandas as pd

class MZIAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MZI Ring Resonator Analysis Tool")
        self.root.geometry("1400x900")
        
        # Initialize symbolic variables
        self.setup_symbolic_variables()
        
        # Initialize parameter variables
        self.setup_variables()
        
        # Create GUI layout
        self.create_widgets()
        
        # Initial calculation
        self.update_plots()
    
    def setup_symbolic_variables(self):
        """Initialize all symbolic variables and expressions"""
        # Symbolic Definitions
        self.r1, self.r2, self.r3, self.r4 = sp.symbols('r1 r2 r3 r4', real=True)
        self.a1, self.a2 = sp.symbols('a1 a2', real=True)
        self.phi1, self.phi2 = sp.symbols('phi1 phi2', real=True)
        
        self.t1, self.t2 = sp.sqrt(1 - self.r1**2), sp.sqrt(1 - self.r2**2)
        self.t3, self.t4 = sp.sqrt(1 - self.r3**2), sp.sqrt(1 - self.r4**2)
        
        self.Phi1, self.Phi2 = sp.symbols('Phi1 Phi2', real=True)
        
        # Transmission functions
        self.trans1 = (self.r1 - self.a1 * sp.exp(sp.I * self.phi1)) / (1 - self.r1 * self.a1 * sp.exp(sp.I * self.phi1))
        self.trans2 = (self.r2 - self.a2 * sp.exp(sp.I * self.phi2)) / (1 - self.r2 * self.a2 * sp.exp(sp.I * self.phi2))
        
        # Amplitude transmission terms
        self.A_tau1 = sp.sqrt((self.a1**2 + self.r1**2 - 2 * self.a1 * self.r1 * sp.cos(self.phi1)) /
                        (1 + (self.a1 * self.r1)**2 - 2 * self.a1 * self.r1 * sp.cos(self.phi1)))
        self.A_tau2 = sp.sqrt((self.a2**2 + self.r2**2 - 2 * self.a2 * self.r2 * sp.cos(self.phi2)) /
                        (1 + (self.a2 * self.r2)**2 - 2 * self.a2 * self.r2 * sp.cos(self.phi2)))
        
        # Amplitude of reflected field
        self.A_rho1 = sp.sqrt((sp.sqrt(self.a1**2) * self.t1**2 * self.t3**2) /
                        (1 + self.a1**2 * self.r1**2 * self.r3**2 - 2 * self.a1 * self.r1 * self.r3 * sp.cos(self.phi1)))
        self.A_rho2 = sp.sqrt((sp.sqrt(self.a2**2) * self.t2**2 * self.t4**2) /
                        (1 + self.a2**2 * self.r2**2 * self.r4**2 - 2 * self.a2 * self.r2 * self.r4 * sp.cos(self.phi2)))
        
        # Phase expressions for transmission (OC and UC cases)
        self.phi1A = (sp.pi + self.phi1 + sp.atan((self.r1 * sp.sin(self.phi1)) / (self.a1 - self.r1 * sp.cos(self.phi1))) +
                      sp.atan((self.a1 * self.r1 * sp.sin(self.phi1)) / (1 - self.a1 * self.r1 * sp.cos(self.phi1))))
        
        self.phi1B = sp.atan((self.a1 * (self.r1**2 - 1) * sp.sin(self.phi1)) /
                        (((1 + self.a1**2) * self.r1) - (self.a1 * (1 + self.r1**2) * sp.cos(self.phi1))))
        
        self.phi2A = (sp.pi + self.phi2 + sp.atan((self.r2 * sp.sin(self.phi2)) / (self.a2 - self.r2 * sp.cos(self.phi2))) +
                      sp.atan((self.a2 * self.r2 * sp.sin(self.phi2)) / (1 - self.a2 * self.r2 * sp.cos(self.phi2))))
        
        self.phi2B = sp.atan((self.a2 * (self.r2**2 - 1) * sp.sin(self.phi2)) /
                        (((1 + self.a2**2) * self.r2) - (self.a2 * (1 + self.r2**2) * sp.cos(self.phi2))))
        
        # Reflection phases (independent of OC/UC state, depends on r3/r4)
        # For reflection, the phase is determined by the round-trip and coupling geometry
        self.phir1 = self.phi1 / 2 + sp.atan((self.a1 * self.r1 * self.r3 * sp.sin(self.phi1)) /
                                            (1 - self.a1 * self.r1 * self.r3 * sp.cos(self.phi1)))
        self.phir2 = self.phi2 / 2 + sp.atan((self.a2 * self.r2 * self.r4 * sp.sin(self.phi2)) /
                                            (1 - self.a2 * self.r2 * self.r4 * sp.cos(self.phi2)))
        
        # Interferometric expressions
        self.Theta_transA = sp.atan((self.A_tau1 * sp.sin(self.Phi1) + self.A_tau2 * sp.sin(self.Phi2)) /
                                   (self.A_tau1 * sp.cos(self.Phi1) + self.A_tau2 * sp.cos(self.Phi2)))
        
        self.Theta_transB = sp.atan((self.A_tau1 * sp.sin(self.Phi1) + self.A_rho2 * sp.sin(self.Phi2)) /
                                   (self.A_tau1 * sp.cos(self.Phi1) + self.A_rho2 * sp.cos(self.Phi2)))
        
        self.Theta_transC = sp.atan((self.A_rho1 * sp.sin(self.Phi1) + self.A_rho2 * sp.sin(self.Phi2)) /
                                   (self.A_rho1 * sp.cos(self.Phi1) + self.A_rho2 * sp.cos(self.Phi2)))
        
        self.IoutA = (1/4) * (self.A_tau1**2 + self.A_tau2**2 + 2 * self.A_tau1 * self.A_tau2 * sp.cos(self.Phi1 - self.Phi2))
        self.IoutB = (1/4) * (self.A_tau1**2 + self.A_rho2**2 + 2 * self.A_tau1 * self.A_rho2 * sp.cos(self.Phi1 - self.Phi2))
        self.IoutC = (1/4) * (self.A_rho1**2 + self.A_rho2**2 + 2 * self.A_rho1 * self.A_rho2 * sp.cos(self.Phi1 - self.Phi2))
    
    def setup_variables(self):
        """Initialize GUI variables"""
        # Physical parameters
        self.lambda0 = tk.DoubleVar(value=1548.36e-9)
        self.lambda_res1 = tk.DoubleVar(value=1550e-9)
        self.lambda_res2 = tk.DoubleVar(value=1550e-9)
        self.R1 = tk.DoubleVar(value=15e-6)
        self.R2 = tk.DoubleVar(value=15e-6)
        self.n = tk.DoubleVar(value=2.9)
        
        # Attenuations
        self.alpha1 = tk.DoubleVar(value=0.1)
        self.alpha2 = tk.DoubleVar(value=0.1)
        
        # Resonator parameters
        self.r1_val = tk.DoubleVar(value=0.998)
        self.r2_val = tk.DoubleVar(value=0.9)
        self.r3_val = tk.DoubleVar(value=1.0)
        self.r4_val = tk.DoubleVar(value=1.0)
        self.Q1_val = tk.DoubleVar(value=5e5)
        self.Q2_val = tk.DoubleVar(value=6e5)
        
        # Phase shifters
        self.eta1 = tk.StringVar(value="0")
        self.eta2 = tk.StringVar(value="0")
        
        # Sweep parameters
        self.phi1min = tk.DoubleVar(value=-0.5)
        self.phi1max = tk.DoubleVar(value=0.5)
        self.phi1_steps = tk.IntVar(value=12000)
        
        # Configuration
        self.config = tk.StringVar(value='A')
        self.force_res1_phase = tk.StringVar(value='Auto')
        self.force_res2_phase = tk.StringVar(value='Auto')
        
        # Plot colors
        self.color_scheme = tk.StringVar(value='Default')
        
        # Status variables
        self.status_text = tk.StringVar(value="Ready")
        self.res1_coupling = tk.StringVar(value="")
        self.res2_coupling = tk.StringVar(value="")
    
    def get_phase_value(self, phase_str):
        """Convert phase string to numerical value"""
        phase_dict = {
            "0": 0,
            "π/4": np.pi/4,
            "π/2": np.pi/2,
            "3π/4": 3*np.pi/4,
            "π": np.pi,
            "5π/4": 5*np.pi/4,
            "3π/2": 3*np.pi/2,
            "7π/4": 7*np.pi/4,
            "2π": 2*np.pi
        }
        return phase_dict.get(phase_str, 0)
    
    def create_widgets(self):
        """Create the main GUI layout"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel for controls
        control_frame = ttk.Frame(main_frame, width=350)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        control_frame.pack_propagate(False)
        
        # Right panel for plots
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.create_control_panel(control_frame)
        self.create_plot_panel(plot_frame)
        
        # Status bar
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(status_frame, textvariable=self.status_text).pack(side=tk.LEFT)
        ttk.Label(status_frame, textvariable=self.res1_coupling).pack(side=tk.RIGHT, padx=(0, 10))
        ttk.Label(status_frame, textvariable=self.res2_coupling).pack(side=tk.RIGHT, padx=(0, 10))
    
    def create_control_panel(self, parent):
        """Create the left control panel"""
        # Create scrollable frame
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Title
        title_label = ttk.Label(scrollable_frame, text="MZI Analysis Parameters", font=('Arial', 12, 'bold'))
        title_label.pack(pady=(0, 10))

        # System diagram - shows the currently selected optical configuration
        self.create_system_diagram(scrollable_frame)

        # Configuration selection
        config_frame = ttk.LabelFrame(scrollable_frame, text="Configuration", padding=5)
        config_frame.pack(fill=tk.X, pady=(0, 10))
        
        config_descriptions = {
            'A': 'Both sides coupled',
            'B': 'Upper coupled, lower inserted', 
            'C': 'Both inserted'
        }
        
        for config in ['A', 'B', 'C']:
            text = f"Config {config}: {config_descriptions[config]}"
            ttk.Radiobutton(config_frame, text=text, variable=self.config, 
                           value=config, command=self.update_plots).pack(anchor=tk.W)
        
        # Physical parameters
        phys_frame = ttk.LabelFrame(scrollable_frame, text="Physical Parameters", padding=5)
        phys_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.create_slider_with_entry(phys_frame, "λ₀ (nm)", self.lambda0, 1400e-9, 1700e-9, 1e-9, scientific=True, scale=1e9)
        self.create_slider_with_entry(phys_frame, "n", self.n, 1.0, 4.0, 0.01)
        self.create_slider_with_entry(phys_frame, "R1 (μm)", self.R1, 5e-6, 50e-6, 1e-6, scientific=True, scale=1e6)
        self.create_slider_with_entry(phys_frame, "R2 (μm)", self.R2, 5e-6, 50e-6, 1e-6, scientific=True, scale=1e6)
        
        # Resonator 1 parameters
        res1_frame = ttk.LabelFrame(scrollable_frame, text="Resonator 1", padding=5)
        res1_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.create_slider_with_entry(res1_frame, "r1", self.r1_val, 0.0, 1.0, 0.001)
        self.create_log_slider_for_Q(res1_frame, "Q1", self.Q1_val, q_min=1e1, q_max=9e9)
        self.create_slider_with_entry(res1_frame, "α1 (dB/cm)", self.alpha1, 0.01, 100.0, 0.01)
        self.create_slider_with_entry(res1_frame, "λres1 (nm)", self.lambda_res1, 1400e-9, 1700e-9, 1e-9, scientific=True, scale=1e9)
        
        ttk.Label(res1_frame, text="Phase Type:").pack(anchor=tk.W)
        phase1_frame = ttk.Frame(res1_frame)
        phase1_frame.pack(fill=tk.X)
        for phase in ['Auto', 'OC', 'UC']:
            ttk.Radiobutton(phase1_frame, text=phase, variable=self.force_res1_phase, 
                           value=phase, command=self.update_plots).pack(side=tk.LEFT)
        
        # Resonator 2 parameters
        res2_frame = ttk.LabelFrame(scrollable_frame, text="Resonator 2", padding=5)
        res2_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.create_slider_with_entry(res2_frame, "r2", self.r2_val, 0.0, 1.0, 0.001)
        self.create_log_slider_for_Q(res2_frame, "Q2", self.Q2_val, q_min=1e1, q_max=9e9)
        self.create_slider_with_entry(res2_frame, "α2 (dB/cm)", self.alpha2, 0.01, 100.0, 0.01)
        self.create_slider_with_entry(res2_frame, "λres2 (nm)", self.lambda_res2, 1400e-9, 1700e-9, 1e-9, scientific=True, scale=1e9)
        
        ttk.Label(res2_frame, text="Phase Type:").pack(anchor=tk.W)
        phase2_frame = ttk.Frame(res2_frame)
        phase2_frame.pack(fill=tk.X)
        for phase in ['Auto', 'OC', 'UC']:
            ttk.Radiobutton(phase2_frame, text=phase, variable=self.force_res2_phase, 
                           value=phase, command=self.update_plots).pack(side=tk.LEFT)
        
        # Additional coupling parameters
        add_frame = ttk.LabelFrame(scrollable_frame, text="Additional Coupling (r3, r4)", padding=5)
        add_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.create_slider_with_entry(add_frame, "r3", self.r3_val, 0.0, 1.0, 0.001)
        self.create_slider_with_entry(add_frame, "r4", self.r4_val, 0.0, 1.0, 0.001)
        
        ttk.Label(add_frame, text="Note: r3, r4 are essential for Config B & C", font=('Arial', 8)).pack(anchor=tk.W)
        
        # Phase shifters
        phase_shift_frame = ttk.LabelFrame(scrollable_frame, text="Phase Shifters", padding=5)
        phase_shift_frame.pack(fill=tk.X, pady=(0, 10))
        
        # η1 dropdown
        eta1_frame = ttk.Frame(phase_shift_frame)
        eta1_frame.pack(fill=tk.X, pady=2)
        ttk.Label(eta1_frame, text="η1:", width=8).pack(side=tk.LEFT)
        eta1_combo = ttk.Combobox(eta1_frame, textvariable=self.eta1, 
                                 values=["0", "π/4", "π/2", "3π/4", "π", "5π/4", "3π/2", "7π/4", "2π"],
                                 state="readonly", width=10)
        eta1_combo.pack(side=tk.RIGHT)
        eta1_combo.bind('<<ComboboxSelected>>', lambda e: self.update_plots())
        
        # η2 dropdown
        eta2_frame = ttk.Frame(phase_shift_frame)
        eta2_frame.pack(fill=tk.X, pady=2)
        ttk.Label(eta2_frame, text="η2:", width=8).pack(side=tk.LEFT)
        eta2_combo = ttk.Combobox(eta2_frame, textvariable=self.eta2, 
                                 values=["0", "π/4", "π/2", "3π/4", "π", "5π/4", "3π/2", "7π/4", "2π"],
                                 state="readonly", width=10)
        eta2_combo.pack(side=tk.RIGHT)
        eta2_combo.bind('<<ComboboxSelected>>', lambda e: self.update_plots())
        
        # Sweep range
        sweep_frame = ttk.LabelFrame(scrollable_frame, text="Sweep Range", padding=5)
        sweep_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.create_entry(sweep_frame, "φ1 min", self.phi1min)
        self.create_entry(sweep_frame, "φ1 max", self.phi1max)
        self.create_entry(sweep_frame, "Steps", self.phi1_steps)
        
        # Plot options
        plot_frame = ttk.LabelFrame(scrollable_frame, text="Plot Options", padding=5)
        plot_frame.pack(fill=tk.X, pady=(0, 10))
        
        color_frame = ttk.Frame(plot_frame)
        color_frame.pack(fill=tk.X, pady=2)
        ttk.Label(color_frame, text="Colors:", width=8).pack(side=tk.LEFT)
        color_combo = ttk.Combobox(color_frame, textvariable=self.color_scheme,
                                  values=["Default", "Dark", "Bright", "Scientific"],
                                  state="readonly", width=12)
        color_combo.pack(side=tk.RIGHT)
        color_combo.bind('<<ComboboxSelected>>', lambda e: self.update_plots())
        
        # Control buttons
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(button_frame, text="Update Plots", command=self.update_plots).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Show Zoomed", command=self.show_zoomed).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Export Data", command=self.export_data).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Reset Parameters", command=self.reset_parameters).pack(fill=tk.X, pady=2)
        
        # Bind mousewheel to canvas for scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _bind_to_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        def _unbind_from_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
        
        canvas.bind('<Enter>', _bind_to_mousewheel)
        canvas.bind('<Leave>', _unbind_from_mousewheel)

    def create_system_diagram(self, parent):
        """Create the small panel that shows a schematic of the currently
        selected optical configuration (A/B/C). Diagrams are pre-drawn
        images stored in the 'assets' folder next to this script."""
        diagram_frame = ttk.LabelFrame(parent, text="Current System", padding=5)
        diagram_frame.pack(fill=tk.X, pady=(0, 10))

        self.diagram_label = ttk.Label(diagram_frame, anchor='center', justify='center')
        self.diagram_label.pack(fill=tk.BOTH, expand=True)

        # Preload the config diagrams once, keyed by config letter. A config
        # with no image available (e.g. 'C' until one is provided) simply
        # won't have an entry, and draw_system_diagram() falls back to a
        # text description instead.
        #
        # These are loaded with Tkinter's own built-in PhotoImage (PNG only)
        # rather than Pillow's ImageTk, since ImageTk requires Pillow's Tk
        # bindings to exactly match the Tcl/Tk build Python is linked
        # against. That mismatch is a common source of a cryptic
        # "TclError: Bad mode" crash, especially in conda/Anaconda
        # environments (e.g. Spyder) where Pillow was installed via pip.
        # Using tk.PhotoImage avoids that dependency entirely. The images
        # are pre-sized to the panel width ahead of time (see assets/ —
        # regenerate with Pillow offline if you replace them).
        self.config_images = {}
        assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
        image_files = {'A': 'config-A.png', 'B': 'config-B.png', 'C': 'config-C.png'}

        for cfg, filename in image_files.items():
            path = os.path.join(assets_dir, filename)
            if not os.path.isfile(path):
                continue
            try:
                self.config_images[cfg] = tk.PhotoImage(file=path)
            except Exception as e:
                print(f"Could not load diagram for Config {cfg}: {e}")

    def draw_system_diagram(self):
        """Update the system panel to match the currently selected
        configuration (called whenever the plots are recalculated)."""
        if not hasattr(self, 'diagram_label'):
            return

        config = self.config.get()
        img = self.config_images.get(config)

        if img is not None:
            self.diagram_label.configure(image=img, text='')
            self.diagram_label.image = img  # keep a reference so it isn't garbage-collected
        else:
            config_desc = {
                'A': 'Both resonators side-coupled',
                'B': 'Res1 side-coupled, Res2 inserted',
                'C': 'Both resonators inserted'
            }
            self.diagram_label.configure(
                image='',
                text=f"Config {config}: {config_desc.get(config, '')}\n\n(diagram not yet available)",
                font=('Arial', 9)
            )
            self.diagram_label.image = None

    def create_slider_with_entry(self, parent, label, variable, min_val, max_val, resolution, scientific=False, scale=1):
        """Create a slider with editable entry box"""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(frame, text=f"{label}:", width=12).pack(side=tk.LEFT)
        
        # Scale widget
        scale_widget = ttk.Scale(frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL, 
                               variable=variable, command=lambda x: self.on_slider_change())
        scale_widget.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        
        # Entry widget for manual input
        entry_var = tk.StringVar()
        if scientific:
            entry_var.set(f"{variable.get()*scale:.3g}")
        else:
            entry_var.set(f"{variable.get():.4f}")
            
        entry = ttk.Entry(frame, textvariable=entry_var, width=10)
        entry.pack(side=tk.RIGHT)
        
        # Bind entry changes
        def on_entry_change(event):
            try:
                val = float(entry_var.get())
                if scientific:
                    val = val / scale
                if min_val <= val <= max_val:
                    variable.set(val)
                    self.update_plots()
                else:
                    # Reset to current value if out of range
                    if scientific:
                        entry_var.set(f"{variable.get()*scale:.3g}")
                    else:
                        entry_var.set(f"{variable.get():.4f}")
            except ValueError:
                # Reset to current value if invalid
                if scientific:
                    entry_var.set(f"{variable.get()*scale:.3g}")
                else:
                    entry_var.set(f"{variable.get():.4f}")
        
        entry.bind('<Return>', on_entry_change)
        entry.bind('<FocusOut>', on_entry_change)
        
        # Store references for updates
        scale_widget.entry_var = entry_var
        scale_widget.scientific = scientific
        scale_widget.scale_factor = scale
        
    def create_log_slider_for_Q(self, parent, label, variable, q_min=1e1, q_max=9e9):
        """
        Create a logarithmic slider for Q (so Q spans many decades but is easy to control).
        The slider internally works in log10-space; 'variable' stores the actual Q value.
        """
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)

        ttk.Label(frame, text=f"{label}:", width=12).pack(side=tk.LEFT)

        log_min = np.log10(q_min)
        log_max = np.log10(q_max)

        log_var = tk.DoubleVar(value=float(np.log10(variable.get())))

        def on_log_scale(x):
            qval = 10.0 ** log_var.get()
            variable.set(qval)
            entry_var.set(f"{qval:.3g}")
            self.on_slider_change()

        scale_widget = ttk.Scale(frame, from_=log_min, to=log_max, orient=tk.HORIZONTAL,
                                 variable=log_var, command=lambda x: on_log_scale(x))
        scale_widget.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5,5))

        entry_var = tk.StringVar(value=f"{variable.get():.3g}")
        entry = ttk.Entry(frame, textvariable=entry_var, width=12)
        entry.pack(side=tk.RIGHT)

        def on_entry_change(event=None):
            try:
                val = float(entry_var.get())
                if q_min <= val <= q_max:
                    variable.set(val)
                    log_var.set(np.log10(val))
                    self.update_plots()
                else:
                    entry_var.set(f"{variable.get():.3g}")
            except Exception:
                entry_var.set(f"{variable.get():.3g}")

        entry.bind('<Return>', on_entry_change)
        entry.bind('<FocusOut>', on_entry_change)

        scale_widget.entry_var = entry_var
        scale_widget.scientific = True
        scale_widget.scale_factor = 1.0

    def create_entry(self, parent, label, variable):
        """Create a labeled entry field"""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(frame, text=f"{label}:", width=8).pack(side=tk.LEFT)
        entry = ttk.Entry(frame, textvariable=variable, width=12)
        entry.pack(side=tk.RIGHT)
        entry.bind('<Return>', lambda e: self.update_plots())
    
    def create_plot_panel(self, parent):
        """Create the plotting panel"""
        # Create matplotlib figure
        self.fig = Figure(figsize=(12, 8), dpi=100, constrained_layout=True)
        # layout handled by constrained_layout=True
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Navigation toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, parent)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.toolbar = toolbar
    
    def get_color_scheme(self):
        """Get color scheme based on selection"""
        schemes = {
            'Default': ['blue', 'red', 'green', 'magenta', 'cyan', 'orange', 'purple'],
            'Dark': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2'],
            'Bright': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF'],
            'Scientific': ['#0d47a1', '#d32f2f', '#388e3c', '#7b1fa2', '#f57c00', '#c2185b', '#5d4037']
        }
        return schemes.get(self.color_scheme.get(), schemes['Default'])
    
    def on_slider_change(self):
        """Handle slider changes"""
        # Update entry labels for sliders with entries
        for child in self.root.winfo_children():
            self.update_slider_entries(child)
        
        # Update plots with slight delay to avoid too frequent updates
        self.root.after_idle(self.update_plots)
    
    def update_slider_entries(self, widget):
        """Recursively update slider entry values"""
        try:
            for child in widget.winfo_children():
                if isinstance(child, ttk.Scale) and hasattr(child, 'entry_var'):
                    if hasattr(child, 'scientific') and child.scientific:
                        child.entry_var.set(f"{child.get() * child.scale_factor:.3g}")
                    else:
                        child.entry_var.set(f"{child.get():.4f}")
                else:
                    self.update_slider_entries(child)
        except:
            pass
    
    def calculate_system(self):
        """Perform the main MZI calculation"""
        try:
            self.status_text.set("Calculating...")
            self.root.update()
            
            # Get parameter values
            lambda0_val = self.lambda0.get()
            lambda_res1_val = self.lambda_res1.get()
            lambda_res2_val = self.lambda_res2.get()
            R1_val = self.R1.get()
            R2_val = self.R2.get()
            n_val = self.n.get()
            r1_val = self.r1_val.get()
            r2_val = self.r2_val.get()
            r3_val = self.r3_val.get()
            r4_val = self.r4_val.get()
            Q1_val = self.Q1_val.get()
            Q2_val = self.Q2_val.get()
            alpha1_val = self.alpha1.get()
            alpha2_val = self.alpha2.get()
            
            # Phase shifter values
            eta1_val = self.get_phase_value(self.eta1.get())
            eta2_val = self.get_phase_value(self.eta2.get())
            
            # Calculate phi1 array
            phi1_vals = np.linspace(self.phi1min.get(), self.phi1max.get(), self.phi1_steps.get())
            
            # Calculate round-trip losses from multiple sources
            d1 = 2 * np.pi * R1_val
            d2 = 2 * np.pi * R2_val
            
            # Loss from Q-factor
            alpha1_q = (2 * np.pi * n_val) / (lambda0_val * Q1_val)
            alpha2_q = (2 * np.pi * n_val) / (lambda0_val * Q2_val)
            
            # Additional material loss (convert dB/cm to 1/m)
            alpha1_mat = alpha1_val * np.log(10) / 10 * 100  # Convert dB/cm to 1/m
            alpha2_mat = alpha2_val * np.log(10) / 10 * 100
            
            # Total loss
            alpha1_total = alpha1_q + alpha1_mat
            alpha2_total = alpha2_q + alpha2_mat
            
            # Round-trip amplitude
            a1_val = np.exp(-alpha1_total * d1 / 2)
            a2_val = np.exp(-alpha2_total * d2 / 2)
            
            # Determine coupling states
            r1_minus_a1 = r1_val - a1_val
            r2_minus_a2 = r2_val - a2_val
            is_res1_overcoupled = r1_minus_a1 < 0
            is_res2_overcoupled = r2_minus_a2 < 0
            
            # Update status
            self.res1_coupling.set(f"Res1: {'OC' if is_res1_overcoupled else 'UC'} (r-a={r1_minus_a1:.4f})")
            self.res2_coupling.set(f"Res2: {'OC' if is_res2_overcoupled else 'UC'} (r-a={r2_minus_a2:.4f})")
            
            # Determine phase types
            if self.force_res1_phase.get() == 'Auto':
                use_res1_oc = is_res1_overcoupled
            else:
                use_res1_oc = (self.force_res1_phase.get() == 'OC')
                
            if self.force_res2_phase.get() == 'Auto':
                use_res2_oc = is_res2_overcoupled
            else:
                use_res2_oc = (self.force_res2_phase.get() == 'OC')
            
            # Select transmission phase expressions
            if use_res1_oc:
                phase1_expr = self.phi1A
            else:
                phase1_expr = self.phi1B
                
            if use_res2_oc:
                phase2_expr = self.phi2A
            else:
                phase2_expr = self.phi2B
                
            phase2_sub = phase2_expr.subs(self.phi2, self.phi1)
            
            # Build configuration with proper parameter usage
            config = self.config.get()
            if config == 'A':
                # Both sides coupled - use transmission phases with phase shifters
                theta_expr = self.Theta_transA.subs({
                    self.Phi1: phase1_expr + eta1_val,
                    self.Phi2: phase2_sub + eta2_val,
                    self.A_tau1: self.A_tau1,
                    self.A_tau2: self.A_tau2.subs(self.phi2, self.phi1)
                })
                iout_expr = self.IoutA.subs({
                    self.Phi1: phase1_expr + eta1_val,
                    self.Phi2: phase2_sub + eta2_val,
                    self.A_tau1: self.A_tau1,
                    self.A_tau2: self.A_tau2.subs(self.phi2, self.phi1)
                })
                lambda_params = (self.phi1, self.r1, self.r2, self.a1, self.a2)
                eval_params = (phi1_vals, r1_val, r2_val, a1_val, a2_val)
                
            elif config == 'B':
                # Upper coupled, lower inserted - use transmission (Res1) + reflection (Res2, with r2 & r4)
                theta_expr = self.Theta_transB.subs({
                    self.Phi1: phase1_expr + eta1_val,
                    self.Phi2: self.phir2.subs(self.phi2, self.phi1) + eta2_val,
                    self.A_tau1: self.A_tau1,
                    self.A_rho2: self.A_rho2.subs(self.phi2, self.phi1)
                })
                iout_expr = self.IoutB.subs({
                    self.Phi1: phase1_expr + eta1_val,
                    self.Phi2: self.phir2.subs(self.phi2, self.phi1) + eta2_val,
                    self.A_tau1: self.A_tau1,
                    self.A_rho2: self.A_rho2.subs(self.phi2, self.phi1)
                })
                # ⚠ include both r2 and r4 in lambda_params/eval_params
                lambda_params = (self.phi1, self.r1, self.r2, self.r4, self.a1, self.a2)
                eval_params = (phi1_vals, r1_val, r2_val, r4_val, a1_val, a2_val)

                
            else:  # Config C
                # Both inserted - use reflection phases (requires r3 and r4)
                theta_expr = self.Theta_transC.subs({
                    self.Phi1: self.phir1 + eta1_val,
                    self.Phi2: self.phir2.subs(self.phi2, self.phi1) + eta2_val,
                    self.A_rho1: self.A_rho1,
                    self.A_rho2: self.A_rho2.subs(self.phi2, self.phi1)
                })
                iout_expr = self.IoutC.subs({
                    self.Phi1: self.phir1 + eta1_val,
                    self.Phi2: self.phir2.subs(self.phi2, self.phi1) + eta2_val,
                    self.A_rho1: self.A_rho1,
                    self.A_rho2: self.A_rho2.subs(self.phi2, self.phi1)
                })
                lambda_params = (self.phi1, self.r1, self.r2, self.r3, self.r4, self.a1, self.a2)
                eval_params = (phi1_vals, r1_val, r2_val, r3_val, r4_val, a1_val, a2_val)
            
            # Final substitution
            theta_expr = theta_expr.subs(self.phi2, self.phi1)
            iout_expr = iout_expr.subs(self.phi2, self.phi1)
            
            # Lambdify and evaluate
            theta_func = sp.lambdify(lambda_params, theta_expr, 'numpy')
            iout_func = sp.lambdify(lambda_params, iout_expr, 'numpy')
            
            # Individual resonator functions
            trans1_func = sp.lambdify((self.phi1, self.r1, self.a1), abs(self.trans1)**2, 'numpy')
            trans2_func = sp.lambdify((self.phi1, self.r2, self.a2), abs(self.trans2.subs(self.phi2, self.phi1))**2, 'numpy')
            
            # Phase functions for plotting
            if config == 'A':
                phase1_plot_expr = phase1_expr
                phase2_plot_expr = phase2_sub
            elif config == 'B':
                phase1_plot_expr = phase1_expr  # Transmission phase for res1
                phase2_plot_expr = self.phir2.subs(self.phi2, self.phi1)  # Reflection phase for res2
            else:  # Config C
                phase1_plot_expr = self.phir1  # Reflection phase for res1
                phase2_plot_expr = self.phir2.subs(self.phi2, self.phi1)  # Reflection phase for res2
            
            phase1_func = sp.lambdify(lambda_params, phase1_plot_expr, 'numpy')
            phase2_func = sp.lambdify(lambda_params, phase2_plot_expr, 'numpy')
            
            # Calculate results
            theta_vals = theta_func(*eval_params)
            iout_vals = iout_func(*eval_params)
            trans1_vals = trans1_func(phi1_vals, r1_val, a1_val)
            trans2_vals = trans2_func(phi1_vals, r2_val, a2_val)
            phase1_vals = phase1_func(*eval_params)
            phase2_vals = phase2_func(*eval_params)
            
            # Group index calculation (more robust)
            # ---------- after theta, phase, transmission calculations ----------
            # Numerical derivative step: compute dθ/dφ
            dphi = phi1_vals[1] - phi1_vals[0]
            dtheta_dphi_num = np.gradient(theta_vals, dphi)  # dθ/dφ

            # Approximate group index (device) using n_eff * dθ/dφ
            c = 299792458.0
            n_eff = n_val
            L_rt = d1  # Use resonator 1 round-trip length as reference (same φ variable)
            ng_num = n_eff * dtheta_dphi_num

            # Analytical symbolic derivative (if possible)
            try:
                dtheta_dphi_sym = sp.diff(theta_expr, self.phi1)
                ng_sym_expr = n_eff * dtheta_dphi_sym
                ng_func = sp.lambdify(lambda_params, ng_sym_expr, 'numpy')
                ng_vals = ng_func(*eval_params)
            except Exception:
                ng_vals = ng_num

            # Group delay τ_g = dθ/dω ≈ (dθ/dφ) * (n_eff * L_rt / c)
            tau_g_num = dtheta_dphi_num * (n_eff * L_rt / c)

            # Group velocity v_g = c / (n_eff * dθ/dφ)
            with np.errstate(divide='ignore', invalid='ignore'):
                v_g_num = c / (n_eff * dtheta_dphi_num)

            self.results = {
                'phi1_vals': phi1_vals,
                'theta_vals': theta_vals,
                'iout_vals': iout_vals,
                'trans1_vals': trans1_vals,
                'trans2_vals': trans2_vals,
                'phase1_vals': phase1_vals,
                'phase2_vals': phase2_vals,
                'ng_vals': ng_vals,
                'ng_num': ng_num,
                'use_res1_oc': use_res1_oc,
                'use_res2_oc': use_res2_oc,
                'eta1_val': eta1_val,
                'eta2_val': eta2_val,
                'config': config,
                'a1_val': a1_val,
                'a2_val': a2_val,
                'lambda0_val': lambda0_val,
                'lambda_res1_val': lambda_res1_val,
                'lambda_res2_val': lambda_res2_val
            }
            
            self.status_text.set("Calculation complete")
            return True
            
        except Exception as e:
            self.status_text.set(f"Error: {str(e)}")
            messagebox.showerror("Calculation Error", f"An error occurred during calculation:\n{str(e)}")
            return False
    
    def update_plots(self):
        """Update all plots"""
        if not self.calculate_system():
            return
        
        # Clear previous plots
        self.fig.clear()
        
        # Get results and colors
        r = self.results
        config = r['config']
        colors = self.get_color_scheme()
        
        # Create subplots
        ax1 = self.fig.add_subplot(2, 3, 1)
        ax2 = self.fig.add_subplot(2, 3, 2)
        ax3 = self.fig.add_subplot(2, 3, 3)
        ax4 = self.fig.add_subplot(2, 3, 4)
        ax5 = self.fig.add_subplot(2, 3, 5)
        ax6 = self.fig.add_subplot(2, 3, 6)
        
        # Plot 1: Total Output Intensity
        ax1.plot(r['phi1_vals'], r['iout_vals'], color=colors[0], linewidth=2)
        ax1.set_title(f'Config {config}: Total Output')
        ax1.set_xlabel('φ1')
        ax1.set_ylabel('Output Intensity')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Output Phase
        ax2.plot(r['phi1_vals'], r['theta_vals'], color=colors[1], linewidth=2)
        ax2.set_title(f'Config {config}: Output Phase')
        ax2.set_xlabel('φ1')
        ax2.set_ylabel('Output Phase (θ)')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Individual Phases with coupling state info
        if config == 'A':
            phase1_label = f'Res1 {"OC" if r["use_res1_oc"] else "UC"} (Trans)'
            phase2_label = f'Res2 {"OC" if r["use_res2_oc"] else "UC"} (Trans)'
        elif config == 'B':
            phase1_label = f'Res1 {"OC" if r["use_res1_oc"] else "UC"} (Trans)'
            phase2_label = 'Res2 (Refl)'
        else:  # Config C
            phase1_label = 'Res1 (Refl)'
            phase2_label = 'Res2 (Refl)'
        
        ax3.plot(r['phi1_vals'], r['phase1_vals'], color=colors[2], linewidth=2, label=phase1_label)
        ax3.plot(r['phi1_vals'], r['phase2_vals'], color=colors[3], linewidth=2, label=phase2_label)
        ax3.set_title('Individual Phases')
        ax3.set_xlabel('φ1')
        ax3.set_ylabel('Phase')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Individual Transmissions
        ax4.plot(r['phi1_vals'], r['trans1_vals'], color=colors[4], linewidth=2, label='Resonator 1')
        ax4.plot(r['phi1_vals'], r['trans2_vals'], color=colors[5], linewidth=2, label='Resonator 2')
        ax4.set_title('Individual Transmissions')
        ax4.set_xlabel('φ1')
        ax4.set_ylabel('Transmission')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Group Index (filtered for better visualization)
        # Filter extreme values for better display
        ng_percentile = np.percentile(np.abs(r['ng_vals']), 90)
        ng_mask = np.abs(r['ng_vals']) <= ng_percentile * 2
        ng_finite = np.isfinite(r['ng_vals'])
        ng_display_mask = ng_mask & ng_finite
        
        ng_num_percentile = np.percentile(np.abs(r['ng_num']), 90)
        ng_num_mask = np.abs(r['ng_num']) <= ng_num_percentile * 2
        ng_num_finite = np.isfinite(r['ng_num'])
        ng_num_display_mask = ng_num_mask & ng_num_finite
        
        if np.any(ng_display_mask):
            ax5.plot(r['phi1_vals'][ng_display_mask], r['ng_vals'][ng_display_mask], color=colors[0], 
                    linewidth=2, label='Analytical', alpha=0.8)
        if np.any(ng_num_display_mask):
            ax5.plot(r['phi1_vals'][ng_num_display_mask], r['ng_num'][ng_num_display_mask], color=colors[1], 
                    linewidth=2, linestyle='--', label='Numerical', alpha=0.7)
        ax5.set_title('Group Index (ng = -dθ/dφ)')
        ax5.set_xlabel('φ1')
        ax5.set_ylabel('Group Index')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Group velocity v_g (replaces unwrapped phase)
        # Primary axis: group velocity in m/s (v_g = c / (n_eff * dθ/dφ))
        # Secondary axis: v_g normalized to c
        c = 299792458.0
        phi_vals = r['phi1_vals']

        # Try to get precomputed values; fall back to computing dθ/dφ and v_g numerically
        vg = r.get('v_g_num', None)
        dtheta_dphi = r.get('dtheta_dphi_num', None)

        if dtheta_dphi is None:
            # numerical derivative fallback
            try:
                dphi_step = phi_vals[1] - phi_vals[0]
            except Exception:
                dphi_step = np.gradient(phi_vals).mean()
            dtheta_dphi = np.gradient(r['theta_vals'], dphi_step)

        if vg is None:
            n_eff_val = self.n.get() if hasattr(self, 'n') else r.get('n', 1.0)
            # avoid divide-by-zero warnings during compute
            with np.errstate(divide='ignore', invalid='ignore'):
                vg = c / (n_eff_val * dtheta_dphi)

        # Mask unreasonable values for plotting (infs, NaNs, or >> c)
        finite = np.isfinite(vg)
        reasonable = finite & (np.abs(vg) < 10 * c)  # keep within 10*c for visualization

        if np.any(reasonable):
            ax6.plot(phi_vals[reasonable], vg[reasonable], linewidth=2)
            ax6.set_title('Group Velocity v_g (m/s) — v_g = c / (n_eff · dθ/dφ)')
            ax6.set_xlabel('φ1')
            ax6.set_ylabel('v_g (m/s)')
            ax6.grid(True, alpha=0.3)

            # Secondary axis: normalized to c
            ax6b = ax6.twinx()
            ax6b.plot(phi_vals[reasonable], vg[reasonable] / c, linestyle=':', linewidth=1)
            ax6b.set_ylabel('v_g / c')

            # Optional: lightly mark masked points (singular regions)
            if np.any(~reasonable):
                # mark singularities with a faint vertical band (visual cue)
                idx_singular = np.where(~reasonable)[0]
                # draw small vertical markers at singular phi values
                for ind in idx_singular[::max(1, len(idx_singular)//40)]:  # sample markers if many
                    ax6.axvline(phi_vals[ind], color='gray', alpha=0.12, linewidth=0.8)
        else:
            ax6.text(0.5, 0.5, 'Group velocity undefined\n(too many singularities)', ha='center', va='center')
            ax6.set_title('Group Velocity')
            ax6.set_xlabel('φ1')
            ax6.set_ylabel('v_g (m/s)')
            ax6.grid(True, alpha=0.3)
        
        # Add main title with phase shifter info
        eta1_str = self.eta1.get()
        eta2_str = self.eta2.get()
        config_desc = {'A': 'Both coupled', 'B': 'Upper coupled, lower inserted', 'C': 'Both inserted'}
        
        self.fig.suptitle(f'MZI Analysis - Config {config}: {config_desc[config]}\n' + 
                         f'Phase shifters: η₁={eta1_str}, η₂={eta2_str} | λ₀={r["lambda0_val"]*1e9:.1f}nm', 
                         fontsize=12)
        
        # Update canvas
        self.canvas.draw()
        self.draw_system_diagram()
        self.status_text.set("Plots updated")
    
    def show_zoomed(self):
        """Show zoomed version of plots focusing on interesting regions"""
        if not hasattr(self, 'results'):
            messagebox.showwarning("No Data", "Please calculate data first by clicking 'Update Plots'")
            return
        
        # Create new window
        zoom_window = tk.Toplevel(self.root)
        zoom_window.title("Zoomed Plots - Focused Regions")
        zoom_window.geometry("1200x800")
        
        # Create figure for zoomed plots
        fig_zoom = Figure(figsize=(12, 8), dpi=100)
        fig_zoom.subplots_adjust(hspace=0.4, wspace=0.3, left=0.08, right=0.95, top=0.93, bottom=0.08)
        
        # Create canvas
        canvas_zoom = FigureCanvasTkAgg(fig_zoom, zoom_window)
        canvas_zoom.draw()
        canvas_zoom.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Navigation toolbar
        toolbar_zoom = NavigationToolbar2Tk(canvas_zoom, zoom_window)
        toolbar_zoom.update()
        toolbar_zoom.pack(side=tk.BOTTOM, fill=tk.X)
        self.toolbar_zoom = toolbar_zoom
        
        # Get results and colors
        r = self.results
        config = r['config']
        colors = self.get_color_scheme()
        
        # Smart zoom function - focus on regions with significant variation
        def find_interesting_regions(data, phi_vals, num_regions=3):
            """Find regions with highest variation for zooming"""
            # Calculate local variance using sliding window
            window_size = max(10, len(data) // 20)  # At least 10 points, or 5% window
                
            variances = []
            centers = []
            
            for i in range(window_size, len(data) - window_size):
                local_var = np.var(data[i-window_size:i+window_size])
                variances.append(local_var)
                centers.append(i)
            
            if len(variances) == 0:
                return [(0, len(phi_vals))]
            
            # Find peaks in variance
            variances = np.array(variances)
            centers = np.array(centers)
            
            # Sort by variance and take top regions
            top_indices = np.argsort(variances)[-min(num_regions, len(variances)):]
            
            regions = []
            for idx in top_indices:
                center_idx = centers[idx]
                start_idx = max(0, center_idx - window_size)
                end_idx = min(len(phi_vals), center_idx + window_size)
                regions.append((start_idx, end_idx))
            
            return regions
        
        # Find interesting regions for different plots
        output_regions = find_interesting_regions(r['iout_vals'], r['phi1_vals'])
        phase_regions = find_interesting_regions(r['theta_vals'], r['phi1_vals'])
        
        # Create zoomed subplots
        ax1 = fig_zoom.add_subplot(2, 3, 1)
        ax2 = fig_zoom.add_subplot(2, 3, 2)
        ax3 = fig_zoom.add_subplot(2, 3, 3)
        ax4 = fig_zoom.add_subplot(2, 3, 4)
        ax5 = fig_zoom.add_subplot(2, 3, 5)
        ax6 = fig_zoom.add_subplot(2, 3, 6)
        
        # Plot 1: Output Intensity - zoomed to most interesting region
        if output_regions:
            start, end = output_regions[-1]  # Most interesting region
            ax1.plot(r['phi1_vals'][start:end], r['iout_vals'][start:end], color=colors[0], linewidth=2)
            ax1.set_title(f'Config {config}: Output (Zoomed to Peak Variation)')
            ax1.set_xlabel('φ1')
            ax1.set_ylabel('Output Intensity')
            ax1.grid(True, alpha=0.3)
        else:
            ax1.plot(r['phi1_vals'], r['iout_vals'], color=colors[0], linewidth=2)
            ax1.set_title(f'Config {config}: Output')
            ax1.set_xlabel('φ1')
            ax1.set_ylabel('Output Intensity')
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Output Phase - zoomed to most interesting region
        if phase_regions:
            start, end = phase_regions[-1]
            ax2.plot(r['phi1_vals'][start:end], r['theta_vals'][start:end], color=colors[1], linewidth=2)
            ax2.set_title(f'Config {config}: Phase (Zoomed to Peak Variation)')
            ax2.set_xlabel('φ1')
            ax2.set_ylabel('Output Phase (θ)')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.plot(r['phi1_vals'], r['theta_vals'], color=colors[1], linewidth=2)
            ax2.set_title(f'Config {config}: Phase')
            ax2.set_xlabel('φ1')
            ax2.set_ylabel('Output Phase (θ)')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Individual Phases - zoom to resonance regions
        # Find resonances (minima in transmission)
        res1_minima = np.where(r['trans1_vals'] < 0.1)[0]
        
        if len(res1_minima) > 0:
            center = res1_minima[len(res1_minima)//2]
            window = max(50, len(r['phi1_vals']) // 20)
            start = max(0, center - window)
            end = min(len(r['phi1_vals']), center + window)
            
            if config == 'A':
                phase1_label = f'Res1 {"OC" if r["use_res1_oc"] else "UC"} (Trans)'
                phase2_label = f'Res2 {"OC" if r["use_res2_oc"] else "UC"} (Trans)'
            elif config == 'B':
                phase1_label = f'Res1 {"OC" if r["use_res1_oc"] else "UC"} (Trans)'
                phase2_label = 'Res2 (Refl)'
            else:  # Config C
                phase1_label = 'Res1 (Refl)'
                phase2_label = 'Res2 (Refl)'
            
            ax3.plot(r['phi1_vals'][start:end], r['phase1_vals'][start:end], color=colors[2], linewidth=2, label=phase1_label)
            ax3.plot(r['phi1_vals'][start:end], r['phase2_vals'][start:end], color=colors[3], linewidth=2, label=phase2_label)
            ax3.set_title('Individual Phases (Zoomed to Resonance)')
            ax3.set_xlabel('φ1')
            ax3.set_ylabel('Phase')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            # Fallback to full range if no clear resonances found
            if config == 'A':
                phase1_label = f'Res1 {"OC" if r["use_res1_oc"] else "UC"} (Trans)'
                phase2_label = f'Res2 {"OC" if r["use_res2_oc"] else "UC"} (Trans)'
            elif config == 'B':
                phase1_label = f'Res1 {"OC" if r["use_res1_oc"] else "UC"} (Trans)'
                phase2_label = 'Res2 (Refl)'
            else:  # Config C
                phase1_label = 'Res1 (Refl)'
                phase2_label = 'Res2 (Refl)'
                
            ax3.plot(r['phi1_vals'], r['phase1_vals'], color=colors[2], linewidth=2, label=phase1_label)
            ax3.plot(r['phi1_vals'], r['phase2_vals'], color=colors[3], linewidth=2, label=phase2_label)
            ax3.set_title('Individual Phases')
            ax3.set_xlabel('φ1')
            ax3.set_ylabel('Phase')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Transmissions - zoom to show resonance detail
        if len(res1_minima) > 0:
            center = res1_minima[len(res1_minima)//2]
            window = max(30, len(r['phi1_vals']) // 30)  # Tighter zoom
            start = max(0, center - window)
            end = min(len(r['phi1_vals']), center + window)
            
            ax4.plot(r['phi1_vals'][start:end], r['trans1_vals'][start:end], color=colors[4], linewidth=2, label='Resonator 1')
            ax4.plot(r['phi1_vals'][start:end], r['trans2_vals'][start:end], color=colors[5], linewidth=2, label='Resonator 2')
            ax4.set_title('Transmissions (Zoomed to Resonance)')
            ax4.set_xlabel('φ1')
            ax4.set_ylabel('Transmission')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.plot(r['phi1_vals'], r['trans1_vals'], color=colors[4], linewidth=2, label='Resonator 1')
            ax4.plot(r['phi1_vals'], r['trans2_vals'], color=colors[5], linewidth=2, label='Resonator 2')
            ax4.set_title('Individual Transmissions')
            ax4.set_xlabel('φ1')
            ax4.set_ylabel('Transmission')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # Plot 5: Group Index - focus on physically meaningful range
        ng_reasonable = np.abs(r['ng_vals']) < np.percentile(np.abs(r['ng_vals']), 80)
        ng_finite = np.isfinite(r['ng_vals'])
        ng_mask = ng_reasonable & ng_finite
        
        if np.any(ng_mask):
            ax5.plot(r['phi1_vals'][ng_mask], r['ng_vals'][ng_mask], color=colors[0], 
                    linewidth=2, label='Analytical', alpha=0.8)
            
            # Add numerical if different enough
            ng_num_reasonable = np.abs(r['ng_num']) < np.percentile(np.abs(r['ng_num']), 80)
            ng_num_finite = np.isfinite(r['ng_num'])
            ng_num_mask = ng_num_reasonable & ng_num_finite
            
            if np.any(ng_num_mask):
                ax5.plot(r['phi1_vals'][ng_num_mask], r['ng_num'][ng_num_mask], color=colors[1], 
                        linewidth=1, linestyle=':', label='Numerical', alpha=0.6)
            
            ax5.set_title('Group Index (Filtered Range)')
            ax5.set_xlabel('φ1')
            ax5.set_ylabel('Group Index')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # Plot 6: Phase slope (derivative) - shows group index regions clearly
        # phase_slope now uses numerical dθ/dφ computed earlier
        phase_slope = r.get('dtheta_dphi_num', np.gradient(r['theta_vals'], r['phi1_vals']))
        # filter for reasonable visualization
        slope_reasonable = np.abs(phase_slope) < np.percentile(np.abs(phase_slope[np.isfinite(phase_slope)]), 85)
        slope_finite = np.isfinite(phase_slope)
        slope_mask = slope_reasonable & slope_finite
        if np.any(slope_mask):
            ax6.plot(r['phi1_vals'][slope_mask], phase_slope[slope_mask], color=colors[6], linewidth=2)
            ax6.set_title('Phase slope dθ/dφ (shows dispersion regions)')
            ax6.set_xlabel('φ1')
            ax6.set_ylabel('dθ/dφ')
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'Phase slope undefined', ha='center', va='center')
            ax6.set_title('Phase slope dθ/dφ')
            ax6.set_xlabel('φ1')
            ax6.set_ylabel('dθ/dφ')
            ax6.grid(True, alpha=0.3)
        # Add title
        eta1_str = self.eta1.get()
        eta2_str = self.eta2.get()
        config_desc = {'A': 'Both coupled', 'B': 'Upper coupled, lower inserted', 'C': 'Both inserted'}
        
        fig_zoom.suptitle(f'MZI Analysis - Config {config}: {config_desc[config]} (SMART ZOOM)\n' + 
                         f'Phase shifters: η₁={eta1_str}, η₂={eta2_str} | λ₀={r["lambda0_val"]*1e9:.1f}nm', 
                         fontsize=14)
        
        canvas_zoom.draw()
    
    def export_data(self):
        """Export calculated data to CSV file"""
        if not hasattr(self, 'results'):
            messagebox.showwarning("No Data", "Please calculate data first by clicking 'Update Plots'")
            return
        
        # Get save location
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")],
            title="Save Analysis Results"
        )
        
        if not filename:
            return
        
        try:
            # Prepare data
            r = self.results
            data = {
                'phi1': r['phi1_vals'],
                'total_output': r['iout_vals'],
                'output_phase': r['theta_vals'],
                'unwrapped_phase': np.unwrap(r['theta_vals']),
                'res1_transmission': r['trans1_vals'],
                'res2_transmission': r['trans2_vals'],
                'res1_phase': r['phase1_vals'],
                'res2_phase': r['phase2_vals'],
                'group_index_analytical': r['ng_vals'],
                'group_index_numerical': r['ng_num']
            }
            
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Add metadata as comments
            metadata = {
                'Configuration': r['config'],
                'r1': self.r1_val.get(),
                'r2': self.r2_val.get(),
                'r3': self.r3_val.get(),
                'r4': self.r4_val.get(),
                'Q1': self.Q1_val.get(),
                'Q2': self.Q2_val.get(),
                'alpha1_dB_per_cm': self.alpha1.get(),
                'alpha2_dB_per_cm': self.alpha2.get(),
                'lambda0_nm': self.lambda0.get() * 1e9,
                'lambda_res1_nm': self.lambda_res1.get() * 1e9,
                'lambda_res2_nm': self.lambda_res2.get() * 1e9,
                'R1_um': self.R1.get() * 1e6,
                'R2_um': self.R2.get() * 1e6,
                'n': self.n.get(),
                'eta1': self.eta1.get(),
                'eta2': self.eta2.get(),
                'phi1_min': self.phi1min.get(),
                'phi1_max': self.phi1max.get(),
                'steps': self.phi1_steps.get(),
                'res1_phase_type': 'OC' if r['use_res1_oc'] else 'UC',
                'res2_phase_type': 'OC' if r['use_res2_oc'] else 'UC',
                'a1_calculated': r['a1_val'],
                'a2_calculated': r['a2_val']
            }
            
            # Save file
            if filename.endswith('.xlsx'):
                with pd.ExcelWriter(filename) as writer:
                    df.to_excel(writer, sheet_name='Data', index=False)
                    pd.DataFrame(list(metadata.items()), columns=['Parameter', 'Value']).to_excel(
                        writer, sheet_name='Parameters', index=False)
            else:
                # Save as CSV with metadata header
                with open(filename, 'w', newline='') as f:
                    f.write("# MZI Ring Resonator Analysis Results\n")
                    for key, value in metadata.items():
                        f.write(f"# {key}: {value}\n")
                    f.write("#\n")
                    df.to_csv(f, index=False)
            
            messagebox.showinfo("Export Successful", f"Data exported successfully to:\n{filename}")
            self.status_text.set(f"Data exported to {filename}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Error exporting data:\n{str(e)}")
    
    def reset_parameters(self):
        """Reset all parameters to default values"""
        result = messagebox.askyesno("Reset Parameters", 
                                   "Are you sure you want to reset all parameters to default values?")
        if result:
            # Reset to default values
            self.lambda0.set(1548.36e-9)
            self.lambda_res1.set(1550e-9)
            self.lambda_res2.set(1550e-9)
            self.R1.set(15e-6)
            self.R2.set(15e-6)
            self.n.set(2.9)
            self.alpha1.set(0.1)
            self.alpha2.set(0.1)
            self.r1_val.set(0.998)
            self.r2_val.set(0.9)
            self.r3_val.set(1.0)
            self.r4_val.set(1.0)
            self.Q1_val.set(5e5)
            self.Q2_val.set(6e5)
            self.eta1.set("0")
            self.eta2.set("0")
            self.phi1min.set(-0.5)
            self.phi1max.set(0.5)
            self.phi1_steps.set(12000)
            self.config.set('A')
            self.force_res1_phase.set('Auto')
            self.force_res2_phase.set('Auto')
            self.color_scheme.set('Default')
            
            # Update plots
            self.update_plots()
    
    def show_help_window(self):
        """Show help information about the MZI analysis"""
        help_window = tk.Toplevel(self.root)
        help_window.title("MZI Analysis Help")
        help_window.geometry("600x500")
        
        help_text = """
MZI Ring Resonator Analysis Tool - Help

CONFIGURATIONS:
• Config A (Both coupled): Both resonators are side-coupled to the waveguide
• Config B (Upper coupled, lower inserted): Res1 side-coupled, Res2 inserted in path
• Config C (Both inserted): Both resonators inserted in the optical path

PARAMETERS:
• λ₀: Operating wavelength
• n: Effective refractive index
• R1, R2: Ring radii
• r1, r2: Coupling coefficients (main couplers)
• r3, r4: Additional coupling coefficients (essential for reflection configs)
• Q1, Q2: Quality factors
• α1, α2: Material attenuation (dB/cm)
• λres1, λres2: Resonant wavelengths
• η1, η2: Phase shifters

COUPLING STATES:
• OC (Overcoupled): r < a (coupling dominates loss)
• UC (Undercoupled): r > a (loss dominates coupling)
• For reflection: OC/UC doesn't apply, uses geometric phase

PLOTS EXPLAINED:
• Total Output: Interferometric output intensity
• Output Phase: Overall MZI phase response
• Individual Phases: Phase of each resonator (Trans/Refl mode shown)
• Individual Transmissions: Single resonator responses
• Group Index: n_g ≈ n_eff · dθ/dφ (device group index); negative values indicate anomalous dispersion
• Unwrapped Phase: Continuous phase (removes 2π jumps)

ZOOM FEATURE:
Smart zoom focuses on regions with highest variation, particularly near resonances.

TIPS:
• For Config B & C: Set r3, r4 < 1 for meaningful reflection
• Use different η1, η2 values to optimize interference
• Monitor coupling states in status bar
        """
        
        text_widget = tk.Text(help_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert(tk.END, help_text)
        text_widget.config(state=tk.DISABLED)
        
        # Add scrollbar
        scrollbar_help = ttk.Scrollbar(help_window, orient="vertical", command=text_widget.yview)
        scrollbar_help.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.config(yscrollcommand=scrollbar_help.set)
    
    def validate_parameters(self):
        """Validate parameter ranges and relationships"""
        issues = []
        
        # Check coupling parameters
        if self.config.get() in ['B', 'C']:
            if self.r3_val.get() >= 1.0:
                issues.append("Warning: r3 = 1 means no coupling for reflection (Config B/C)")
            if self.r4_val.get() >= 1.0:
                issues.append("Warning: r4 = 1 means no coupling for reflection (Config B/C)")
        
        # Check physical consistency
        if self.R1.get() <= 0 or self.R2.get() <= 0:
            issues.append("Error: Ring radii must be positive")
        
        if self.Q1_val.get() <= 0 or self.Q2_val.get() <= 0:
            issues.append("Error: Quality factors must be positive")
        
        if self.alpha1.get() < 0 or self.alpha2.get() < 0:
            issues.append("Error: Attenuation cannot be negative")
        
        # Check sweep parameters
        if self.phi1min.get() >= self.phi1max.get():
            issues.append("Error: φ1 min must be less than φ1 max")
        
        if self.phi1_steps.get() < 10:
            issues.append("Warning: Very few steps may give poor resolution")
        
        return issues
    
    def show_parameter_info(self):
        """Show current parameter information and validation"""
        issues = self.validate_parameters()
        
        info_window = tk.Toplevel(self.root)
        info_window.title("Parameter Information")
        info_window.geometry("500x400")
        
        text_widget = tk.Text(info_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
        
        # Current parameters
        info_text = "CURRENT PARAMETERS:\n"
        info_text += f"Configuration: {self.config.get()}\n"
        info_text += f"λ₀: {self.lambda0.get()*1e9:.2f} nm\n"
        info_text += f"n: {self.n.get():.3f}\n"
        info_text += f"R1: {self.R1.get()*1e6:.1f} μm, R2: {self.R2.get()*1e6:.1f} μm\n"
        info_text += f"r1: {self.r1_val.get():.4f}, r2: {self.r2_val.get():.4f}\n"
        info_text += f"r3: {self.r3_val.get():.4f}, r4: {self.r4_val.get():.4f}\n"
        info_text += f"Q1: {self.Q1_val.get():.1e}, Q2: {self.Q2_val.get():.1e}\n"
        info_text += f"α1: {self.alpha1.get():.3f} dB/cm, α2: {self.alpha2.get():.3f} dB/cm\n"
        info_text += f"η1: {self.eta1.get()}, η2: {self.eta2.get()}\n\n"
        
        if hasattr(self, 'results'):
            r = self.results
            info_text += "CALCULATED VALUES:\n"
            info_text += f"a1 (loss): {r['a1_val']:.4f}\n"
            info_text += f"a2 (loss): {r['a2_val']:.4f}\n"
            info_text += f"r1-a1: {self.r1_val.get()-r['a1_val']:.4f} ({'OC' if self.r1_val.get()<r['a1_val'] else 'UC'})\n"
            info_text += f"r2-a2: {self.r2_val.get()-r['a2_val']:.4f} ({'OC' if self.r2_val.get()<r['a2_val'] else 'UC'})\n\n"
        
        if issues:
            info_text += "VALIDATION ISSUES:\n"
            for issue in issues:
                info_text += f"• {issue}\n"
        else:
            info_text += "✓ All parameters are valid\n"
        
        text_widget.insert(tk.END, info_text)
        text_widget.config(state=tk.DISABLED)
    
    def add_menu_bar(self):
        """Add menu bar to the main window"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Export Data...", command=self.export_data)
        file_menu.add_separator()
        file_menu.add_command(label="Reset Parameters", command=self.reset_parameters)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Show Zoomed Plots", command=self.show_zoomed)
        view_menu.add_command(label="Parameter Info", command=self.show_parameter_info)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="User Guide", command=self.show_help_window)

def main():
    """Main function to run the GUI application"""
    root = tk.Tk()
    app = MZIAnalysisGUI(root)
    app.add_menu_bar()  # Add menu bar
    root.mainloop()

if __name__ == "__main__":
    main()