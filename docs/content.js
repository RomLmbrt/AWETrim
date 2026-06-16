const CONTENT = {
  "awetrim": {
    title: "AWETrim",
    text: "AWETrim is the central quasi-steady multi-fidelity framework. It connects flight-data reconstruction, aerostructural simulation, reduced-order modelling, and trajectory optimisation into one workflow for soft-kite AWES analysis.",
    bullets: ["CasADi-based system model", "Interfaces to VSM, PSS and EKF-AWE", "Fast enough for simulation and optimisation"],
    image: "img/placeholder.svg",
    caption: "Add an overview image, workflow animation, or rendered kite system here.",
    links: [{label: "Repository README", url: "https://github.com/awegroup/AWETrim"}]
  },
  "experimental-flight-data": {
    title: "Experimental Flight Data",
    text: "Raw measured data from flight tests provides the starting point for reconstruction. Typical signals include position, velocity, tether force, reel-out speed and onboard sensor measurements.",
    bullets: ["Raw CSV logs", "Pre-processed HDF5 files", "Sensor measurements and operational data"],
    image: "img/placeholder.svg",
    caption: "Add a photo of the experimental setup or a raw-data plot."
  },
  "ekf-awe": {
    title: "EKF-AWE Experimental Reconstruction",
    text: "EKF-AWE processes flight logs using an Extended Kalman Filter to estimate kite states, aerodynamic coefficients and wind velocity. In AWETrim, this forms the bridge between real flight data and model validation or tuning.",
    bullets: ["State reconstruction", "Wind estimation", "Aerodynamic coefficient identification"],
    image: "img/placeholder.svg",
    caption: "Add a reconstructed trajectory, wind-speed estimate, or EKF validation plot.",
    links: [{label: "EKF-AWE repository", url: "https://github.com/ocayon/EKF-AWE"}]
  },
  "wind-state-estimation": {
    title: "Wind and State Estimation",
    text: "The reconstructed states and wind estimates are used to understand real flight conditions and to compare experimental behaviour with the model predictions.",
    bullets: ["Estimated position and velocity", "Wind vector reconstruction", "Inputs for validation and tuning"],
    image: "img/placeholder.svg",
    caption: "Add a time series of estimated wind or kite states."
  },
  "system-kite": {
    title: "System / Kite Characteristics",
    text: "The system definition contains the geometry and hardware properties of the kite, tether, KCU and winch. For the examples, AWETrim uses the TU Delft LEI-V3 kite configuration.",
    bullets: ["Mass and geometry", "Aero and structural configuration files", "Tether, KCU and winch parameters"],
    image: "img/placeholder.svg",
    caption: "Add a kite geometry rendering or parameter table."
  },
  "environmental-conditions": {
    title: "Environmental Conditions",
    text: "Environmental inputs define the wind field and atmosphere used by the simulations. AWETrim includes uniform, logarithmic and tabulated wind models.",
    bullets: ["Uniform wind", "Logarithmic shear", "Tabulated or reconstructed inflow"],
    image: "img/placeholder.svg",
    caption: "Add a wind profile or turbulence plot."
  },
  "operational-constraints": {
    title: "Operational Constraints",
    text: "Operational limits define the feasible flight envelope during simulation and optimisation. They may include tether-force limits, reel-speed bounds, steering limits, depower settings and path constraints.",
    bullets: ["Force and speed limits", "Control bounds", "Flight-envelope constraints"],
    image: "img/placeholder.svg",
    caption: "Add a constraint envelope or optimisation-bound figure."
  },
  "shared-kinematics": {
    title: "Shared Kinematics",
    text: "The kinematic layer provides common reference-frame definitions and course-frame descriptions so that the different fidelity levels use consistent motion variables.",
    bullets: ["Course-frame kinematics", "Reference-frame transforms", "Shared state definitions"],
    image: "img/placeholder.svg",
    caption: "Add a coordinate-frame diagram."
  },
  "aero-structural": {
    title: "Aero-Structural Kite Model",
    text: "The high-fidelity model couples VSM aerodynamics with PSS structural deformation. It iterates aerodynamic loads and deformed geometry to obtain loaded wing shapes and force coefficients across flight conditions.",
    bullets: ["VSM aerodynamic loads", "PSS flexible structure", "Aitken-relaxed fixed-point coupling"],
    image: "img/placeholder.svg",
    caption: "Add a loaded wing shape or pressure/load distribution plot.",
    links: [{label: "Vortex Step Method", url: "https://github.com/awegroup/Vortex-Step-Method"}, {label: "Particle System Simulator", url: "https://github.com/awegroup/Particle_System_Simulator"}]
  },
  "rom": {
    title: "Reduced-Order Kite Model",
    text: "The reduced-order model is fitted from aerostructural sweep results. It gives quasi-steady aerodynamic coefficients as functions of flight variables and control inputs, enabling efficient trajectory simulation and optimisation.",
    bullets: ["Coefficient fitting", "CasADi SystemModel", "Fast trajectory simulation"],
    image: "img/placeholder.svg",
    caption: "Add fitted CL/CD/CS surfaces or ROM validation plots."
  },
  "tether-models": {
    title: "Tether Models",
    text: "Tether models represent the force contribution of the tether and its interaction with the kite dynamics. The current structure supports tether variants inside the system model.",
    bullets: ["Tether force", "Drag contribution", "Coupling to kite dynamics"],
    image: "img/placeholder.svg",
    caption: "Add a tether-force or tether-drag illustration."
  },
  "winch-models": {
    title: "Winch Models",
    text: "Winch models describe reel-in and reel-out operation, coupling the flight trajectory to ground-station power production and operational limits.",
    bullets: ["Reel speed", "Tether length evolution", "Power-cycle simulation"],
    image: "img/placeholder.svg",
    caption: "Add a reel-out speed or power-cycle plot."
  },
  "wind-models": {
    title: "Wind Models",
    text: "Wind models provide the inflow used for simulation, validation and optimisation. AWETrim can use idealised profiles or wind estimates reconstructed from experimental data.",
    bullets: ["Uniform profiles", "Logarithmic profiles", "Tabulated wind fields"],
    image: "img/placeholder.svg",
    caption: "Add a wind field or vertical wind-profile image."
  },
  "trajectory-parametrization": {
    title: "Trajectory Parametrization",
    text: "Trajectory parametrisation defines path patterns such as B-spline curves, uploops, downloops or helices. These parameters become optimisation variables for power-cycle analysis.",
    bullets: ["B-spline path patterns", "Helix and loop trajectories", "Control and path parameters"],
    image: "img/placeholder.svg",
    caption: "Add a 3D trajectory or path-parameter figure."
  },
  "operational-optimization": {
    title: "Operational Optimization",
    text: "Operational optimisation uses the reduced-order model and CasADi Opti to search for path and control parameters that improve performance while satisfying constraints.",
    bullets: ["CasADi Opti", "Path-parameter optimisation", "Power-cycle constraints"],
    image: "img/placeholder.svg",
    caption: "Add an optimisation convergence or optimal-trajectory plot."
  },
  "performance-assessment": {
    title: "Performance Assessment",
    text: "Performance assessment evaluates quantities such as power production, aerodynamic efficiency, tether loads and the sensitivity of results to model choices or operating conditions.",
    bullets: ["Cycle power", "Loads and efficiency", "Sensitivity analysis"],
    image: "img/placeholder.svg",
    caption: "Add power, load, or efficiency plots."
  },
  "design-analysis": {
    title: "Design and Model Analysis",
    text: "Design and model analysis uses the framework to compare configurations, investigate sensitivities and evaluate how modelling assumptions affect kite performance and stability.",
    bullets: ["Configuration comparison", "Model validation", "Stability and sensitivity studies"],
    image: "img/placeholder.svg",
    caption: "Add a design comparison or stability plot."
  }
};
