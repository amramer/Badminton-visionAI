
## Code Descriptions:

### **app.py**
- **Purpose**: The main entry point for the application. This script is responsible for running the entire video analysis pipeline, utilizing models and tools to process and analyze badminton match videos.

### **config/**
- **config.py**: Contains Python configuration settings, including paths to models, video files, and other parameters required for the system.
- **config.yaml**: A YAML file storing hyperparameters (e.g., batch size, learning rate) for training models, paths for input/output data, etc. It helps in managing settings for different environments.
- **__init__.py**: Initializes the `config` directory as a Python package to facilitate modular code.

### **data/**
- **court_keypoints.json**: Contains data related to court keypoints (coordinates) used in the court detection model.
- **images/**: Folder for storing image files used for testing, preprocessing, or visualization.
- **raw_videos/**: Contains raw video files that serve as the input for video analysis.
- **processed/**: Contains processed video data, such as tracking outputs, annotations, and other results from the analysis.

### **docs/**
- **installation.md**: A guide to setting up the project locally, including environment setup, dependencies, and prerequisites.
- **model_docs.md**: Detailed documentation for the models used in the project, including their architectures, training processes, and usage instructions.

### **environment.yml**
- Configuration file for setting up a Conda environment. It includes dependencies and versions of the packages used in the project.

### **LICENSE.txt**
- Contains licensing information, detailing the terms under which the project can be used or redistributed.

### **models/**
- **court_detection/**: Folder for scripts or pre-trained models related to detecting the court in video frames.
- **players_tracking/**: Contains models and code related to detecting and tracking players in the videos.
- **shuttle_tracking/**: Includes models and scripts for tracking the shuttlecock during matches.

### **notebooks/**
- **data_exploration.ipynb**: Jupyter notebook for exploring the dataset, including data preprocessing and initial analyses.
- **training_models.ipynb**: Jupyter notebook for training the models, including experimentation with different algorithms, hyperparameters, etc.

### **outputs/**
- **logs/**: Stores log files that capture training progress, errors, and debugging output.
- **tracking_results/**: Folder to store output video files showing tracking results (e.g., player, shuttle, and court annotations).
- **visuals/**: Contains visualizations of model results, like plots, graphs, or GIFs showing tracked elements in the video.

### **README.md**
- The project overview file that provides a description of the project, setup instructions, usage details, and other important information about the project.

### **requirements.txt**
- A list of Python packages required to run the project. It is used for setting up the environment via pip.

### **tools/**
- **analysis/**: Contains scripts for analyzing and visualizing results, including performance metrics (e.g., speed of players, distances covered) and annotated frame visualization.
  - **metrics.py**: Contains code for calculating key metrics, such as player speed, distances covered, etc.
  - **visualizer.py**: Code for visualizing the results, such as generating plots or annotating frames with detected objects (players, shuttlecock).
- **court_detection/**: Contains the court detection model and related code.
  - **court_detection.py**: Implementation of the court detection algorithm, which identifies and marks the court in video frames.
- **players_tracking/**: Includes code for player detection and tracking.
  - **player_detection.py**: Implements player detection and tracking logic, including assigning IDs to players and following their movements.
- **shuttle_tracking/**: Contains scripts for tracking the shuttlecock during the match.
  - **shuttlecock_tracking.py**: Algorithm for detecting and tracking the shuttlecock across video frames.
- **__init__.py**: Initializes the `tools` folder as a Python package.

### **utils/**
- **bbox_utils.py**: Contains utility functions for working with bounding boxes, such as intersection and area calculations.
- **calculation.py**: Includes functions for calculating performance metrics such as player speed, distances, or angle of movement.
- **logger.py**: A logging utility to capture training progress, errors, and important events.
- **player_utils.py**: Utilities specific to player tracking, like ID management and player feature extraction.
- **pose_utils.py**: Functions for handling pose estimation tasks, such as keypoint detection for player poses.
- **video_utils.py**: Utilities for working with videos, such as frame extraction and video manipulation.
- **__init__.py**: Initializes the `utils` folder as a Python package.

