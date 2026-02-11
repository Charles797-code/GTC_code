# GTC_code
GTC:Learning Global Temporal Context for TimeSeries Forecasting with MissingData

📂 Project Structure
The project is organized as follows:
GTC_Project/
├── data_provider/          # Data loading and mask generation
├── dataset/                # Store your .csv datasets here (e.g., ETTh1.csv)
├── exp/                    # Experiment logic
├── layers/                 # Neural network layers (RevON, etc.)
├── models/                 # Model definitions
│   └── GTC.py              # The GTC model core code
├── utils/                  # Utility functions (metrics, tools, timefeatures)
├── main.py                 # Entry point: Configuration and auto-execution script
├── run.py                  # Argument parsing and experiment initiator
└── requirements.txt        # Python dependencies

🚀 Quick Start
1.Ensure you have Python 3.1+ installed. Install the required dependencies:numpy,matplotlib,pandas,scikit-learn,torch
(You can run pip install -r requirements.txt)

2.Data Preparation
Place your time series data files (CSV format) in the ./dataset/ directory（You need to create by yourself).
The default supported datasets are:ETTh1,ETTh2, ETTm1,ETTm2,Weather,Exchange.

3.Configuration
Open main.py to modify the training configuration. You do not need to use command-line arguments; everything is controlled here.
And The hyperparameter configuration in the current main.py is set for the ETTh1 dataset. For the configurations of other experimental results, please refer to the appendix in the paper.

4.Run the Model
To start the training and evaluation process, simply run: python main.py

5.Environment:
This project has been tested in the Windows 10 (RTX 4060 Laptop) environment using Python 3.13 and CUDA 11.8.
