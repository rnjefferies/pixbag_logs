# pixbag_logs

pixbag_logs is a collection of Python scripts designed to process and analyse ROS bag files generated during remote operation experiments. I am using the TUM TOD Software stack, adapted to work with the PixKit automated vehicle (see https://github.com/rnjefferies/remote_drivingCAN). 

The project provides tools for extracting data, detecting steering reversals, and generating visualisations. This is a work in progress, so expect scripts for analysing the rest of the ROS bag soon. Any data found in this repository is phoney. 

## Features

- **Data Extraction**: Extracts operator and vehicle data from ROS bag files.
- **Reversal Detection**: Detects reversals based on filtered driving data.
- **Visualisation**: Generates plots with reversal markers for detailed analysis.
- **CSV Output**: Summarises reversal data for each participant and condition.

## Installation

**Clone the Repository**

To get started, clone the repository:

        git clone https://github.com/yourusername/Pix_Logs.git
        cd pixbag_logs

**Install Dependencies**

Install the required Python packages using pip:

        pip install -r requirements.txt

## Usage 

- Ensure the ROS bag logging launch file is correctly configured for your experimental setup (i.e., participants ID and condition), and the output path is set to "/pixbag_logs/rosbag_files". 

- Enter the /scripts subdirectory and run: 

        python3 read_rosbag_to_multiple_csv.py

- You can then run any of the scripts:

        python3 steering_reversal_rate.py 
        python3 throttle_reversal_rate.py 
        python3 vehicle_speed_reversal_rate.py
        python3 vehicle_wheel_reversal_rate.py

- For plotting the reversals adjust the sample rate, cut-off frequency, and threshold (i.e., gap size) accordingly. Read the Markkula and Engström (2006) paper cited below for more details. 

## Acknowledgements

The method for detecting and visualising reversals was adapted from:

- Markkula, G orcid.org/0000-0003-0244-1582 and Engström, J (2006) A Steering Wheel Reversal Rate Metric for Assessing Effects of Visual and Cognitive Secondary Task Load. In: Proceedings of the 13th ITS World Congress. 13th ITS World Congress, 08-12 Oct
2006, London, UK.

- Zhao, Lin. (2023). Teleoperation and the influence of driving feedback on drivers' behaviour and experience. 10.13140/RG.2.2.21612.85128. 

This project was made possible thanks to the contributions of the open-source community. I rely on several fantastic libraries, including:

- [pandas](https://pandas.pydata.org/) - For data manipulation.
- [numpy](https://numpy.org/) - For numerical computing.
- [matplotlib](https://matplotlib.org/) - For creating visualizations.
- [scipy](https://scipy.org/) - For signal processing.

Thanks!
