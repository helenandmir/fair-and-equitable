# README

## Overview

This repository contains code for experiments conducted in the accompanying paper. The code allows you to generate data, run various algorithms, and visualize the results.

## Table of Contents

1. [Data Creation](#data-creation)
   - [Data Sources](#data-sources)
   - [Generating Synthetic Data](#generating-synthetic-data)
   - [Preparing for Fair and FairEq Algorithms](#preparing-for-fair-and-faireq-algorithms)
2. [Running Algorithms](#running-algorithms)
   - [Fair and FairEq Algorithms](#fair-and-faireq-algorithms)
   - [Hungarian and MaxMinOptimalReps Algorithms](#hungarian-and-maxminoptimalreps-algorithms)
3. [Visualization](#visualization)
4. [Notes](#notes)

## Data Creation

### Data Sources

The datasets for Temperature (Temp), Businesses, and Points of Interest (POI) can be found at the following links:

- **Global Temperature Records (1850-2022):**  
  [https://www.kaggle.com/datasets/maso0dahmed/global-temperature-records-1850-2022](https://www.kaggle.com/datasets/maso0dahmed/global-temperature-records-1850-2022)

- **Issued Licenses (Businesses in New York City):**  
  [https://data.cityofnewyork.us/Business/Issued-Licenses/w7w3-xahh/data](https://data.cityofnewyork.us/Business/Issued-Licenses/w7w3-xahh/data)

- **Points of Interest (POI) Database:**  
  [https://www.kaggle.com/datasets/ehallmar/points-of-interest-poi-database](https://www.kaggle.com/datasets/ehallmar/points-of-interest-poi-database)

### Generating Synthetic Data

- **v.random Approach:**  
  To create synthetic data where the sensitive attribute values are distributed using the **v.random** approach, run the `random_data_generation` script.

- **v.contagion Approach:**  
  To create synthetic data where the sensitive attribute values are distributed using the **v.contagion** approach, run the `contagion_data_generation` script.

**Note:** In both scripts, uncomment the experiment you wish to run.

- **Saving the Data:**  
  - For data generated using the **v.random** approach, save the files in the folders: `trail01`, `trail02`, `trail03`.
  - For data generated using the **v.contagion** approach, save the files in the folders: `trail1`, `trail2`, `trail3`.

### Preparing for Fair and FairEq Algorithms

Before running the **Fair** and **FairEq** algorithms, create a CSV file that stores the neighborhood radius for each point by running the code in the `create_radius_table` script.

## Running Algorithms

### Fair and FairEq Algorithms

To run the **Fair** or **FairEq** algorithms:

1. Run the code in the `main` script.
2. Inside the `main` function:
   - Uncomment the experiment you wish to run.
   - Update the fields `fileA`, `fileB`, and `attributs_list` according to the files you are using.
     - **Note:** If you are running experiments on the synthetic data, there is no need to change the `attributs_list` field.

### Hungarian and MaxMinOptimalReps Algorithms

After obtaining a list of reps, you can run:

- **Hungarian Algorithm:**  
  Run the `Hungarian` script.

- **MaxMinOptimalReps Algorithm:**  
  Run the `Hungarian2` script.

## Visualization

To visualize the distribution of points and centers (reps) for the different algorithms, use the `plot_dots` script.

## Notes

- Ensure all necessary dependencies and libraries are installed before running the scripts.
- For any questions or issues, please contact me at [helen.sternbach@mail.huji.ac.il](mailto:helen.sternbach@mail.huji.ac.il).