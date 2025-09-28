# ME2281 ANN Assignment: GA-Based Training

## Description
This project implements an Artificial Neural Network (ANN) trained using a Genetic Algorithm (GA) to recognize typed letters T, C, and H from a 3x3 binary pixel input grid.  
**Importantly, the entire ANN and GA were implemented purely in C++ without using any external libraries or toolboxes.** All computations, including feedforward activation, weight updates, crossover, and mutation, were manually coded.

The network was trained for 1000 generations, using GA parameters such as crossover rate, mutation rate, and elitism.

## Project Structure
- `src/` : C++ source code for ANN + GA implementation (no libraries used)  
- `excel_validation/` : Manual feedforward validation using Excel  
- `logs/` : Fitness and training logs  
- `report/` : Assignment report and documentation  

## Observations
- ANN successfully recognized T, C, and H from the input grid.  
- Letter "None" was consistently misclassified as "T", likely due to overfitting on letter patterns.  
- The Genetic Algorithm gradually refined weights and thresholds, improving overall ANN performance.  
- Manual validation in Excel helped ensure the correctness of feedforward computations before training.  

