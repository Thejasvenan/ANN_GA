# ME2281 ANN Assignment: GA-Based Training

## Description
This project implements an Artificial Neural Network (ANN) trained using a Genetic Algorithm (GA) to recognize typed letters T, C, and H from a 3x3 binary pixel input grid.

The network was trained for 1000 generations, using GA parameters such as crossover rate, mutation rate, and elitism.

## Project Structure
- `src/` : C++ source code for ANN + GA implementation
- `excel_validation/` : Manual feedforward validation using Excel
- `logs/` : Fitness and training logs
- `report/` : Assignment report and documentation

## Observations

* ANN successfully recognized T, C, H from the input grid.
* Letter "None" was consistently misclassified as "T", likely due to overfitting on letter patterns.
* GA improved ANN weights and thresholds gradually over generations.
