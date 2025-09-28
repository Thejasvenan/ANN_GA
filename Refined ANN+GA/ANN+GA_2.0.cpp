#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <bitset>
#include <conio.h>
#include <ctime>
#include <algorithm>
#include <fstream> 
using namespace std;

// Constants for ANN structure
const int NUM_INPUTS = 9;
const int NUM_HIDDEN_NODES = 8;
const int NUM_OUTPUT_NODES = 2;
const int NUM_WEIGHTS = (NUM_INPUTS * NUM_HIDDEN_NODES) + (NUM_HIDDEN_NODES * NUM_OUTPUT_NODES);
const int NUM_THRESHOLDS = NUM_HIDDEN_NODES + NUM_OUTPUT_NODES;
const int CHROMOSOME_LENGTH = NUM_WEIGHTS + NUM_THRESHOLDS;
const int POPULATION_SIZE = 30;
const double WEIGHT_MIN = -1.0;
const double WEIGHT_MAX = 1.0;
const double THRESHOLD_MIN = 0.0;
const double THRESHOLD_MAX = 1.0;
const int BITS_PER_GENE = 12;
const int GENE_SIZE = BITS_PER_GENE * CHROMOSOME_LENGTH;
const double CROSSOVER_RATE = 0.8;
const double MUTATION_RATE = 0.01;
const int NUM_ELITES = 4;

// Structure for an individual (chromosome)
struct Individual {
    string chromosome;  // Single binary string representing the individual
    double fitness;

    // Default constructor to initialize fitness
    Individual() : fitness(0.0) {}
};


// Function to decode a gene from binary string to double
double decodeGene(const string& gene, double min, double max) {
    int gx = bitset<BITS_PER_GENE>(gene).to_ulong();
    double px = min + ((max - min) * gx / ((1 << BITS_PER_GENE) - 1));  // (2^12 - 1)
    return px;
}

vector<double> decodeChromosome(const string& chromosome) {
    vector<double> parameters;

    // Decode weights
    for (int i = 0; i < NUM_WEIGHTS; i++) {
        string gene = chromosome.substr(i * BITS_PER_GENE, BITS_PER_GENE);
        parameters.push_back(decodeGene(gene, WEIGHT_MIN, WEIGHT_MAX));
    }

    // Decode thresholds
    for (int i = 0; i < NUM_THRESHOLDS; i++) {
        string gene = chromosome.substr((NUM_WEIGHTS + i) * BITS_PER_GENE, BITS_PER_GENE);
        parameters.push_back(decodeGene(gene, THRESHOLD_MIN, THRESHOLD_MAX));
    }

    return parameters;
}

// Step Activation Function
double stepFunction(double value, double threshold) {
    return (value >= threshold) ? 1.0 : 0.0;
}

// ANN Feedforward Function
vector<double> feedForward(const vector<double>& input, const Individual& ind) {
    vector<double> decodedParameters = decodeChromosome(ind.chromosome);
    vector<double> hiddenOutputs(NUM_HIDDEN_NODES);
    vector<double> finalOutputs(NUM_OUTPUT_NODES);

    int wIndex = 0;

    // INPUT -> HIDDEN
    for (int i = 0; i < NUM_HIDDEN_NODES; ++i) {
        double sum = 0.0;
        for (int j = 0; j < NUM_INPUTS; ++j) {
            sum += input[j] * decodedParameters[wIndex++];
        }
        hiddenOutputs[i] = stepFunction(sum, decodedParameters[NUM_WEIGHTS + i]); // Activation function
    }

    // HIDDEN -> OUTPUT
    for (int i = 0; i < NUM_OUTPUT_NODES; ++i) {
        double sum = 0.0;
        for (int j = 0; j < NUM_HIDDEN_NODES; ++j) {
            sum += hiddenOutputs[j] * decodedParameters[wIndex++];
        }
        finalOutputs[i] = stepFunction(sum, decodedParameters[NUM_WEIGHTS + NUM_HIDDEN_NODES + i]); // Activation function
    }

    return finalOutputs;
}

vector<Individual> generateInitialPopulation() {
    vector<Individual> population;

    for (int i = 0; i < POPULATION_SIZE; i++) {
        Individual ind;
        ind.chromosome = "";

        // Generate a random 1176-bit chromosome
        for (int j = 0; j < GENE_SIZE; j++) {
            ind.chromosome += (rand() % 2) ? '1' : '0';
        }

        ind.fitness = 0.0;
        population.push_back(ind);
    }

    return population;
}

// Function to calculate fitness value based on total error
double calculateFitness(double totalError) {
    return 1.0 / (1.0 + totalError);  // Lower error gives higher fitness value
}

// Function to calculate total error
void evaluateFitness(vector<Individual>& population, const vector<vector<double>>& trainingInputs, const vector<vector<double>>& expectedOutputs) {
    for (Individual& ind : population) {
        double totalError = 0.0;

        for (size_t i = 0; i < trainingInputs.size(); ++i) {
            vector<double> output = feedForward(trainingInputs[i], ind);

            for (size_t j = 0; j < output.size(); ++j) {
                double error = expectedOutputs[i][j] - output[j];
                totalError += pow(error, 2); // Squared error
            }
        }

        ind.fitness = calculateFitness(totalError);
    }
    // Sort population by fitness in descending order
    sort(population.begin(), population.end(), [](const Individual& a, const Individual& b) {
        return a.fitness > b.fitness;
        });
}

// Tournament selection to select two parents 
pair<Individual, Individual> tournamentSelection(const vector<Individual>& population) {
    vector<Individual> tournament;

    // Randomly pick 5
    for (int i = 0; i < 5; i++) {
        int idx = rand() % population.size();
        tournament.push_back(population[idx]);
    }

    // Sort the 5 based on fitness
    sort(tournament.begin(), tournament.end(), [](const Individual& a, const Individual& b) {
        return a.fitness > b.fitness;
        });

    // Return best 2
    return { tournament[0], tournament[1] };
}

// Crossover function
pair<Individual, Individual> crossover(const Individual& parent1, const Individual& parent2) {
    Individual child1 = parent1;
    Individual child2 = parent2;

    double R = ((double)rand() / RAND_MAX);
    if (R <= CROSSOVER_RATE) {
        // Pick a random crossover point between 1 and GENE_SIZE-1
        int crossoverPoint = rand() % (GENE_SIZE - 1) + 1;

        // Perform single-point crossover
        child1.chromosome = parent1.chromosome.substr(0, crossoverPoint) + parent2.chromosome.substr(crossoverPoint);
        child2.chromosome = parent2.chromosome.substr(0, crossoverPoint) + parent1.chromosome.substr(crossoverPoint);
    }

    return { child1, child2 };
}

// Mutation function
void mutate(Individual& ind) {
    for (int i = 0; i < GENE_SIZE; ++i) {
        if (((double)rand() / RAND_MAX) <= MUTATION_RATE) {
            // Flip the bit
            ind.chromosome[i] = (ind.chromosome[i] == '0') ? '1' : '0';
        }
    }
}

// Create next generation using elitism, crossover, and mutation
vector<Individual> createNextGeneration(const vector<Individual>& currentPopulation) {
    vector<Individual> newPopulation;

    // Elitism: carry over the best 4 individual unchanged
    for (int i = 0; i < NUM_ELITES; i++) {
        newPopulation.push_back(currentPopulation[i]);
    }

    while (newPopulation.size() < POPULATION_SIZE) {
        // Select parents using tournament
        pair<Individual, Individual> parents = tournamentSelection(currentPopulation);

        // Crossover
        pair<Individual, Individual> children = crossover(parents.first, parents.second);

        // Mutation
        mutate(children.first);
        mutate(children.second);

        // Add children to new population
        newPopulation.push_back(children.first);
        if (newPopulation.size() < POPULATION_SIZE)
            newPopulation.push_back(children.second);
    }

    return newPopulation;
}

int main() {
	srand(time(0)); // Seed for random number generation
    vector<Individual> population = generateInitialPopulation();

    // Inputs: each is a 3x3 grid flattened to a 9-element vector
    vector<vector<double>> trainingInputs = {
        {1,1,0,0,1,0,1,1,0}, // C
        {0,1,0,0,1,0,1,1,1}, // T
        {1,0,1,1,1,1,1,0,1}, // H
        {1,0,0,1,0,0,1,0,0}  // None
    };

    // Corresponding expected outputs
    vector<vector<double>> expectedOutputs = {
        {1, 0}, // C
        {0, 1}, // T
        {1, 1}, // H
        {0, 0}  // None
    };

    int generations = 1000; // or however many you want
    for (int gen = 0; gen < generations; ++gen) {
        evaluateFitness(population, trainingInputs, expectedOutputs);

        // Open file in append mode to log fitness values
        ofstream fitnessLog("fitness_log.txt", ios::app);

        if (fitnessLog.is_open()) {
            fitnessLog << gen << "," << population[0].fitness << endl; // Log generation number & best fitness
            fitnessLog.close();
        }
        else {
            cerr << "Error: Unable to open fitness_log.txt for writing." << endl;
        }

        // Log weights and thresholds for the best individual every 200 generations
        if (gen % 200 == 0 || gen == 1000) { // Log selected generations
            ofstream genLog("generation_results.txt", ios::app);

            if (genLog.is_open()) {
                genLog << "Generation: " << gen << endl;

                vector<double> decodedParameters = decodeChromosome(population[0].chromosome);

                // Log weights for input-to-hidden layer (9x8)
                genLog << "Weights (Input to Hidden Layer):" << endl;
                for (int i = 0; i < NUM_HIDDEN_NODES; ++i) {
                    genLog << "w" << (i + 1) << "_:" << endl;
                    for (int j = 0; j < NUM_INPUTS; ++j) {
                        genLog << decodedParameters[i * NUM_INPUTS + j] << endl;
                    }
                }

                // Log weights for hidden-to-output layer (8x2)
                genLog << "\nWeights (Hidden to Output Layer):" << endl;
                int hiddenToOutputStart = NUM_INPUTS * NUM_HIDDEN_NODES;
                for (int i = 0; i < NUM_OUTPUT_NODES; ++i) {
                    genLog << "w'" << (i + 1) << "_:" << endl;
                    for (int j = 0; j < NUM_HIDDEN_NODES; ++j) {
                        genLog << decodedParameters[hiddenToOutputStart + i * NUM_HIDDEN_NODES + j] << endl;
                    }
                }

                // Log thresholds for hidden layer
                genLog << "\nThresholds (Hidden Layer):" << endl;
                int thresholdsStart = NUM_WEIGHTS;
                for (int i = 0; i < NUM_HIDDEN_NODES; ++i) {
                    genLog << decodedParameters[thresholdsStart + i] << endl;
                }

                // Log thresholds for output layer
                genLog << "\nThresholds (Output Layer):" << endl;
                for (int i = 0; i < NUM_OUTPUT_NODES; ++i) {
                    genLog << decodedParameters[thresholdsStart + NUM_HIDDEN_NODES + i] << endl;
                }

                // Compute outputs and errors for each training pattern
                genLog << "\nTraining Patterns:" << endl;
                for (size_t i = 0; i < trainingInputs.size(); ++i) {
                    vector<double> output = feedForward(trainingInputs[i], population[0]);
                    double totalError = 0.0;

                    genLog << "Pattern " << i + 1 << " - Outputs: (" << output[0] << ", " << output[1] << ") ";

                    for (size_t j = 0; j < output.size(); ++j) {
                        double error = pow(expectedOutputs[i][j] - output[j], 2);
                        totalError += error;
                    }

                    genLog << "Error: " << totalError << endl;
                }

                // Log fitness value
                genLog << "\nFitness: " << population[0].fitness << endl;
                genLog << "-------------------------------------------" << endl;
                genLog.close();
            }
            else {
                cerr << "Error: Unable to open generation_results.txt for writing." << endl;
            }
        }

        // Open file in append mode to log error values
        ofstream errorLog("error_log.txt", ios::app);

        if (errorLog.is_open()) {
            double totalGenerationError = 0.0;

            // Compute total error for the best individual in this generation
            for (size_t i = 0; i < trainingInputs.size(); ++i) {
                vector<double> output = feedForward(trainingInputs[i], population[0]); // Best individual
                for (size_t j = 0; j < output.size(); ++j) {
                    double error = expectedOutputs[i][j] - output[j];
                    totalGenerationError += pow(error, 2); // Squared error
                }
            }

            // Log generation number and total error for the best individual
            errorLog << gen << "," << totalGenerationError << endl;
            errorLog.close();
        }
        else {
            cerr << "Error: Unable to open error_log.txt for writing." << endl;
        }


        // Create next generation
        population = createNextGeneration(population);
    }

    Individual bestIndividual = population[0];
    vector<vector<double>> testInput = {
    { 0,0,0,1,1,1,1,0,1 }, // Test for 'C'
    { 0,0,1,1,1,1,0,0,1 }, // Test for 'T'
    { 1,0,1,1,1,1,1,0,1 }, // Test for 'H'
    { 1,1,1,0,1,0,0,1,1 },  // Test for 'None'
    { 1,1,0,1,0,0,1,1,0 }, // Test for 'C'
    { 0,0,1,1,1,1,0,0,1 }, // Test for 'T'
    { 1,0,1,1,1,1,1,0,1 }, // Test for 'H'
    { 0,0,1,0,1,0,1,0,0 }  // Test for 'None'
    };

    for (size_t i = 0; i < testInput.size(); ++i) {
        vector<double> result = feedForward(testInput[i], bestIndividual);
        // Interpret the prediction
        char predictedLetter;
        if (result[0] == 1 && result[1] == 0) {
            predictedLetter = 'C';
        }
        else if (result[0] == 0 && result[1] == 1) {
            predictedLetter = 'T';
        }
        else if (result[0] == 1 && result[1] == 1) {
            predictedLetter = 'H';
        }
        else if (result[0] == 0 && result[1] == 0) {
            predictedLetter = 'N'; // Representing 'None' with 'N'
        }
        else {
            predictedLetter = '?'; // For unknown patterns
        }

        cout << "Prediction for input " << i + 1 << ": " << predictedLetter << endl;
    }
    // Save weights and thresholds of the best individual to a file
    ofstream outputFile("best_individual.txt");
    if (outputFile.is_open()) {
        vector<double> decodedParameters = decodeChromosome(bestIndividual.chromosome);

        // Save weights for input-to-hidden layer (9x8)
        outputFile << "Weights (Input to Hidden Layer):" << endl;
        for (int i = 0; i < NUM_HIDDEN_NODES; ++i) {
            outputFile << "w" << (i + 1) << "_:" << endl;
            for (int j = 0; j < NUM_INPUTS; ++j) {
                outputFile << decodedParameters[i * NUM_INPUTS + j] << endl;
            }
        }

        // Save weights for hidden-to-output layer (8x2)
        outputFile << "\nWeights (Hidden to Output Layer):" << endl;
        int hiddenToOutputStart = NUM_INPUTS * NUM_HIDDEN_NODES;
        for (int i = 0; i < NUM_OUTPUT_NODES; ++i) {
            outputFile << "w'" << (i + 1) << "_:" << endl;
            for (int j = 0; j < NUM_HIDDEN_NODES; ++j) {
                outputFile << decodedParameters[hiddenToOutputStart + i * NUM_HIDDEN_NODES + j] << endl;
            }
        }

        // Save thresholds for hidden layer
        outputFile << "\nThresholds (Hidden Layer):" << endl;
        int thresholdsStart = NUM_WEIGHTS;
        for (int i = 0; i < NUM_HIDDEN_NODES; ++i) {
            outputFile << decodedParameters[thresholdsStart + i] << endl;
        }

        // Save thresholds for output layer
        outputFile << "\nThresholds (Output Layer):" << endl;
        for (int i = 0; i < NUM_OUTPUT_NODES; ++i) {
            outputFile << decodedParameters[thresholdsStart + NUM_HIDDEN_NODES + i] << endl;
        }

        outputFile.close();
        cout << "Best individual's weights and thresholds saved to 'best_individual.txt'." << endl;
    }
    else {
        cerr << "Error: Unable to open 'best_individual.txt' for writing." << endl;
    }

    _getch();
    return 0;
}