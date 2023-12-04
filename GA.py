import numpy as np

def initialize_population(size, N):
    return np.random.rand(size, N)

def calculate_fitness(chromosome, mean_returns, cov_matrix, environmental_scores, social_scores, governance_scores, R, E, S, G):
    portfolio_mean_return = np.sum(mean_returns * chromosome)
    portfolio_covariance = np.sum(np.outer(chromosome, chromosome) * cov_matrix)
    environmental_constraint = np.sum(environmental_scores * chromosome)
    social_constraint = np.sum(social_scores * chromosome)
    governance_constraint = np.sum(governance_scores * chromosome)
    
    # Calculate fitness based on constraints
    constraint_penalty = max(0, R - portfolio_mean_return) + \
                         max(0, E - environmental_constraint) + \
                         max(0, S - social_constraint) + \
                         max(0, G - governance_constraint)
    
    return portfolio_covariance + constraint_penalty

def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1) - 1)
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

def mutate(chromosome, mutation_rate):
    mutation_indices = np.where(np.random.rand(len(chromosome)) < mutation_rate)
    chromosome[mutation_indices] = np.random.rand(len(mutation_indices))
    return chromosome

def post_process_portfolio(portfolio):
    portfolio /= np.sum(portfolio)
    return portfolio

def genetic_algorithm(N, R, E, S, G, population_size, mutation_rate, num_generations, mean_returns, cov_matrix, environmental_scores, social_scores, governance_scores):
    population = initialize_population(population_size, N)
    
    for generation in range(num_generations):
        # fitness of the current population is evaluated
        fitness_values = np.array([calculate_fitness(chromosome, mean_returns, cov_matrix, environmental_scores, social_scores, governance_scores, R, E, S, G) for chromosome in population])
        
        # parents get selected
        selected_indices = np.random.choice(population_size, size=population_size, replace=False)
        parents = population[selected_indices]
        
        # new population is created through crossover and mutation
        new_population = []
        for i in range(0, population_size, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]
            
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            
            new_population.extend([child1, child2])
        
        population = np.array(new_population)
    
    best_portfolio = post_process_portfolio(population[np.argmin(fitness_values)])
    return best_portfolio

def main():
    # Get user input for parameters
    N = int(input("Enter the number of assets (N): "))
    R = float(input("Enter the desired mean return (R): "))
    E = float(input("Enter the desired environmental score (E): "))
    S = float(input("Enter the desired social score (S): "))
    G = float(input("Enter the desired governance score (G): "))
    population_size = int(input("Enter the population size: "))
    mutation_rate = float(input("Enter the mutation rate: "))
    num_generations = int(input("Enter the number of generations: "))
    
    # Get user input for mean returns, covariance matrix, and scores
    mean_returns = np.array([float(input(f"Enter mean return for asset {i + 1}: ")) for i in range(N)])
    cov_matrix = np.array([[float(input(f"Enter covariance between assets {i + 1} and {j + 1}: ")) for j in range(N)] for i in range(N)])
    environmental_scores, social_scores, governance_scores = get_input_scores(N)

    best_portfolio = genetic_algorithm(N, R, E, S, G, population_size, mutation_rate, num_generations, mean_returns, cov_matrix, environmental_scores, social_scores, governance_scores)

    print("Best Portfolio Proportions:", best_portfolio)

if __name__ == "__main__":
    main()
