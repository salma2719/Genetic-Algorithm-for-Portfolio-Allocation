# Genetic-Algorithm-for-Portfolio-Allocation
This Python script implements a Genetic Algorithm for portfolio optimization. The algorithm aims to find the optimal asset allocation for a portfolio based on user-defined constraints, such as the desired mean return, environmental score, social score, and governance score. Each potential solution represents an individual in the population and is encoded into a string or chromosome. In the case of ESG-constrained portfolio optimization, a chromosome within the genetic algorithm represents a potential investment portfolio. The chromosome is composed of genes, each of which encodes the allocation weight of a specific asset in the portfolio. The entire chromosome, which is equivalent to an individual in a population, represents a complete portfolio allocation. An individual's fitness is determined with regard to a certain objective
function called the fitness function.

# Features
- Genetic Algorithm for portfolio optimization.
- User input for customizing parameters, mean returns, covariance matrix, and scores.
