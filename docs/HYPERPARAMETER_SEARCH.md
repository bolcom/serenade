Hyperparameter Search
===

Hyperparameter optimization is the problem of optimizing a loss function over a configuration space. Among different methods provided in this repository for performing such task, we select the **Tree-Structured Parzen Estimator (TPE)** as the default one because it quickly achieves low validation errors when compared to Exhaustive Grid Search (see the [paper by Bergstra et al.](http://proceedings.mlr.press/v28/bergstra13.pdf) for more details). From the original paper:

> "In this work we restrict ourselves to tree-structured configuration spaces. Configuration spaces are tree-structured in the sense that some leaf variables (e.g. the number of hidden units in the 2nd layer of a Deep Belief Network - DBN) are only well-defined when node variables (e.g. a discrete choice of how many layers to use) take particular values."
> 
> <cite>Bergstra et al.</cite>

There is a [crate in Rust with the TPE implementation (link)](https://docs.rs/tpe/0.1.1/tpe/). This approach instantiates one optimizer for each hyperparameter to optimize -- each instance of TpeOptimizer tries to search out the value which could minimize/maximize the evaluation result for such hyperparameter. 

How does it work? After defining the objective function and the domain of the hyperparameters space:
1. Randomly select sets of hyperparameters and obtain their score (goal metric value by using the objective function).
2. Sort the collected observations by score and divide them into two groups based on a threshold. The first group (x1) contains observations that have scores above the threshold and the second one (x2) - all other observations.
3. Two densities l(x1) and g(x2) are modeled using Parzen Estimators (also known as kernel density estimators) which are a simple average of kernels centered on existing data points.
4. Draw sample hyperparameters from l(x1), evaluating them in terms of l(x1)/g(x2), and returning the set that yields the minimum value under l(x1)/g(x1) corresponding to the greatest expected improvement. These hyperparameters are then evaluated on the objective function.
5. Update the observation list from step 1.
6. Repeat steps 2-5 with a fixed number of trials or until the time limit is reached.
