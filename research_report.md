## methodology/data source

## illegibility metrics
- explain how each one is calculated
- correlation of different illegibility metrics to each other across the dataset

## illegibility vs length

## illegibility vs correctness

## length vs correctness

## Is illegibility contagious?
- How much does early illegibility predict later illegibility?

## Do illegible chunks have more entropy? Does entropy show up earlier or later?
- Need to think this one through because entropy is conditional on earlier tokens

## How does the model decide when to stop?
- Plot probability of think tokens at each position
- Plot probability of think tokens after each chunk
- What's the correlation between legibility of a given chunk and end think token probability?

## Which components of reasoning are important for the answer?
- Figure out which questions have an answer inside {boxed}
- For those questions, which tokens have the highest direct logit attribution to the answer token? what if we ablate them?
- For those questions, which chunks have the highest average DLA to the answer token?
- What's the correlation between legibility and these metrics?

## Rollout generation
- explain how questions were selected and how rollouts were generated

### Which chunks are the most important in determinining the correct answer?
- Plot counterfactual importance

### Do our rollouts give us any more information on illegibility contagion?