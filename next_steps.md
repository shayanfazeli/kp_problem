## Inverse Reinforcement Learning / Apprenticeship Learning

## Causal Inference - Other Approaches
* Bayesian networks, incorporating more domain knowledge
* Temporal graph representation learning, encoding the structure of the data if any

## Interpretability
* Leveraging gradient-based interpretations (`LIME` in particular, for tabular, and leveraging grad/feature-based methods implemented in `captum` for DNNs)

## Deep Neural Networks
* Selecting feature-set and balanced (either via weighing or sampling) representations
* Preparing an NN-structure clearly disentangling the role of the treatment (for example, concatenating the corresponding representation, with an additive sum of the representations of the rest, or enforcing attention weights, etc.)
* Training and hyper-parameter tuning on small subsets to validate

## Reinforcement Learning
* Leveraging papers such as [this](https://arxiv.org/pdf/1710.11248.pdf) and [imitation](https://imitation.readthedocs.io/en/latest/algorithms/airl.html) codebase. 