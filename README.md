# Online Learning for Restless Bandits

This code implements and evaluates algorithms for the paper "Optimistic Whittle Index Policy: Online Learning for Restless Bandits" from AAAI 2023. In this paper we introduce UCWhittle, an upper confidence bound (UCB) based algorithm to learn Whittle index policies for restless bandits when transition probabilities are unknown.

[Optimistic Whittle Index Policy: Online Learning for Restless Bandits](https://arxiv.org/abs/2205.15372)
Kai Wang*, Lily Xu*, Aparna Taneja, Milind Tambe

Due to the sensitive nature of the maternal health data from ARMMAN, we are unable to share the real dataset but we include the synthetic data simulator.

```
@inproceedings{wang2023online,
  title={Optimistic Whittle Index Policy: Online Learning for Restless Bandits},
  author={Wang, Kai and Xu, Lily and Taneja, Aparna and Tambe, Milind},
  booktitle={Proc.~Thirty-Seventh AAAI Conference on Artificial Intelligence (AAAI-23)},
  year={2023},
}
```

This project is licensed under the terms of the MIT license.

## Files
- `main.py` - main driver
- `baselines.py` - baseline algorithms: optimal, random, WIQL
- `compute_whittle.py` - compute Whittle index for Whittle index threshold policy
- `simulator.py` - simulator based on OpenAI gym
- `uc_whittle.py` - UCWhittle: penalty-minimizing
- `ucw_value.py` - UCWhittle: value-maximizing
- `utils.py` - utilities across all files
- `get_armman_data.py` - get real ARMMAN data
- `process_results.py` - compile and process output CSVs


## Usage
To execute the code and run experiments comparing UCWhittle and the baselines, run:

```sh
python main.py
```

To vary the settings, use the options:

```sh
python main.py -N 8 -H 20 -T 30 -B 3 -D synthetic
```

The options are

- `N` - number of arms
- `H` - episode length
- `T` - number of episodes
- `B` - budget (total number of resources to spend)
- `D` - dataset to use {synthetic, real}
- `E` - number of epochs, where each epoch has a different population with different true transition probabilities
- `d` - discount factor
- `s` - random seed
- `V` - verbose output (default False)
- `L` - local code execution (will display matplotlib graphs; default False)
- `p` - prefix for file writing


## Requirements
python==3.9.12
seaborn==0.11.2
matplotlib==3.5.1
gym==0.21.0
pandas==1.4.1
scipy==1.7.3
scikit-learn==1.0.2
numpy==1.21.5
gurobi==9.5.1
