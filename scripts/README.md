# Scripts

We provide in this folder some scripts to get started with the source code.
Before running these scripts, environments given in the <code>./envs/</code> folder must be properly configured, following the provided instructions.

<br>

## General

For each method presented in the paper, we provide a script to run the full transfer pipeline described in the paper, that is:

- Training the reference models
- Training the target models
- Training the adBiC layer for each state, on the reference models' outputs
- Evaluating the target models corrected by the adBic parameters learned

We provide four scripts for each of the backbone CIL methods considered in our work:

- `run_lwf.sh` for __LwF__
- `run_lucir.sh` for __LUCIR__
- `run_siw.sh` for __SIW__
- `run_ft_plus.sh` for __FT+__

For all scripts, the syntax is identical:

```bash
source ./scripts/run_lucir.sh {reference} {target} {num_workers}
```

Where `reference` and `target` are the datasets to be picked for transfer, and `num_workers` the number of workers to use in dataloaders. For example, to transfer from `1mixed100` to `cifar100` while using __LUCIR__ with 8 workers, run:

```bash
source ./scripts/run_lucir.sh 1mixed100 cif100 8
```

<br>

## Reproducing

To easily reproduce the results from the paper, we extracted logits from the __CIFAR-100__ datasets using __LUCIR__ and __LwF__ models, and trained and averaged adBiC models following the algorithm described in the paper. These logits and adBiC layers are saved in the <code>./data/</code> folder.

To directly reproduce results with <code>S=10</code> for __LUCIR__ and __LwF__, run the following command:

```bash
source ./scripts/run_eval.sh {lucir or lwf}
```

For other states and methods, you can run the full pipeline using the commands described in the [General](#general) section. Note that fully reproducing the results from the main paper will require averaging the parameters learned across multiple reference datasets, which should result in a small bump in accuracy (as described in Table 3. of the paper).

<br>

## Debugging

To make sure your environments are properly configured, you can run the debugging script which executes the full pipeline described in the paper on top of a LUCIR model.
After positioning yourself in the main folder, run the following script:

```bash
# Replacing paths in the mock dataset lists
abs_path=$(realpath ./datasets/mock_data/)
find ./datasets/ -type f -exec sed -i "s#./datasets/mock_data/#${abs_path}/#g" {} +

# Running the script
source ./scripts/run_lucir.sh mock_ref mock_target 2
```

If everything works properly, the "<code>Debugging script done</code>" prompt should appear.
