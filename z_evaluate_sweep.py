import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import wandb

api = wandb.Api()

sweep_id = 'zftou44c'
sweep = api.sweep(f"nicoweio/dsea-corn/{sweep_id}")

# filter successful runs
runs = list(filter(lambda run: run.state == 'finished', sweep.runs))

print(f"{len(sweep.runs)} total runs, {len(runs)} finished")

# █ v1
# df = pd.DataFrame([dict(run.summary) | dict(run.config) for run in runs])
# █ END v1

# █ v2
# Each run has multiple values for metrics like wd_all
# Create a new dataframe with one row per *metric*
# The columns shall get the suffix '_single', e.g. wd_all_single

data_list = []
for run in runs:
    metric_keys = [key for key in run.summary.keys() if key.endswith('_all')]
    metric_lengths = [len(run.summary[key]) for key in metric_keys]
    assert len(set(metric_lengths)) == 1, "Not all metrics have same length"
    for i in range(metric_lengths[0]):
        data = {}
        for key in metric_keys:
            newkey = key.replace('_all', '_single')
            data[newkey] = run.summary[key][i]
        data_list.append(
            data | dict(run.config)
        )

df_single = pd.DataFrame(data_list)
# █ END v2


CONFIG = {
    'batch_size': ['boxplot', ],
    'epsilon': ['scatterplot', ],
    'learning_rate': ['scatterplot', ],
    'num_epochs': ['boxplot', ],
    'J_factor': ['boxplot', ],
}
METRICVAR = 'wd'

for configvar in CONFIG:
    for plot_kind in CONFIG[configvar]:
        # metrics = []
        # hyperparams = []
        # for run in runs:
        #     for val in run.summary[f"test/{METRICVAR}_all"]:
        #         metrics.append(val)
        #         hyperparams.append(run.config[configvar])

        # # sort by hyperparam
        # hyperparams, metrics = zip(*sorted(zip(hyperparams, metrics)))
        # hyperparams = list(hyperparams)
        # metrics = list(metrics)

        # sort by hyperparam and convert back to list
        # metrics, hyperparams = list(zip(*sorted(zip(metrics, hyperparams))))

        if plot_kind == 'boxplot':
            sns.boxplot(data=df_single, y=f'test/{METRICVAR}_single', x=configvar)
            # sns.boxplot(x=hyperparams, y=metrics)
        elif plot_kind == 'scatterplot':
            sns.scatterplot(data=df_single, y=f'test/{METRICVAR}_single', x=configvar, s=100, marker='s')

        sns.despine()
        # plt.title(f"{configvar} vs {METRICVAR}")
        plt.xlabel(configvar)
        plt.ylabel(METRICVAR)
        plt.savefig(f"build/plots/{configvar}_vs_{METRICVAR}_{plot_kind}.pdf")
        plt.clf()
