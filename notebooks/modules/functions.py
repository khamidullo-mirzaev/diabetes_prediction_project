from typing import List, Optional, Tuple
import pandas as pd

def mean_to_median(data: pd.DataFrame, figsize: Tuple[int, int], path: Optional[str] = None) -> None:
    
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    # Selecting numerical and object data
    num_data = data.select_dtypes(['int', 'float'])
    # num_data.drop(target_col, axis=1, inplace=True)

    # Calculating the difference between median and mean relative to the mean
    diff = (num_data.mean() - num_data.median()) / num_data.mean()

    # Plotting
    plt.figure(figsize=(figsize[0], figsize[1]))
    bars = sns.barplot(x=diff.values, y=diff.index, palette='colorblind', hue=diff.index)

    # Adding numeric values next to the bars
    for bar, value in zip(bars.patches, diff.values):
        if value < 0:
            plt.text(bar.get_width() - 0.01, bar.get_y() + bar.get_height() / 2, f'{int(value*100)}', 
                    ha='right', va='center', color='black', fontsize=7)
        else:
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f'{int(value*100)}', 
                    ha='left', va='center', color='black', fontsize=7)

    # Customizing title and x-ticks
    plt.title('Relation of the mean to median', fontsize=10, pad=20)
    plt.ylabel(None)
    plt.xlabel('Ratio %', fontsize=10, labelpad=20)
    plt.xticks(np.arange(-1, 1.1, 0.1), rotation=90)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.grid(axis='y', linewidth=.2, linestyle='--', color='grey', alpha=.5)
    plt.tight_layout()

    # Showing plot
    if path:
        plt.savefig(path)
    plt.show()
    

def feature_plot(data: pd.DataFrame, features: List[str], pair: str, figsize: Tuple[int, int], bins: int = 15, hue: Optional[str] = None, path: Optional[str] = None) -> None:
    
    import matplotlib.pyplot as plt
    import seaborn as sns

    num_features = len(features)
    
    # Creating a figure with subplots
    fig, axs = plt.subplots(num_features, 3, figsize=(figsize[0], figsize[1]))

    # Chechking axs is a 2D array even if num_features is 1
    if num_features == 1:
        axs = [axs]

    for i, feature in enumerate(features):
        # Distribution plot
        sns.histplot(data=data, x=feature, kde=True, bins=bins, ax=axs[i][0], hue=hue, edgecolor='w', color='darkblue')
        axs[i][0].set_title(f'{feature}', fontsize=10)
        axs[i][0].set_xlabel(None, fontsize=10)
        axs[i][0].set_ylabel('Density', fontsize=10)
        axs[i][0].tick_params(axis='x', rotation=45)
        axs[i][0].axvline(data[feature].mean(), color='red', linewidth=2, ls='--', label='MEA', alpha=.8)
        axs[i][0].axvline(data[feature].median(), color='lime', linewidth=2, ls='--', label='MED', alpha=.8)
        axs[i][0].grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.7, which='both', axis='both')

        # Scatter plot
        sns.scatterplot(data=data, x=feature, y=pair, ax=axs[i][1], hue=hue, edgecolor='w', color='darkblue')
        axs[i][1].set_title(f'{feature} <-> {pair}', fontsize=10)
        axs[i][1].set_xlabel(feature, fontsize=10)
        axs[i][1].set_ylabel(pair, fontsize=10)
        axs[i][1].tick_params(axis='x', rotation=45)
        
        # Box plot
        sns.boxplot(data=data, x=feature, width=0.5, notch=True, linewidth=1, ax=axs[i][2], 
                    boxprops=dict(facecolor="skyblue"), whiskerprops=dict(color="orange"),
                    flierprops=dict(marker='o', markerfacecolor='red', markeredgecolor='white', markersize=8), legend=False)
        axs[i][2].set_title(f'{feature}', fontsize=10)
        axs[i][2].set_ylabel('Value', fontsize=10)
        axs[i][2].set_xlabel(None, fontsize=10)
        axs[i][2].tick_params(axis='x', rotation=45)
    
    if path:
        plt.savefig(path)
    
    plt.tight_layout()
    plt.show()