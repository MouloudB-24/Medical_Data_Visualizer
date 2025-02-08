import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Determine of a person is overweight
df['overweight'] = (df['weight'] / (df['height'] * 0.01)**2 > 25).astype(int)

# Normalize cholesterol and glucose data
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)


# Bar graph
def draw_cat_plot():
    # Create a new DataFrame
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'],
        var_name='variable',
        value_name='value')

    # Group and format DataFrame df_cat
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index()
    df_cat = df_cat.rename(columns={0: 'total'})

    # Create graph
    fig = sns.catplot(data=df_cat, x='variable', y='total', kind='bar', col='cardio', hue='value')

    # Save the graph
    fig.savefig('catplot.png')

    return fig.figure


# Heat map
def draw_heat_map():
    # Calculate the limits
    height_low, height_high = df['height'].quantile([0.025, 0.975])
    weight_low, weight_high = df['weight'].quantile([0.025, 0.975])

    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                 (df['height'].between(height_low, height_high)) &
                 (df['weight'].between(weight_low, weight_high))]

    # Calculate correlation
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the correlation matrix
    sns.heatmap(
        corr,
        mask=mask,
        cmap='coolwarm',
        annot=True,
        fmt='.1f',
        linewidths=0.5,
        square=True,
        ax=ax)

    # Save figure
    fig.savefig('heatmap.png')
    return fig


if __name__ == '__main__':
    draw_cat_plot()
    draw_heat_map()
