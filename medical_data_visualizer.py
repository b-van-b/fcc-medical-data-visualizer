import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv("medical_examination.csv")

# Add 'overweight' column
df["overweight"] = ((df.weight / (df.height / 100) ** 2) > 25) * 1

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
df.cholesterol = (df.cholesterol > 1) * 1
df.gluc = (df.gluc > 1) * 1

# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = df.melt(
        id_vars=["cardio"],
        value_vars=["active", "alco", "cholesterol", "gluc", "overweight", "smoke"],
    )

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    # df_cat = None

    # Draw the catplot with 'sns.catplot()'
    catplot = sns.catplot(
        x="variable", col="cardio", hue="value", kind="count", data=df_cat
    )
    catplot.set(ylabel="total")

    # Get the figure for the output
    fig = catplot.fig

    # Do not modify the next two lines
    fig.savefig("catplot.png")
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    # drop diastolic > systolic blood pressure
    drop_bp = df.ap_lo <= df.ap_hi
    # drop height < 2.5th or > 97.5th
    drop_ht = (df.height >= df.height.quantile(0.025)) & (
        df.height <= df.height.quantile(0.975)
    )
    # drop weight < 2.5th or > 97.5th
    drop_wt = (df.weight >= df.weight.quantile(0.025)) & (
        df.weight <= df.weight.quantile(0.975)
    )
    # apply filters
    df_heat = df[drop_bp & drop_ht & drop_wt]

    # print(f"Dropped {len(df)-len(df_heat)} rows.")

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    with sns.axes_style("ticks"):
        fig, ax = plt.subplots(figsize=(12, 7))
        ax = sns.heatmap(
            corr,
            square=True,
            annot=True,
            linewidths=0.5,
            fmt=".1f",
            mask=mask,
        )
    # Draw the heatmap with 'sns.heatmap()'

    # Do not modify the next two lines
    fig.savefig("heatmap.png")
    return fig
