#----------------- Packages
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt



#----------------- Function
def plot_pred_histogram(true_value, 
                        plot_sk, plot_normal, plot_bgd, plot_sgd, plot_mbgd,
                        mse_sk, mse_normal, mse_bgd, mse_sgd, mse_mbgd,
                        rsq_sk, rsq_normal, rsq_bgd, rsq_sgd, rsq_mbgd):
    """Function which returns a 3 rows with 2 columns of plots comparing the predicted distribution with the testing set"""


    # Predicted values list
    dist_pred = [plot_sk, plot_normal,
                 plot_bgd, plot_sgd,
                 plot_mbgd]


    # Building the legend markers
    red_patch = mpatches.Patch(color='salmon', label='True')
    skyblue_patch = mpatches.Patch(color='skyblue', label='Sklearn')
    blue_patch = mpatches.Patch(color='blue', label='Normal')
    green_patch = mpatches.Patch(color='green', label='Batch')
    maroon_patch = mpatches.Patch(color='maroon', label='Stochastic')
    purple_patch = mpatches.Patch(color='#7F00FF', label='Minibatch')

    # Set up the legend handles
    dist_patches = [skyblue_patch, blue_patch,
                    green_patch, maroon_patch,
                    purple_patch]

    # Colors of the plots
    color_list = ['skyblue', 'blue',
                  'green', 'maroon',
                  '#7F00FF']


    # Create axes (2 x 2 ) and (1 x 1) plots
    axes = [1, 2, 3, 4, (5, 6)]

    # Append MSE and Rsquared to titles of each plot
    mse_title = [mse_sk, mse_normal,
                 mse_bgd, mse_sgd,
                 mse_mbgd]

    r2_title = [rsq_sk, rsq_normal,
                rsq_bgd, rsq_sgd,
                rsq_mbgd]


    # Create figure
    fig = plt.figure(figsize=(20, 10))

    # Iterate through each of the previous lists and populate the axes
    for ax, i, j, c, mse_, r2_ in zip(axes, dist_pred, dist_patches, color_list, mse_title, r2_title):

        # Create 3 x 2 axes
        fig_ax = fig.add_subplot(3, 2, ax)

        # Legend true value and predicted value)
        patches = [red_patch, j]

        # Create histograms of the true and predicted values
        fig_ax.hist(true_value, bins=10, density=True, color='salmon')
        fig_ax.hist(i, bins=10, alpha=0.60, density=True, color=c)

        # Figure parameters
        plt.setp(fig_ax.get_xticklabels(), fontsize=14)
        plt.setp(fig_ax.get_yticklabels(), fontsize=14)

        # Set titles
        fig_ax.set_xlabel("Output", fontsize=18)
        fig_ax.set_ylabel("Count", fontsize=18)

        # Append the MSE and Rsq values to title
        full_title = "MSE = " + str(round(mse_, 2)) + \
            ", $R^2$ = " + str(round(r2_, 3))
        fig_ax.set_title(full_title, fontsize=18)

        # Set legend
        legend = fig_ax.legend(handles=patches, loc='upper right',
                               borderaxespad=0., fontsize=16)

    # Adjust margins of subplot
    plt.subplots_adjust(bottom=0.1, right=0.8, top=2.0)

    return fig