#----------------- Packages
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt



#----------------- Function
def plot_cost_gd(cost_bgd, cost_sgd, cost_mbgd,
                 bgd_iterations, sgd_iterations, mbgd_iterations):
    """Function which returns a 3 rows with 2 columns of plots comparing the predicted distribution with the testing set"""


    # Predicted values list
    cost_gd = [cost_bgd, cost_sgd, cost_mbgd]
    iteration_gd = [bgd_iterations, sgd_iterations, mbgd_iterations]

    # Building the legend markers
    green_patch = mpatches.Patch(color='green', label='Batch')
    maroon_patch = mpatches.Patch(color='maroon', label='Stochastic')
    purple_patch = mpatches.Patch(color='#7F00FF', label='Minibatch')

    # Set up the legend handles
    gd_patches = [green_patch, maroon_patch,
                    purple_patch]

    # Colors of the plots
    color_list = ['green', 'maroon',
                  '#7F00FF']


    # Create axes (2 x 2 ) and (1 x 1) plots
    axes = [1, 2, (3, 4)]

    title = ['Batch Gradient Descent', 'Stochastic Gradient Descent',
             'Minibatch Gradient Descent']



    # Create figure
    fig = plt.figure(figsize=(20, 10))

    # Iterate through each of the previous lists and populate the axes
    for ax, i, j, k, c, title_ in zip(axes, cost_gd, iteration_gd,  gd_patches, color_list, title):

        # Create 3 x 2 axes
        fig_ax = fig.add_subplot(2, 2, ax)

        # Legend true value and predicted value)
        patches = [k]

        # Create histograms of the true and predicted values
        fig_ax.plot(range(0, j, 1), i, color=c, linewidth=1, linestyle='dashed', marker = 'o')

        # Figure parameters
        plt.setp(fig_ax.get_xticklabels(), fontsize=16)
        plt.setp(fig_ax.get_yticklabels(), fontsize=16)

        # Set titles
        fig_ax.set_xlabel("Iterations", fontsize=18)
        fig_ax.set_ylabel("Cost (J)", fontsize=18)

        # Append the MSE and Rsq values to title
        fig_ax.set_title(title_, fontsize=20)

        # Set legend
        legend = fig_ax.legend(handles=patches, loc='upper right',
                               borderaxespad=0., fontsize=16)

    # Adjust margins of subplot
    plt.subplots_adjust(bottom=0.1, left = 0.3, right=0.9, top=1.5)

    return fig