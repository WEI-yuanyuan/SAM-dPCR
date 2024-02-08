import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import csv
import os

def statsPlot (outputDirectory):
    plt.rcParams['font.family'] = 'Arial'
    for filename in os.listdir(outputDirectory):
        # Check for txt files
        if filename.endswith(".txt"):
            input_file_path = os.path.join(outputDirectory, filename)
            file_base_name = os.path.splitext(filename)[0]
            output_file_path = os.path.join(outputDirectory, file_base_name)

            # Read the data from the txt file
            with open(input_file_path) as f:
                data = np.array([line.split() for line in f.readlines()[1:]])
                areas = data[:, 0].astype(float)
                ious = data[:, 1].astype(float)
                stability_scores = data[:, 2].astype(float)
                classifications = data[:, 3]

                # Convert area to diameter
                diameters = 2 * np.sqrt(areas / np.pi)

                # Negate the diameter for negative classifications
                diameters[classifications == 'Negative'] *= -1

                # Plot 3D scatter plot of diameters, IoUs, and stability scores
                scatterFig = plt.figure(figsize=(8, 6))
                ax = scatterFig.add_subplot(111, projection='3d')
                # Set IoU and Stability Score axes to 0-1
                ax.set_ylim(np.min(ious), 1)
                ax.set_zlim(np.min(stability_scores), 1)
                # Set the background to white
                # Set the background to white
                scatterFig.patch.set_facecolor('white')
                ax.set_facecolor('white')  # This sets the background of the plot itself

                # Remove the panes (sides of the 3d box)
                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False

                # Set the linewidth of the gridlines
                ax.xaxis._axinfo["grid"]['linewidth'] = 0.5
                ax.yaxis._axinfo["grid"]['linewidth'] = 0.5
                ax.zaxis._axinfo["grid"]['linewidth'] = 0.5
                # Define custom colors
                colors = {'Positive': '#FF1111', 'Negative': '#1010FF'}

                # Plot each class with a different color
                for label in np.unique(classifications):
                    indices = np.where(classifications == label)
                    ax.scatter(diameters[indices], ious[indices], stability_scores[indices], color = colors[label], label=label, s=50, alpha=None)

                # Labeling
                ax.set_xlabel('Diameter/(pixels)', fontdict={'fontname': 'Arial', 'fontsize': 15}, labelpad=10)
                ax.set_ylabel('Predicted IoU', fontdict={'fontname': 'Arial', 'fontsize': 15}, labelpad=10)
                ax.set_zlabel('Stability score', fontdict={'fontname': 'Arial', 'fontsize': 15}, labelpad=10)
                ax.legend(prop = {'family': 'Arial'}, loc='upper left')

                # Adjust the aspect ratio
                ax.set_box_aspect([np.ptp(diameters), 40, 40])

                # Save the 3D scatter plot
                scatterFig.savefig(output_file_path + "_3d_scatter", dpi=300)
                plt.close(scatterFig)
                print(f"Saved: {output_file_path}_3d_scatter.png")
                
                # Plot error bar plot of different class of droplets diameters
                positive_diameters = diameters[classifications == 'Positive']
                negative_diameters = np.abs(diameters[classifications == 'Negative'])

                # 计算每个类别的平均直径和标准差
                positive_mean = np.mean(positive_diameters)
                positive_std = np.std(positive_diameters)
                negative_mean = np.mean(negative_diameters)
                negative_std = np.std(negative_diameters)

                # 创建柱形图
                barFig = plt.figure(figsize=(3, 6))  # Adjust the figsize to make the plot wider
                bars = plt.bar(['+', '-'], [positive_mean, negative_mean], color=['#E887D4', '#7FACD6'], edgecolor='black', linewidth=1.2)

                # 添加误差条
                plt.errorbar(['+', '-'], [positive_mean, negative_mean], yerr=[positive_std, negative_std], fmt='none', color='black', capsize=5)

                # Adding labels and legend
                plt.xlabel('Droplet class', fontdict={'fontname': 'Arial', 'fontsize': 15})
                plt.ylabel('Droplet diameter', fontdict={'fontname': 'Arial', 'fontsize': 15})
                plt.ylim(0, None)
                plt.tight_layout()

                # Save the bar plot
                plt.savefig(output_file_path + "_bar_plot", dpi=300)
                plt.close(barFig)
                print(f"Saved: {output_file_path}_bar_plot.png")