import matplotlib.pyplot as plt
import pandas as pd

class Plotter:
    """
    A class to plot distributions of various user score types.
    """

    def plot_distributions(self, user_data_list, user_ids):
        """
        Plot histograms for the distributions of scores for specified user IDs.

        Parameters:
            user_data_list (list of dict): A list of dictionaries containing user data.
                Each dictionary should include:
                - 'User_ID' (str or int): The unique identifier for a user.
                - 'Topological_Score' (float): The topological score for the user.
                - 'User_Feature_Score' (float): The feature-based score for the user.
                - 'Website_Impressions' (float): The website impressions score for the user.
                - 'Total_Score' (float): The total score for the user.
            user_ids (list of str or int): A list of user IDs to filter the data for plotting.

        Returns:
            None: Displays the histograms for the specified score types.
        """
        df = pd.DataFrame(user_data_list)
        df = df[df['User_ID'].isin(user_ids)]

        plt.figure(figsize=(12, 10))
        for i, score_type in enumerate(['Topological_Score', 'User_Feature_Score', 'Website_Impressions', 'Total_Score']):
            plt.subplot(2, 2, i + 1)
            plt.hist(df[score_type], bins=30, alpha=0.7, label=score_type, color=f'C{i}')
            plt.title(f"Distribution of {score_type}")
            plt.xlabel("Scores")
            plt.ylabel("Frequency")
            plt.legend()
            plt.grid()
        plt.suptitle("Score Distributions")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
