import matplotlib.pyplot as plt

class Plotter:
    """
    A class to plot distributions of various user score types and impressions.
    """
    
    def __init__(self):
        """
        Initializes the Plotter class and applies a predefined plot style.
        """
        plt.style.use('seaborn-darkgrid')  # Apply a better style
    
    def plot_scores(self, user_data, user_ids):
        """
        Plots score distributions for different score types across users.

        Parameters:
            user_data (dict): A dictionary containing score data for users.
                Example format:
                {
                    'user_id_1': {
                        'Topological_Score': 0.4,
                        'User_Feature_Score': 0.05,
                        'Website_Impressions': 0.0,
                        'Total_Score': 0.215
                    },
                    ...
                }
            user_ids (list of str or int): A list of user IDs to filter the data for plotting.

        Returns:
            None: Displays a line plot for different score types.
        """
        score_types = ['Topological_Score', 'User_Feature_Score', 'Website_Impressions', 'Total_Score']
        
        plt.figure(figsize=(12, 8))
        for score_type in score_types:
            scores = [user_data[uid].get(score_type, 0.0) for uid in user_ids if uid in user_data]
            plt.plot(user_ids, scores, marker='o', linestyle='-', label=score_type)
        
        plt.xlabel("User ID")
        plt.ylabel("Score Value")
        plt.title("User Score Distributions")
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid()
        plt.show()
    
    def plot_impressions(self, aggregated_data, user_ids):
        """
        Plots impression trends for users over time using a line chart.

        Parameters:
            aggregated_data (dict): A dictionary containing aggregated impression data for users.
                Example format:
                {
                    'user_id_1': {
                        'Daily_Impressions': {date1: value1, date2: value2, ...},
                        'Weekly_Impressions': {...},
                        'Monthly_Impressions': {...}
                    },
                    ...
                }
            user_ids (list of str or int): A list of user IDs to filter the data for plotting.

        Returns:
            None: Displays a line plot for daily impressions.
        """
        plt.figure(figsize=(12, 8))
        for user_id in user_ids:
            if user_id in aggregated_data:
                user_impressions = aggregated_data[user_id]
                daily_impressions = list(user_impressions.get('Daily_Impressions', {}).values())
                days = list(range(1, len(daily_impressions) + 1))
                plt.plot(days, daily_impressions, marker='o', linestyle='-', label=f'User {user_id}')
        
        plt.xlabel("Days")
        plt.ylabel("Impressions")
        plt.title("Daily Impressions Over Time")
        plt.legend()
        plt.grid()
        plt.show()
    
    def plot_impression_and_score_analysis(self, aggregated_data, user_data, user_ids):
        """
        Wrapper function to call both score and impression plotting functions.

        Parameters:
            aggregated_data (dict): A dictionary containing aggregated impression data for users.
            user_data (dict): A dictionary containing score data for users.
            user_ids (list of str or int): A list of user IDs to filter the data for plotting.

        Returns:
            None: Calls the individual functions to display both score and impression plots.
        """
        self.plot_scores(user_data, user_ids)
        self.plot_impressions(aggregated_data, user_ids)
