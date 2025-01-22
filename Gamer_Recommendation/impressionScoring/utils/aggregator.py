import pandas as pd

class Aggregator:
    """
    A class to aggregate user impressions over different time periods (daily, weekly, monthly).
    """

    def aggregate_impressions(self, user_scores):
        """
        Aggregate user impressions into daily, weekly, and monthly totals.

        Parameters:
            user_scores (list of dict): A list of dictionaries containing user scores and impression data.
                Each dictionary should include:
                - 'Timestamp' (str or datetime): The timestamp of the impression.
                - 'Impressions' (int or float): The number of impressions.

        Returns:
            dict: A dictionary containing aggregated impressions:
                - 'Daily_Impressions': Pandas Series with daily impression totals.
                - 'Weekly_Impressions': Pandas Series with weekly impression totals.
                - 'Monthly_Impressions': Pandas Series with monthly impression totals.
        """
        df = pd.DataFrame(user_scores)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True).dt.tz_localize(None)
        df['Date'] = df['Timestamp'].dt.date
        df['Week'] = df['Timestamp'].dt.to_period('W').apply(lambda r: r.start_time)
        df['Month'] = df['Timestamp'].dt.to_period('M').apply(lambda r: r.start_time)

        daily_impressions = df.groupby('Date')['Impressions'].sum()
        weekly_impressions = df.groupby('Week')['Impressions'].sum()
        monthly_impressions = df.groupby('Month')['Impressions'].sum()

        return {
            'Daily_Impressions': daily_impressions,
            'Weekly_Impressions': weekly_impressions,
            'Monthly_Impressions': monthly_impressions
        }
