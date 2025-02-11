import pandas as pd
from datetime import datetime

class Aggregator:
    def __init__(self):
        """Initialize the aggregator with storage for previous impressions."""
        self.previous_impressions = {}  # Stores last known impressions per user per time period

    def aggregate_impressions(self, user_scores):
        """
        Aggregate user impressions into daily, weekly, and monthly differences per user.

        Parameters:
            user_scores (list of dict): List of user impression data.
                - 'user_id' (str): The user identifier.
                - 'Timestamp' (str or datetime): The timestamp of the impression.
                - 'crew_impression' (int or float): The total impressions.

        Returns:
            dict: Aggregated impressions structured as:
                {
                    'user_id_1': {
                        'Daily_Impressions': {date_1: diff_value, date_2: diff_value, ...},
                        'Weekly_Impressions': {week_1: diff_value, week_2: diff_value, ...},
                        'Monthly_Impressions': {month_1: diff_value, month_2: diff_value, ...}
                    },
                    'user_id_2': { ... },
                    ...
                }
        """
        df = pd.DataFrame(user_scores)

        # Handle empty DataFrame case
        if df.empty or "Timestamp" not in df:
            return self._initialize_default_impressions(user_scores)

        # Convert 'Timestamp' column to datetime, handling errors
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', utc=True)

        # Drop rows where Timestamp is NaT (missing values)
        df = df.dropna(subset=['Timestamp'])

        # Ensure every user_id exists in the DataFrame
        unique_user_ids = {entry["user_id"] for entry in user_scores}
        if df.empty:
            return self._initialize_default_impressions(user_scores)

        df['Timestamp'] = df['Timestamp'].dt.tz_localize(None)  # Remove timezone info

        # Extract time periods
        df['Date'] = df['Timestamp'].dt.date
        df['Week'] = df['Timestamp'].dt.to_period('W').apply(lambda r: r.start_time if pd.notna(r) else None)
        df['Month'] = df['Timestamp'].dt.to_period('M').apply(lambda r: r.start_time if pd.notna(r) else None)

        # Replace NaN values in 'crew_impression' with 0 (prevents sum errors)
        df['crew_impression'] = df['crew_impression'].fillna(0)

        # Initialize storage for results
        aggregated_data = {user_id: {"Daily_Impressions": {}, "Weekly_Impressions": {}, "Monthly_Impressions": {}} for user_id in unique_user_ids}

        # Process each user_id separately
        for user_id in unique_user_ids:
            user_group = df[df["user_id"] == user_id].sort_values("Timestamp")

            # Compute daily, weekly, and monthly sums
            daily_sum = user_group.groupby("Date")["crew_impression"].sum().to_dict()
            weekly_sum = user_group.groupby("Week")["crew_impression"].sum().to_dict()
            monthly_sum = user_group.groupby("Month")["crew_impression"].sum().to_dict()

            # Compute differences to avoid cumulative sums
            daily_diff = self._compute_differences(user_id, "daily", daily_sum)
            weekly_diff = self._compute_differences(user_id, "weekly", weekly_sum)
            monthly_diff = self._compute_differences(user_id, "monthly", monthly_sum)

            # Store results in structured format
            aggregated_data[user_id]["Daily_Impressions"] = daily_diff
            aggregated_data[user_id]["Weekly_Impressions"] = weekly_diff
            aggregated_data[user_id]["Monthly_Impressions"] = monthly_diff

        return aggregated_data

    def _compute_differences(self, user_id, period, current_values):
        """
        Compute the difference between the current and previous period values.

        Ensures the difference is always non-negative.
        """
        previous_values = self.previous_impressions.get(user_id, {}).get(period, {})

        sorted_dates = sorted(current_values.keys())
        differences = {}

        last_value = previous_values.get(sorted_dates[0], 0) if sorted_dates else 0

        for date in sorted_dates:
            diff = max(current_values[date] - last_value, 0)  # Ensure non-negative values
            differences[date] = diff
            last_value = current_values[date]

        # Store the latest known values
        if user_id not in self.previous_impressions:
            self.previous_impressions[user_id] = {}
        self.previous_impressions[user_id][period] = current_values

        return differences

    def _initialize_default_impressions(self, user_scores):
        """
        Ensures every user ID is included with default impression values (set to 0)
        if no valid data exists.

        Parameters:
            user_scores (List[Dict]): A list of dictionaries containing user scores.
                                    Each dictionary must have a "user_id" key.

        Returns:
            Dict[int, Dict]: A dictionary mapping each user ID to their default impression data.
                            It includes daily, weekly, and monthly impressions initialized to 0.
        """
    
        unique_user_ids = {entry["user_id"] for entry in user_scores}
        today = datetime.today().date()
        start_of_week = pd.Timestamp(datetime.today().replace(hour=0, minute=0, second=0))
        start_of_month = pd.Timestamp(datetime.today().replace(day=1, hour=0, minute=0, second=0))

        default_data = {}
        for user_id in unique_user_ids:
            default_data[user_id] = {
                "Daily_Impressions": {today: 0},
                "Weekly_Impressions": {start_of_week: 0},
                "Monthly_Impressions": {start_of_month: 0},
            }

        return default_data