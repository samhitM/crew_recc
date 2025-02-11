# user_data_fields.py

fields_to_fetch = {
    'Max_Hours': ('crew_online_time', 'crew_user'),  # ASSUMES crew_online_time STORES TOTAL ONLINE HOURS FOR USERS
    'Achievements': ('crew_badge', 'crew_user'),  # ASSUMES crew_badge TRACKS USER ACHIEVEMENTS
    'Total_Impression_Score': ('crew_impression', 'crew_user'),  # ASSUMES crew_impression STORES USER IMPRESSION SCORES
    'Consistent_Engagement': ('total_active_time', 'user_sessions'),  # ASSUMES total_active_sec STORES TOTAL ACTIVE SECONDS; MAY CHANGE TO user_sessions TABLE
    'Longevity': ('created_ts', 'crew_user'),  # ASSUMES created_ts REPRESENTS USER ACCOUNT CREATION TIMESTAMP
    'Event_Participation': ('event_participation', 'crew_user'),  # ASSUMES event_participation STORES USER PARTICIPATION DATA; TABLE NAMES MAY CHANGE
    'Community_Contributions': ('contributions', 'crew_user'),  # ASSUMES contributions TRACKS USER CONTRIBUTIONS TO THE COMMUNITY
    'Social_Interactions': ('social_interactions', 'crew_user'),  # ASSUMES social_interactions STORES USER SOCIAL ENGAGEMENT DATA
}
