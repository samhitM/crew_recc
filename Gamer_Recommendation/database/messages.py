from database.connection import get_db_connection
from fastapi import HTTPException
from typing import List

def get_users_message_stats(user_ids: List[str]):
    """
    Fetches total messages grouped by topic ID for multiple users,
    along with topic members and last message time, sorted by message count and topic size.
    
    Parameters:
    - user_ids (List[str]): A list of user IDs for which message statistics are needed.

    Returns:
    - dict: A dictionary where keys are user IDs, and values are lists of topic statistics:
        - topicId (str): The ID of the topic.
        - role (str): The role of the user within the topic.
        - isMuted (bool): Indicates whether the topic is muted for the user.
        - totalMessages (int): The total number of messages sent in the topic.
        - topicMembers (List[str]): A list of user IDs who are members of the topic.
        - lastMessageTime (datetime): The timestamp of the most recent message in the topic.
      If no user IDs are provided, an empty dictionary `{}` is returned.

    Raises:
    - HTTPException (500): If an error occurs while querying the database.
    """
    
    if not user_ids:
        return {}

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            query = """
                SELECT 
                    tm.member_id,
                    tm.topic_id,
                    tm.role,
                    tm.is_muted,
                    COUNT(m.id) AS total_messages,
                    ARRAY_AGG(DISTINCT tm.member_id) AS topic_members,
                    MAX(m.created_ts) AS last_message_time
                FROM 
                    topic_member tm
                LEFT JOIN 
                    message m ON tm.topic_id = m.topic_id
                WHERE 
                    tm.member_id = ANY(%s)
                GROUP BY 
                    tm.member_id, tm.topic_id, tm.role, tm.is_muted
                ORDER BY 
                    total_messages DESC, LENGTH(ARRAY_TO_STRING(ARRAY_AGG(DISTINCT tm.member_id), ',')) DESC;
            """
            cur.execute(query, (user_ids,))
            results = cur.fetchall()

            # Process results into a dictionary {user_id: [stats]}
            user_stats = {}
            for row in results:
                user_id = row[0]
                if user_id not in user_stats:
                    user_stats[user_id] = []

                user_stats[user_id].append({
                    "topicId": row[1],
                    "role": row[2],
                    "isMuted": row[3],
                    "totalMessages": row[4],
                    "topicMembers": row[5],
                    "lastMessageTime": row[6],
                })

            return user_stats  # Returns {user_id: [{topicStats1}, {topicStats2}, ...]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching user message stats: {str(e)}")
    finally:
        conn.close()