import tweepy
import time
import pandas as pd

# ----------------------------
# STEP 1: Authenticate to Twitter API
# ----------------------------
bearer_token = "AAAAAAAAAAAAAAAAAAAAANfLtAEAAAAA6qTHFX08aTTLexEHdCCNxi0bLT8%3DM5MPLnRg9UlPG2n9HfAhGCOEIJ1YUht9uNFivUleSrPUF7Xxik"  # Replace with your Bearer Token

client = tweepy.Client(bearer_token=bearer_token)

# ----------------------------
# STEP 2: Define Function to Fetch Tweets
# ----------------------------
def fetch_tweets(query, max_results=10, tweet_limit=100, since_id=None):
    """
    Fetch tweets using the Twitter API v2, with progress tracking.

    Args:
        query (str): Search query.
        max_results (int): Number of tweets per request (max 100).
        tweet_limit (int): Total number of tweets to fetch.
        since_id (str): Fetch tweets created after this ID (optional).

    Returns:
        list: List of tweets with selected fields.
    """
    all_tweets = []
    next_token = None
    fetched_count = 0

    while fetched_count < tweet_limit:
        try:
            # Fetch tweets
            response = client.search_recent_tweets(
                query=query,
                max_results=min(max_results, tweet_limit - fetched_count),
                since_id=since_id,
                tweet_fields=["created_at", "author_id", "text", "id"],
                next_token=next_token
            )

            # Append fetched tweets
            if response.data:
                for tweet in response.data:
                    all_tweets.append({
                        "author_id": tweet.author_id,
                        "text": tweet.text,
                        "created_at": tweet.created_at,
                        "id": tweet.id
                    })
                fetched_count += len(response.data)

            # Handle pagination
            next_token = response.meta.get("next_token", None)
            if not next_token:
                break  # Exit if no more data

        except tweepy.errors.TooManyRequests:
            print("Rate limit reached. Waiting for 15 minutes...")
            time.sleep(900)  # Wait for rate limit reset
        except Exception as e:
            print(f"An error occurred: {e}")
            break

    return all_tweets

# ----------------------------
# STEP 3: Define Query and Fetch Data
# ----------------------------

query = "#stocks -is:retweet lang:en"  # Customize your query
tweet_limit = 50  # Adjust as needed
max_results = 10  # Max tweets per request
since_id = None  # Set to the ID of the last fetched tweet if resuming
print("Fetching tweets...")
tweets = fetch_tweets(query=query, max_results=max_results, tweet_limit=tweet_limit, since_id=since_id)

    # ----------------------------
    # STEP 4: Save Data to CSV and Update Since ID
# ----------------------------
if tweets:
    print(f"Fetched {len(tweets)} tweets. Saving to CSV...")
        
    # Save fetched tweets
    df = pd.DataFrame(tweets)
    df.to_csv("tweets.csv", index=False, mode='a', header=not pd.read_csv("tweets.csv").empty if "tweets.csv" in locals() else True)

    # Update since_id to the most recent tweet's ID
    since_id = tweets[0]["id"]
    print(f"Updated since_id to: {since_id}")
    print("Tweets saved to 'tweets.csv'")
else:
    print("No tweets fetched.")
