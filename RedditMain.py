import praw

# Authentication
reddit = praw.Reddit(
    client_id="xvWe5LEUW1ZWNSdvBHphRg",
    client_secret="YiQOEIA23zyL8pJdknWS3lT_h63Zg",
    user_agent="MarketEchoScraper/1.0 by SaurabhKumbhar"
)

# Fetch posts
subreddit = reddit.subreddit("stocks")
for post in subreddit.new(limit=10):
    print(f"Title: {post.title}, Upvotes: {post.score}, URL: {post.url}")
