"""
Live Reddit scraper for r/changemyview threads via PRAW.

Requires a .env file with:
  REDDIT_CLIENT_ID=...
  REDDIT_CLIENT_SECRET=...
  REDDIT_USER_AGENT=nlp-project/1.0 by <your_username>

Register a script app at https://www.reddit.com/prefs/apps
"""

import os
from typing import List, Optional

from dotenv import load_dotenv

from data.schema import Argument, Debate

load_dotenv()


def _get_reddit():
    try:
        import praw
    except ImportError:
        raise ImportError("Install 'praw': pip install praw")

    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "nlp-project/1.0")

    if not client_id or not client_secret:
        raise EnvironmentError(
            "Missing Reddit credentials. Set REDDIT_CLIENT_ID and "
            "REDDIT_CLIENT_SECRET in your .env file."
        )

    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )


def _flatten_comments(comment, depth: int = 0) -> List[dict]:
    """Recursively flatten a comment tree into a list of dicts."""
    results = []
    try:
        body = comment.body
    except AttributeError:
        return results  # MoreComments object — skip

    has_delta = "∆" in body or "!delta" in body.lower()
    results.append({
        "id": comment.id,
        "body": body,
        "author": str(comment.author) if comment.author else "[deleted]",
        "score": comment.score,
        "parent_id": comment.parent_id,
        "delta": has_delta,
        "depth": depth,
    })
    for reply in comment.replies:
        results.extend(_flatten_comments(reply, depth + 1))
    return results


def scrape_cmv(
    limit: int = 50,
    sort: str = "top",
    time_filter: str = "month",
    min_comments: int = 5,
) -> List[Debate]:
    """
    Scrape r/changemyview threads and return them as Debate objects.

    Args:
        limit: number of posts to fetch (Reddit caps at 1000)
        sort: 'top', 'hot', 'new', or 'controversial'
        time_filter: 'day', 'week', 'month', 'year', 'all' (only for 'top')
        min_comments: skip threads with fewer than this many comments
    """
    reddit = _get_reddit()
    subreddit = reddit.subreddit("changemyview")

    if sort == "top":
        posts = subreddit.top(time_filter=time_filter, limit=limit)
    elif sort == "hot":
        posts = subreddit.hot(limit=limit)
    elif sort == "new":
        posts = subreddit.new(limit=limit)
    elif sort == "controversial":
        posts = subreddit.controversial(time_filter=time_filter, limit=limit)
    else:
        raise ValueError(f"Unknown sort '{sort}'")

    debates = []
    for post in posts:
        if post.num_comments < min_comments:
            continue

        post.comments.replace_more(limit=0)  # skip MoreComments placeholders
        flat_comments = []
        for c in post.comments:
            flat_comments.extend(_flatten_comments(c))

        args = [
            Argument(
                id=post.id,
                text=post.selftext,
                arg_type="claim",
                author=str(post.author) if post.author else "[deleted]",
                score=post.score,
                metadata={"title": post.title, "url": post.url},
            )
        ]
        for c in flat_comments:
            arg_type = "counter_claim" if c["delta"] else (
                "claim" if c["depth"] == 0 else "premise"
            )
            args.append(
                Argument(
                    id=c["id"],
                    text=c["body"],
                    arg_type=arg_type,
                    parent_id=c["parent_id"],
                    author=c["author"],
                    score=c["score"],
                    metadata={"delta": c["delta"], "depth": c["depth"]},
                )
            )

        debates.append(
            Debate(
                id=post.id,
                title=post.title,
                source="reddit",
                arguments=args,
                metadata={"subreddit": "changemyview", "sort": sort},
            )
        )
    return debates
