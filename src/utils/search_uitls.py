""" TBA """

# Dependencies
import logging
import os
from dotenv import load_dotenv
import requests
import openai


# Load API keys
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
news_api_key = os.getenv("NEWS_API_KEY")


def fetch_and_summarize_company_news(
    company_name, news_api_key, openai_api_key, model_id="gpt-3.5-turbo", max_articles=3
):
    """
    Fetches recent news articles (only titles and short descriptions) about a company
    and summarizes them using GPT-4.

    Args:
        company_name (str): The name of the company to search for.
        news_api_key (str): Your NewsAPI API key.
        openai_api_key (str): Your OpenAI API key.
        model_id (str): OpenAI model (defult is "gpt-3.5-turbo")
        max_articles (int): The maximum number of articles to fetch and summarize.

    Returns:
        list: A list of dictionaries containing the title, URL, and summary of each news article.
    """
    # Fetch recent news articles using NewsAPI
    news_url = f"https://newsapi.org/v2/everything?q={company_name}&sortBy=publishedAt&apiKey={news_api_key}"
    response = requests.get(news_url)

    if response.status_code != 200:
        print(f"Error fetching news: {response.status_code}")
        return []

    news_data = response.json()
    articles = news_data.get("articles", [])[
        :max_articles
    ]  # Limit to the specified number of articles

    if not articles:
        print("No recent news articles found.")
        return []

    # Prepare a single prompt with titles and short descriptions to summarize all articles in one API call
    prompt = f"Summarize the following recent news articles about {company_name}. Only use the title and description provided:\n\n"
    for i, article in enumerate(articles):
        title = article["title"]
        description = article["description"]
        prompt += f"Article {i+1}:\nTitle: {title}\nDescription: {description}\n\n"

    prompt += "Provide a concise summary for each article in 2-3 sentences.\n"

    # Use OpenAI API to summarize the news articles
    openai.api_key = openai_api_key
    gpt_response = openai.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )

    # Extract the summaries from the response
    summaries = gpt_response.choices[0].message.content.split("\n\n")

    # Combine the articles with their summaries
    summarized_news = []
    for i, article in enumerate(articles):
        summarized_news.append(
            {
                "title": article["title"],
                "url": article["url"],
                "summary": (
                    summaries[i].strip()
                    if i < len(summaries)
                    else "Summary not available."
                ),
            }
        )

    return summarized_news


def main():
    company_name = "google"
    summarized_news = fetch_and_summarize_company_news(
        company_name, news_api_key, openai_api_key
    )

    for news in summarized_news:
        print(
            f"Title: {news['title']}\nURL: {news['url']}\nSummary: {news['summary']}\n"
        )


if __name__ == "__main__":
    main()
