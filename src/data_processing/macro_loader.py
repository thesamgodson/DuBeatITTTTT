import pandas as pd
from pytrends.request import TrendReq
from datetime import datetime
import time

def get_google_trends_data(keywords: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches daily Google Trends data for a list of keywords.
    Handles potential rate limiting by retrying.

    Args:
        keywords (list): A list of keywords to get trends for.
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A DataFrame with the daily trend data, indexed by date.
                      Returns an empty DataFrame on failure.
    """
    pytrends = TrendReq(hl='en-US', tz=360)
    timeframe = f'{start_date} {end_date}'

    attempts = 3
    for attempt in range(attempts):
        try:
            pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo='', gprop='')
            trends_df = pytrends.interest_over_time()

            if trends_df.empty:
                print("Warning: Google Trends returned no data for the specified keywords and timeframe.")
                return pd.DataFrame()

            if 'isPartial' in trends_df.columns:
                trends_df = trends_df.drop(columns=['isPartial'])

            # Resample to daily frequency and forward-fill
            trends_df = trends_df.asfreq('D', method='ffill')

            print(f"Successfully fetched Google Trends data for: {keywords}")
            return trends_df

        except Exception as e:
            print(f"Attempt {attempt + 1}/{attempts} failed: {e}")
            if "response code 429" in str(e).lower():
                print("Rate limited. Waiting for 60 seconds before retrying...")
                time.sleep(60)
            else:
                break # Don't retry on other errors

    print("Failed to fetch Google Trends data after several attempts.")
    return pd.DataFrame()


if __name__ == '__main__':
    # Example usage:
    start = '2020-01-01'
    end = '2024-12-31'
    keywords_to_fetch = ['bitcoin', 'crypto']

    trends_data = get_google_trends_data(keywords_to_fetch, start, end)

    if not trends_data.empty:
        print("\nSample of fetched data:")
        print(trends_data.head())
        print("\nData summary:")
        trends_data.info()
