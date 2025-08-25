import pandas as pd
from pytrends.request import TrendReq

def fetch_google_trends(keyword: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches daily Google Trends data for a given keyword and date range.

    Args:
        keyword (str): The keyword to search for (e.g., 'Bitcoin').
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A DataFrame with 'Date' and the keyword as columns.
                      Returns an empty DataFrame if an error occurs.
    """
    try:
        pytrends = TrendReq(hl='en-US', tz=360)

        # Build the payload
        timeframe = f'{start_date} {end_date}'
        pytrends.build_payload([keyword], cat=0, timeframe=timeframe, geo='', gprop='')

        # Get interest over time
        trends_df = pytrends.interest_over_time()

        if trends_df.empty:
            return pd.DataFrame()

        # Reset index to make 'date' a column and rename it
        trends_df = trends_df.reset_index()
        trends_df = trends_df.rename(columns={'date': 'Date', keyword: 'trends_score'})

        # Keep only the relevant columns
        trends_df = trends_df[['Date', 'trends_score']]

        # Convert Date to the same format as our price data for merging
        trends_df['Date'] = pd.to_datetime(trends_df['Date']).dt.tz_localize(None)

        return trends_df

    except Exception as e:
        print(f"An error occurred while fetching Google Trends data: {e}")
        # Return an empty dataframe in case of frequent request errors
        return pd.DataFrame(columns=['Date', 'trends_score'])
