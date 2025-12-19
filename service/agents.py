import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent

# Load in memory
ws_games = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/KoEx6OWGb5iavmQWIQ6hMQ/world-series-games.csv")
playoff_games = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/Nd4TGZ1HYlZc-p8s06KYCg/playoff-games.csv")
regular_games = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/coIqzejj3J9DlSftshDCqg/regular-season-games.csv")
team_stats = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_Rsg4b5HAFYefjMZ7pJaag/team-stats.csv")
pitching_stats = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/x1dk2AsQQ0COacggGBd68w/pitcher-stats.csv")
batting_stats = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/o4jTZU_aepNOBdYH6L32iw/player-batting-stats.csv")

# Batting (player_batting_stats.csv)
def batting_agent(llm): 
    return create_pandas_dataframe_agent(
        llm, batting_stats,
        verbose=True,
        allow_dangerous_code=True,
        agent_executor_kwargs={"handle_parsing_errors": True}
    )

# Pitching (pitcher_stats.csv)
def pitching_agent(llm):
    return create_pandas_dataframe_agent(
        llm, pitching_stats,
        verbose=True,
        allow_dangerous_code=True,
        agent_executor_kwargs={"handle_parsing_errors": True}
)

# Teams (team_stats.csv)
def team_agent(llm):
    return create_pandas_dataframe_agent(
        llm, team_stats,
        verbose=True,
        allow_dangerous_code=True,
        agent_executor_kwargs={"handle_parsing_errors": True}
    )

# World Series games (world_series_games.csv)
def games_ws_agent(llm):
    return create_pandas_dataframe_agent(
        llm, ws_games,
        verbose=True,
        allow_dangerous_code=True,
        agent_executor_kwargs={"handle_parsing_errors": True}
    )

# Postseason games (playoff_games.csv)
def games_playoffs_agent(llm):
    return create_pandas_dataframe_agent(
        llm, playoff_games,
        verbose=True,
        allow_dangerous_code=True,
        agent_executor_kwargs={"handle_parsing_errors": True}
    )

# Regular season games (regular_season_games.csv)
def games_regular_agent(llm):
    return create_pandas_dataframe_agent(
        llm, regular_games,
        verbose=True,
        allow_dangerous_code=True,
        agent_executor_kwargs={"handle_parsing_errors": True}
    )