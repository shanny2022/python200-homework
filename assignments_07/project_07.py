"""Explore the course World Happiness CSVs using a conversational code agent."""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import pearsonr
from smolagents import CodeAgent, OpenAIServerModel, tool

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "happiness_project"
OUTPUTS_DIR = BASE_DIR / "outputs"
df = None

COLUMN_NAMES = {
    "Country": "country",
    "Regional indicator": "region",
    "Happiness score": "happiness_score",
    "Ladder score": "happiness_score",
    "GDP per capita": "gdp_per_capita",
}

# --- Task 1: Tools ---
@tool
def load_happiness_data() -> dict:
    """Load and merge the 2015–2024 World Happiness CSVs into memory.

    Read semicolon delimiters and decimal commas, normalize the core column
    names, and derive each year from its filename. Replaces the shared dataset.

    Returns:
        Dataset shape and column names, or an error dictionary if loading fails.
    """
    global df
    frames = []
    try:
        paths = sorted(DATA_DIR.glob("world_happiness_*.csv"))
        if not paths:
            return {"error": f"No yearly CSVs found in {DATA_DIR}."}
        for path in paths:
            year_df = pd.read_csv(path, sep=";", decimal=",")
            year_df = year_df.rename(columns=COLUMN_NAMES)
            year_df["year"] = int(path.stem.split("_")[-1])
            required = {"country", "region", "happiness_score", "gdp_per_capita"}
            if not required.issubset(year_df.columns):
                return {"error": f"Missing required columns in {path.name}."}
            for column in ("happiness_score", "gdp_per_capita"):
                year_df[column] = pd.to_numeric(year_df[column], errors="raise")
            frames.append(year_df)
        df = pd.concat(frames, ignore_index=True)
    except (OSError, ValueError) as exc:
        return {"error": str(exc)}
    return {"shape": df.shape, "columns": df.columns.tolist()}


@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for one loaded column.

    Args:
        column: Exact column name, for example happiness_score.

    Returns:
        The pandas describe dictionary, or an error if data or column is missing.
    """
    if df is None:
        return {"error": "Load the happiness data first."}
    if column not in df:
        return {"error": f"Unknown column: {column}."}
    return df[column].describe().to_dict()


@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute Pearson correlation and a two-sided p-value for numeric columns.

    Use complete finite pairs. A rounded p-value of 0.0 means very small,
    not exactly zero. Pooled country-year correlation does not establish causation
    or independence of repeated country observations.

    Args:
        col1: First numeric column name.
        col2: Second numeric column name.

    Returns:
        Column names, pearson_r and p_value rounded to four decimals, or an error.
    """
    if df is None:
        return {"error": "Load the happiness data first."}
    if col1 not in df or col2 not in df:
        return {"error": "One or both columns were not found."}
    if not all(pd.api.types.is_numeric_dtype(df[c]) for c in (col1, col2)):
        return {"error": "Both columns must be numeric."}
    pairs = df[[col1, col2]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(pairs) < 2 or any(pairs.iloc[:, i].nunique() < 2 for i in (0, 1)):
        return {"error": "Need at least two complete pairs and nonconstant columns."}
    r, p = pearsonr(pairs.iloc[:, 0], pairs.iloc[:, 1])
    return {"col1": col1, "col2": col2,
            "pearson_r": round(float(r), 4), "p_value": round(float(p), 4)}


@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> list[dict] | dict:
    """Rank countries in a selected year by a numeric column in descending order.

    Args:
        column: Numeric ranking column, for example happiness_score.
        year: Dataset year to filter, from 2015 through 2024.
        n: Positive number of countries to return; defaults to five.

    Returns:
        A list of country/value dictionaries on success, or an error dictionary.
    """
    if df is None:
        return {"error": "Load the happiness data first."}
    if column not in df or not pd.api.types.is_numeric_dtype(df[column]):
        return {"error": "Choose an existing numeric ranking column."}
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        return {"error": "n must be a positive integer."}
    if isinstance(year, bool) or not isinstance(year, int) or year not in df["year"].values:
        return {"error": "Choose an available integer year (2015–2024)."}
    selected = df.loc[df["year"] == year].dropna(subset=[column])
    return selected.sort_values(column, ascending=False).head(n)[["country", column]].to_dict("records")


# --- Task 2: Agent ---
SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing
correlations, and ranking countries. Write Python code directly only when tools
are insufficient, such as custom plots or grouped comparisons.
For custom analyses use happiness_df, the normalized DataFrame supplied in
additional_args after loading. load_happiness_data returns metadata only, not
rows: never construct a DataFrame from that result. Do not reload CSVs manually.
For a regional time trend use the mean happiness score per region and year.
Save figures to outputs/, label axes and legend clearly, and close figures.
Treat correlations as associations, not causal effects. A p-value rounded to
0.0 is not exactly zero; repeated country-year observations limit inference.
Be concise and student-friendly. Only report results you actually obtained.
"""


def build_agent():
    load_dotenv(BASE_DIR.parent / ".env")
    model = OpenAIServerModel(api_key=os.environ["OPENAI_API_KEY"], model_id="gpt-4o-mini")
    return CodeAgent(
        tools=[load_happiness_data, summarize_column, compute_correlation, get_top_n_countries],
        model=model, instructions=SYSTEM_PROMPT,
        additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats"],
        max_steps=8,
    )


if __name__ == "__main__":
    os.chdir(BASE_DIR)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    agent = build_agent()
    # --- Task 3: Guided queries ---
    queries = [
        "Load the happiness data and tell me its shape and column names.",
        "Summarize the happiness_score column.",
        "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
        "Show me the top 5 happiest countries in 2020.",
        "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to outputs/happiness_by_region.png.",
    ]
    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = agent.run(query, reset=False,
                             additional_args={"happiness_df": df} if df is not None else {})
        print(response)
    if not (OUTPUTS_DIR / "happiness_by_region.png").is_file():
        raise RuntimeError("The agent did not save the required regional plot.")
    print("Verified: outputs/happiness_by_region.png exists.")

    # --- Task 4: My own questions ---
    my_query_1 = "Which five countries had the highest GDP per capita in 2024?"
    response_1 = agent.run(my_query_1, reset=False, additional_args={"happiness_df": df})
    print(response_1)
    # Observed: generated Python called get_top_n_countries, then returned its ranking.
    my_query_2 = (
        "Among countries present in both 2015 and 2024, which five had the largest "
        "increase in happiness_score? Join by country, compute 2024 minus 2015, "
        "and show both scores and the change. Note any limitation of this comparison."
    )
    response_2 = agent.run(my_query_2, reset=False, additional_args={"happiness_df": df})
    print(response_2)
    # Observed: custom Python joined 2015 and 2024 by country, computed changes,
    # sorted them, and returned the top five. No extra analysis tool was needed.

# --- Reflection ---
# 1. Query 3 returned r=0.6313 and p=0.0 and labeled it statistically significant.
#    The numbers match the tool, but the explanation was incomplete: this is a
#    moderately strong positive association, and 0.0 is a rounded tiny p-value,
#    not zero uncertainty. Repeated country observations also limit the usual
#    independence assumption, and this association does not establish causation.
# 2. The agent initially tried .shape on the loader's metadata dictionary, then
#    recovered by indexing its keys. That surprised me because the tool clearly
#    describes its return value. Later, it correctly generated a regional mean
#    line plot and joined countries across 2015 and 2024 without a custom tool.
#    Its comparison gave Ivory Coast the largest increase, 1.4301 points, but
#    exact country-name matching can exclude countries renamed between years.
# 3. A compare_country_years tool could join two years, report missing countries,
#    and return score changes. It would make questions about improvement easier
#    to reproduce and would explicitly flag unmatched names and missing years.
