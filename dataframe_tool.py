import pandas as pd
import duckdb
import io
import contextlib
import traceback
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st

from typing import Any, Dict, Optional
from agno.tools import Toolkit
from agno.utils.log import log_debug, log_info, logger

def safe_exec(code: str, local_vars: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Executes arbitrary Python code safely within a restricted environment.
    Captures stdout and execution errors without polluting the main environment.

    Args:
        code (str): The code string to execute.
        local_vars (dict, optional): Local variable dict to use during execution.

    Returns:
        dict: A dictionary of final local variables, including optional '_stdout' and '_error' keys.
    """
    if local_vars is None:
        local_vars = {}

    allowed_builtins = {
        'print': print,
        'len': len,
        'range': range,
        'list': list,
        'dict': dict,
        'str': str,
        'int': int,
        'float': float,
        'bool': bool,
        'True': True,
        'False': False,
        'None': None,
        '__import__': __import__
    }

    global_env = {"__builtins__": allowed_builtins}
    stdout_capture = io.StringIO()
    final_locals = local_vars.copy()

    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(code, global_env, final_locals)
        final_locals['_stdout'] = stdout_capture.getvalue()
    except Exception as e:
        final_locals['_error'] = traceback.format_exc()
        logger.error(f"Error executing code: {e}\n{traceback.format_exc()}")

    stdout_capture.close()
    return final_locals


class DataFrameTools(Toolkit):
    """
    A toolkit that enables interaction with a Pandas DataFrame using tools like:
    - Metadata inspection
    - SQL queries (via DuckDB)
    - Python code execution (including plotting)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        df_name: str = "df",
        max_rows_display: int = 10,
        allow_sql_query: bool = True,
        allow_python_exec: bool = True,
        duckdb_connection: Optional[Any] = None,
        duckdb_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name="dataframe_tools")
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input `df` must be a Pandas DataFrame.")

        self.df = df
        self.df_name = df_name
        self.max_rows_display = max_rows_display
        self.duckdb_connection = duckdb_connection
        self.duckdb_kwargs = duckdb_kwargs or {}

        self.register(self.get_dataframe_info)
        self.register(self.get_dataframe_head)
        self.register(self.get_dataframe_columns)

        if allow_sql_query:
            try:
                import duckdb
                self.register(self.query_dataframe_sql)
            except ImportError:
                logger.warning("DuckDB not installed. SQL support disabled.")
                allow_sql_query = False

        if allow_python_exec:
            try:
                import matplotlib
                import seaborn
                self.register(self.execute_python_code)
            except ImportError as e:
                logger.warning(f"Missing plotting libraries: {e}")
                self.register(self.execute_python_code)

    def _get_duckdb_connection(self):
        """Returns an existing or new DuckDB connection."""
        if self.duckdb_connection:
            return self.duckdb_connection
        try:
            return duckdb.connect(**self.duckdb_kwargs)
        except Exception as e:
            logger.error(f"Error connecting to DuckDB: {e}")
            raise ConnectionError(f"Could not connect to DuckDB: {e}")

    def get_dataframe_info(self) -> str:
        """Returns DataFrame structure including dtypes, nulls, and shape."""
        log_info(f"Getting info for DataFrame '{self.df_name}'")
        buffer = io.StringIO()
        self.df.info(buf=buffer)
        return f"Shape: {self.df.shape}\n\n{buffer.getvalue()}"

    def get_dataframe_head(self, n: int = 5) -> str:
        """Returns the first `n` rows of the DataFrame as a markdown table."""
        log_info(f"Getting head({n}) for DataFrame '{self.df_name}'")
        if not isinstance(n, int) or n <= 0:
            n = 5
        return self.df.head(n).to_markdown(index=False)

    def get_dataframe_columns(self) -> str:
        """Returns a JSON list of column names in the DataFrame."""
        log_info(f"Getting columns for DataFrame '{self.df_name}'")
        return json.dumps(self.df.columns.tolist())

    def query_dataframe_sql(self, sql_query: str) -> str:
        """
        Runs SQL on the DataFrame using DuckDB. Uses DataFrame as table `df_name`.

        Args:
            sql_query (str): SQL command.

        Returns:
            str: Query result as markdown or formatted text.
        """
        log_info(f"SQL on '{self.df_name}': {sql_query}")
        try:
            con = self._get_duckdb_connection()
            con.register(self.df_name, self.df)
            formatted_sql = sql_query.replace("`", "").strip()
            if not formatted_sql:
                return "Error: Empty SQL query provided."

            query_result = con.sql(formatted_sql)
            result_output = "Query executed successfully, but returned no results."

            try:
                result_df = query_result.fetchdf()
                if not result_df.empty:
                    if len(result_df) > self.max_rows_display:
                        result_output = f"Query Result (first {self.max_rows_display} of {len(result_df)} rows):\n"
                        result_output += result_df.head(self.max_rows_display).to_markdown(index=False)
                    else:
                        result_output = "Query Result:\n" + result_df.to_markdown(index=False)
            except AttributeError:
                results_py = query_result.fetchall()
                if results_py:
                    cols = query_result.columns
                    rows = [" | ".join(map(str, row)) for row in results_py[:self.max_rows_display]]
                    result_output = f"Query Result (first {len(rows)} rows):\n{', '.join(cols)}\n" + "\n".join(rows)
                    if len(results_py) > self.max_rows_display:
                        result_output += f"\n... ({len(results_py) - self.max_rows_display} more rows)"
            except Exception as fetch_err:
                logger.error(f"Fetch error: {fetch_err}")
                result_output = f"Fetch error: {fetch_err}"

            if not self.duckdb_connection:
                con.close()

            log_debug(f"SQL Result Preview: {result_output[:200]}...")
            return result_output

        except Exception as e:
            logger.error(f"SQL Execution Error: {e}\n{traceback.format_exc()}")
            if 'con' in locals() and not self.duckdb_connection:
                try: con.close()
                except: pass
            return f"SQL Execution Error: {e}"

    def execute_python_code(self, code: str) -> str:
        """
        Executes Python code with access to the DataFrame and plotting libs.
        Supports pandas, matplotlib, seaborn, numpy, and streamlit for plot display.

        Args:
            code (str): User-defined code to run.

        Returns:
            str: Execution result, stdout, or error message.
        """
        log_info(f"Running Python code on '{self.df_name}'")
        log_debug(f"Code:\n{code}")

        exec_locals = {
            self.df_name: self.df,
            "pd": pd,
            "np": np,
            "plt": plt,
            "sns": sns,
            "st": st,
        }

        plt.close('all')
        plt.clf()

        captured_stdout = ""
        error_output = None
        plot_displayed = False

        try:
            stdout_capture = io.StringIO()
            with contextlib.redirect_stdout(stdout_capture):
                exec(code, {"__builtins__": __builtins__}, exec_locals)
            captured_stdout = stdout_capture.getvalue()
        except Exception:
            error_output = traceback.format_exc()
            logger.error(f"Python Execution Error:\n{error_output}")
            return f"Error executing code:\n```\n{error_output}\n```"

        if plt.get_fignums():
            try:
                fig = plt.gcf()
                if fig.get_axes():
                    st.pyplot(fig)
                    plot_displayed = True
                    log_info("Displayed plot.")
            except Exception as plot_err:
                error_output = f"Plot display error: {plot_err}\n{traceback.format_exc()}"
                logger.error(error_output)

        plt.clf()
        plt.close('all')

        final_output = ""
        if 'data:image' not in captured_stdout:
            final_output += f"Code output:\n```\n{captured_stdout}\n```\n"
        else:
            final_output += 'Plot generated.'

        if plot_displayed:
            final_output += "\nPlot shown successfully."
        elif error_output:
            final_output += f"\nPlot Error:\n```\n{error_output}\n```"
        elif not captured_stdout:
            final_output = "Code ran fine, no output or plot."

        log_debug(f"Code Result Preview: {final_output[:200]}...")
        return final_output.strip()
