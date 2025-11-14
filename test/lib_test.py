import pandas as pd
import pandas_ta as ta

# Create a dummy dataframe
df = pd.DataFrame()

# This is the line that fails.
# Try to access the help doc for the 'bbands' method.
help(df.ta.bbands)