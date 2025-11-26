import pandas as pd
from hypothesis import given, strategies as st
from hypothesis.extra.pandas import data_frames, column

from clinical_survival.utils import stratified_event_split

EVENT_COL = "event"

@given(
    data_frames([
        column("id", dtype=int),
        column(EVENT_COL, elements=[0, 1]),
        column("age", dtype=int),
    ], index=st.integers(min_value=0, max_value=1000).map(lambda i: pd.RangeIndex(i * 20, (i + 1) * 20))),
    st.floats(min_value=0.1, max_value=0.9),
    st.integers(min_value=0, max_value=100),
)
def test_stratified_event_split_properties(df, test_size, seed):
    """
    Tests properties of the stratified_event_split function.
    """
    if len(df) < 2 or df[EVENT_COL].nunique() < 2:
        return  # Not enough data to split meaningfully

    train_df, test_df = stratified_event_split(
        df=df, event_col=EVENT_COL, test_size=test_size, seed=seed
    )

    # 1. The total number of samples should be conserved.
    assert len(train_df) + len(test_df) == len(df)

    # 2. The train and test sets should be disjoint.
    assert train_df.index.intersection(test_df.index).empty

    # 3. The test set size should be approximately correct.
    expected_test_size = int(len(df) * test_size)
    # Allow for some leeway due to integer rounding.
    assert abs(len(test_df) - expected_test_size) <= 2

    # 4. The event rate should be approximately preserved.
    original_event_rate = df[EVENT_COL].mean()
    train_event_rate = train_df[EVENT_COL].mean()
    test_event_rate = test_df[EVENT_COL].mean()

    if len(train_df) > 0 and len(test_df) > 0:
        assert abs(original_event_rate - train_event_rate) < 0.2
        assert abs(original_event_rate - test_event_rate) < 0.2











