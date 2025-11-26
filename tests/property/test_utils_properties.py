import numpy as np
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays

from clinical_survival.utils import get_event_time_surv_array

# Define a Hypothesis strategy for generating pairs of event and time arrays.
# - Events are either 0 or 1.
# - Times are positive floats.
# - Both arrays will have the same length, between 1 and 100 elements.
input_arrays_strategy = st.integers(min_value=1, max_value=100).flatmap(
    lambda n: st.tuples(
        arrays(np.int8, n, elements=st.integers(min_value=0, max_value=1)),
        arrays(np.float64, n, elements=st.floats(min_value=0.1, max_value=1000)),
    )
)


@given(data=input_arrays_strategy)
def test_get_event_time_surv_array_properties(data):
    """
    Tests properties of the get_event_time_surv_array function using Hypothesis.
    """
    event_arr, time_arr = data

    # Act
    surv_arr = get_event_time_surv_array(event_arr, time_arr)

    # --- Assert Properties ---

    # 1. The output is a numpy array.
    assert isinstance(surv_arr, np.ndarray)

    # 2. The output array has the same length as the input arrays.
    assert len(surv_arr) == len(event_arr)

    # 3. The dtype of the structured array is correct.
    assert surv_arr.dtype.names == ('event', 'time')
    assert surv_arr.dtype['event'] == np.bool_
    assert surv_arr.dtype['time'] == np.float64

    # 4. The event values are correctly converted to booleans.
    expected_events = (event_arr == 1)
    np.testing.assert_array_equal(surv_arr['event'], expected_events)

    # 5. The time values are preserved.
    np.testing.assert_array_equal(surv_arr['time'], time_arr)


