# ─────────────────────────────────────────────
# This is your first unit test file.
# pytest looks for files named test_*.py
# and functions named test_*() inside them.
# ─────────────────────────────────────────────

import pytest

def test_project_imports():
    """Test that core libraries are installed and importable."""
    import pandas as pd
    import numpy as np
    import sklearn
    assert True   # If we got here, imports worked

def test_pandas_basic():
    """Test basic pandas functionality works."""
    import pandas as pd
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    assert len(df) == 3
    assert list(df.columns) == ['a', 'b']

def test_numpy_basic():
    """Test basic numpy functionality works."""
    import numpy as np
    arr = np.array([1, 2, 3, 4, 5])
    assert arr.mean() == 3.0
    assert arr.sum() == 15
