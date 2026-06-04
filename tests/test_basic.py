import pytest

def test_project_imports():
    import pandas as pd
    import numpy as np
    import sklearn
    assert True

def test_pandas_basic():
    import pandas as pd
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    assert len(df) == 3
    assert list(df.columns) == ['a', 'b']

def test_numpy_basic():
    import numpy as np
    arr = np.array([1, 2, 3, 4, 5])
    assert arr.mean() == 3.0
    assert arr.sum() == 15
