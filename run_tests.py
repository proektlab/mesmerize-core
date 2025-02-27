import pytest
import os

os.environ["MESMERIZE_KEEP_TEST_DATA"] = "1"

pytest.main(["-s", "."])
