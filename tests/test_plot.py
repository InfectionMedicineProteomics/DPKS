#!/usr/bin/env python3
"""Tests for the plot module"""

import pytest
from dpks.plot import Plot


@pytest.mark.plot()
def test_plot(plot_object):
    """test the plot object"""

    assert isinstance(plot_object, Plot)
