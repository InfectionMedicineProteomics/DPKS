#!/usr/bin/env python

"""Tests for `dpks` package."""

import pytest

from typer.testing import CliRunner

from dpks import dpks  # noqa: F401
from dpks import cli


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(
        cli.app
    )
    assert result.exit_code in (0, 2)

    help_result = runner.invoke(cli.app, ["--help"])
    assert help_result.exit_code == 0

