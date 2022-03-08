#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

import dpks

extensions = [
    "sphinx.ext.autodoc",
    "sphinx_copybutton",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx_click.ext",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.autoyaml",
]

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"

# General information about the project.
project = "dpks"
copyright = "2022, Aaron Scott"
author = "Aaron Scott"

version = dpks.__version__
release = dpks.__version__

language = None
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
pygments_style = "sphinx"
todo_include_todos = False


html_theme = "sphinx_material"
html_title = "DPKS"
html_theme_options = {
    "nav_title": "DPKS",
    "color_primary": "blue",
    "color_accent": "light-blue",
    "repo_url": "https://github.com/InfectionMedicineProteomics/DPKS",
    "repo_name": "DPKS",
    "globaltoc_depth": 2,
    "globaltoc_collapse": True,
    "globaltoc_includehidden": False,
}
html_sidebars = {
    "**": ["logo-text.html", "globaltoc.html", "localtoc.html", "searchbox.html"]
}


# html_static_path = ["_static"]
htmlhelp_basename = "dpksdoc"
latex_elements = {}

latex_documents = [
    (master_doc, "dpks.tex", "dpks Documentation", "Aaron Scott", "manual"),
]

man_pages = [(master_doc, "dpks", "dpks Documentation", [author], 1)]

texinfo_documents = [
    (
        master_doc,
        "dpks",
        "dpks Documentation",
        author,
        "dpks",
        "One line description of project.",
        "Miscellaneous",
    ),
]
