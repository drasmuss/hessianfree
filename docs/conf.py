import sphinx_rtd_theme
import hessianfree as hf

# General information about the project.
project = u'hessianfree'
copyright = u'2015, Daniel Rasmussen'
author = u'Daniel Rasmussen'
version = hf.__version__
release = hf.__version__

# -- General configuration ------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx'
]

source_suffix = '.rst'
master_doc = 'index'
language = None
exclude_patterns = ['_build']
pygments_style = 'sphinx'
todo_include_todos = True
show_authors = False
autodoc_default_flags = ['members', 'undoc-members', 'show-inheritance']
autodoc_member_order = 'bysource'
intersphinx_mapping = {'numpy': ('http://docs.scipy.org/doc/numpy/', None),
                       'scipy': ('http://docs.scipy.org/doc/scipy/reference/',
                                 None)}

# -- Options for HTML output ----------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
# html_theme_options = {}

html_context = {'show_source': False}
htmlhelp_basename = 'hessianfreedoc'
html_show_sphinx = False
html_use_smartypants = True

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'hessianfree.tex', u'hessianfree Documentation',
     u'Daniel Rasmussen', 'manual'),
]

# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'hessianfree', u'hessianfree Documentation',
     [author], 1)
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'hessianfree', u'hessianfree Documentation',
     author, 'hessianfree', 'Hessian-free optimization for deep networks',
     'Miscellaneous'),
]
