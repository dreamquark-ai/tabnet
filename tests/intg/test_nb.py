import nbformat
import nbclient
import pytest


@pytest.mark.parametrize(
    "nb_name",
    [
        'census_example.ipynb',
        'customizing_example.ipynb',
        'forest_example.ipynb',
        'pretraining_example.ipynb',
        'regression_example.ipynb',
        'multi_task_example.ipynb',
        'multi_regression_example.ipynb',
    ]
)
def test_nb_run(nb_name, context_path):
    notebook_path = context_path / nb_name
    nb = nbformat.read(notebook_path, as_version=4)
    client = nbclient.NotebookClient(nb, )
    client.execute()
