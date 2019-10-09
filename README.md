# DD2421-Machine-Learning
Create a virtual environment by running 
`python3.7 -m venv ./env` inside the Machine Learning directory.
To activate the virtual environemnt, run `source env/bin/activate`.
Any packages should be installed in this environment. 
Run `pip install -r requirements.txt` to install the necessary packages.

To run jupyter notebook inside a virtual environment:
1. Source the virtual environment `source your_env_name/bin/activate`.
2. Run `pip install jupyter`.
3. Add a new kernel to jupyter by running `ipython kernel install --user --name=your_env_name`.
4. Run `jupyter-notebook` to start jupyter notebooks and navigate to your notebook in the web browser.
5. When running lab3.ipynb, change the kernel to your_env_name before executing anything.
