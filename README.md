# Setting up to run scripts using the openai API with Python.

The script will be run in a virtual environment. Start by creating a virtual environment:

On a Mac:
`python3 -m venv venv`

On Windows:
`python -m venv venv`

<br>
After creating the virtual environment, you need to activate it:

On a Mac:
`source venv/bin/activate`

On Windows:
`source venv/Scripts/activate`

<br>
Once the virtual environment is activated, the beginning of your terminal prompt should display (venv).

<br>
Install the necessary modules by running:

On a Mac:
`pip3 install -r requirements.txt`

On Windows:
`pip install -r requirements.txt`

<br>
To run your code, in the command line run:

Without a debugger:
`flask run`

With a debugger:
`flask run --debug`

<br>
The app will run at: http://127.0.0.1:5000/

<br>
To stop the run, click control + C.
Then hard refresh the page. When making changes to your Python, HTML, or JavaScript code (and not using debugger) you'll need to stop the run after each change.

<br>
When finished, quit the run by clicking control + C and close the virtual environment by running:

`deactivate`
