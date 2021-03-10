# Environement virtuel python

## pipenv

### introduction

Pipenv is a dependency manager for Python projects. If you’re familiar with Node.js’ npm or Ruby’s bundler, it is similar in spirit to those tools. While pip can install Python packages, Pipenv is recommended as it’s a higher-level tool that simplifies dependency management for common use cases.

### installation

    pip install --user pipenv
    
    

Note

This does a user installation to prevent breaking any system-wide packages. If pipenv isn’t available in your shell after installation, you’ll need to add the user base’s binary directory to your PATH.

On Linux and macOS you can find the user base binary directory by running python -m site --user-base and adding bin to the end. For example, this will typically print ~/.local (with ~ expanded to the absolute path to your home directory) so you’ll need to add ~/.local/bin to your PATH. You can set your PATH permanently by modifying ~/.profile.

On Windows you can find the user base binary directory by running py -m site --user-site and replacing site-packages with Scripts. For example, this could return C:\Users\Username\AppData\Roaming\Python36\site-packages so you would need to set your PATH to include C:\Users\Username\AppData\Roaming\Python36\Scripts. You can set your user PATH permanently in the Control Panel. You may need to log out for the PATH changes to take effect.

### Installing packages for your project

pipenv manages dependencies on a per-project basis. To install packages, change into your project’s directory (or just an empty directory for this tutorial) and run:

    $ cd project_folder
    $ pipenv install requests

exemple of ouput

    Creating a Pipfile for this project...
    Creating a virtualenv for this project...
    Using base prefix '/usr/local/Cellar/python3/3.6.2/Frameworks/Python.framework/Versions/3.6'
    New python executable in ~/.local/share/virtualenvs/tmp-agwWamBd/bin/python3.6
    Also creating executable in ~/.local/share/virtualenvs/tmp-agwWamBd/bin/python
    Installing setuptools, pip, wheel...done.
    
    Virtualenv location: ~/.local/share/virtualenvs/tmp-agwWamBd
    Installing requests...

    Collecting requests
      Using cached requests-2.18.4-py2.py3-none-any.whl
    Collecting idna<2.7,>=2.5 (from requests)
      Using cached idna-2.6-py2.py3-none-any.whl
    Collecting urllib3<1.23,>=1.21.1 (from requests)
      Using cached urllib3-1.22-py2.py3-none-any.whl
    Collecting chardet<3.1.0,>=3.0.2 (from requests)
      Using cached chardet-3.0.4-py2.py3-none-any.whl
    Collecting certifi>=2017.4.17 (from requests)
      Using cached certifi-2017.7.27.1-py2.py3-none-any.whl
    Installing collected packages: idna, urllib3, chardet, certifi, requests
    Successfully installed certifi-2017.7.27.1 chardet-3.0.4 idna-2.6 requests-2.18.4 urllib3-1.22
    
    Adding requests to Pipfile's [packages]...
    P.S. You have excellent taste!
    
### utilisation des librairies

    $ pipenv run python main.py

## Lower level: virtualenv

### introduction

virtualenv is a tool to create isolated Python environments. virtualenv creates a folder which contains all the necessary executables to use the packages that a Python project would need.

It can be used standalone, in place of Pipenv.

### installation

    pip install virtualenv

### Basic Usage

Create a virtual environment for a project:
    
    cd project_folder
    virtualenv venv

virtualenv venv will create a folder in the current directory which will contain the Python executable files, and a copy of the pip library which you can use to install other packages. The name of the virtual environment (in this case, it was venv) can be anything; omitting the name will place the files in the current directory instead.

You can also use the Python interpreter of your choice (like python2.7).

    virtualenv -p /usr/bin/python2.7 venv

or change the interpreter globally with an env variable in ~/.bashrc:

    export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python2.7

To begin using the virtual environment, it needs to be activated:

    source venv/bin/activate

The name of the current virtual environment will now appear on the left of the prompt (e.g. (venv)Your-Computer:project_folder UserName$) to let you know that it’s active. From now on, any package that you install using pip will be placed in the venv folder, isolated from the global Python installation.

Install packages using the pip command:

    pip install requests

If you are done working in the virtual environment for the moment, you can deactivate it:
    
    $ deactivate

This puts you back to the system’s default Python interpreter with all its installed libraries.

To delete a virtual environment, just delete its folder. (In this case, it would be rm -rf venv.)

After a while, though, you might end up with a lot of virtual environments littered across your system, and it’s possible you’ll forget their names or where they were placed.


## Other Notes

Running virtualenv with the option --no-site-packages will not include the packages that are installed globally. This can be useful for keeping the package list clean in case it needs to be accessed later. [This is the default behavior for virtualenv 1.7 and later.]

In order to keep your environment consistent, it’s a good idea to “freeze” the current state of the environment packages. To do this, run:
    
    $ pip freeze > requirements.txt

This will create a requirements.txt file, which contains a simple list of all the packages in the current environment, and their respective versions. You can see the list of installed packages without the requirements format using pip list. Later it will be easier for a different developer (or you, if you need to re-create the environment) to install the same packages using the same versions:

    $ pip install -r requirements.txt

This can help ensure consistency across installations, across deployments, and across developers.

Lastly, remember to exclude the virtual environment folder from source control by adding it to the ignore list (see Version Control Ignores).


## virtualenvwrapper

virtualenvwrapper provides a set of commands which makes working with virtual environments much more pleasant. It also places all your virtual environments in one place.

To install (make sure virtualenv is already installed):

    $ pip install virtualenvwrapper
    $ export WORKON_HOME=~/Envs
    $ source /usr/local/bin/virtualenvwrapper.sh

(Full virtualenvwrapper install instructions.)

For Windows, you can use the virtualenvwrapper-win.

To install (make sure virtualenv is already installed):

    $ pip install virtualenvwrapper-win

In Windows, the default path for WORKON_HOME is %USERPROFILE%\Envs
Basic Usage

Create a virtual environment:

    $ mkvirtualenv project_folder

This creates the project_folder folder inside ~/Envs.

Work on a virtual environment:

    $ workon project_folder

Alternatively, you can make a project, which creates the virtual environment, and also a project directory inside $WORKON_HOME, which is cd-ed into when you workon project_folder.

    $ mkproject project_folder

virtualenvwrapper provides tab-completion on environment names. It really helps when you have a lot of environments and have trouble remembering their names.

workon also deactivates whatever environment you are currently in, so you can quickly switch between environments.

Deactivating is still the same:
    
    $ deactivate

To delete:

    $ rmvirtualenv venv

Other useful commands

lsvirtualenv
    List all of the environments.
cdvirtualenv
    Navigate into the directory of the currently activated virtual environment, so you can browse its site-packages, for example.
cdsitepackages
    Like the above, but directly into site-packages directory.
lssitepackages
    Shows contents of site-packages directory.

Full list of virtualenvwrapper commands.
virtualenv-burrito

With virtualenv-burrito, you can have a working virtualenv + virtualenvwrapper environment in a single command.
direnv

When you cd into a directory containing a .env, direnv automagically activates the environment.

Install it on Mac OS X using brew:
    
    $ brew install direnv

On Linux follow the instructions at direnv.net <https://direnv.net>

