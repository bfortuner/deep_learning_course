# Deep Learning Part 1
Materials for Deep Learning Certificate created by [fast.ai](http://www.fast.ai/2016/10/08/curriculum)

## Links

* [Blog](http://www.fast.ai/)
* [Curriculum](http://www.fast.ai/2016/10/08/curriculum/)

## Server

We recommended provisioning an external server provided by AWS for running these notebooks. After you create an AWS account, you can run the setup scripts locally (located in the aws directory) to create new instances. From there you can ssh into your host and begin work!

## Install

1. Ensure you have [Anaconda](https://www.continuum.io/downloads) installed on your host
2. Run `pip install -r requirements.txt` to pick up remaining modules

## Running Notebooks

* To open a notebook in your browser, run `jupyter notebook` in the same directory or directory above your notebook
* To run a notebook on the command line see [nbconvert](http://nbconvert.readthedocs.io/en/latest/execute_api.html) and the config options [nbconvert config](http://nbconvert.readthedocs.io/en/latest/config_options.html)
```
  jupyter nbconvert \
    --to notebook --execute redux.ipynb \
    --output output_redux.ipynb \
    --ExecutePreprocessor.timeout=-1 \
    --Application.log_level="DEBUG"
```
	  
