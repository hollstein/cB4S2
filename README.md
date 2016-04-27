# cb4S2 - classical Bayesian for Sentinel-2

**The paper was just submitted and therefore I created this repo. It won't work for you now - please come back after ESA Living Planet Symposium, by then it should work.**

This software is intended to be used for the classification of clouds, cirrus, shadow, water, and clear sky pixels in [Sentinel-2 MSI](https://sentinel.esa.int/web/sentinel/missions/sentinel-2) images. The software is in its infancy, but should be ready to use. The used approach is described in a paper submitted to MDPI remote sensing:

*Ready-To-use Methods for the Detection of Clouds, Cirrus, Snow, Shadow, Water and Clear Sky Pixels in Sentinel-2 MSI Images, remote sensing, André Hollstein, Karl Segl, Luis Guanter, Maximilian Brell, Marta Enesco, submitted, 4/2016*

It is highly likely that the software doesn't work on your system since only limited testing on other platforms than my desktop was done so far. If you have problems, [let me know](http://www.gfz-potsdam.de/en/section/remote-sensing/staff/profil/andre-hollstein/). 

# Install using pip

This software requires at least **python 3.5** and can be installed trough pip:

`pip install git+git://github.com/hollstein/cB4S2.git`

If no install is wanted, a simple clone should work too:

`git clone https://github.com/hollstein/cB4S2.git/`

If you don't have a python distribution already, I recommend to use [Anaconda](https://www.continuum.io/downloads).

If requested, I will try to compile binaries (not before end of May '16, though).

# Licence

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">Classical Bayesian for Sentinel 2</span> by <span xmlns:cc="http://creativecommons.org/ns#" property="cc:attributionName">André Hollstein</span> is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.<br />Based on a work at <a xmlns:dct="http://purl.org/dc/terms/" href="https://github.com/hollstein/cB4S2" rel="dct:source">https://github.com/hollstein/cB4S2</a>.

# Usage

## Command Line

Command line tool with some configurable parameters:

`cB4S2.py -h`

![screen shot of the command line](https://github.com/hollstein/images/blob/master/cB4S2_cmdl.jpg)

## Graphical User Interface

If called without parameters, a little GUI is shown:

![screen shot of the GUI](https://github.com/hollstein/images/blob/master/gui.jpg)

# Products / Results

Some early results:

![result 1](https://github.com/hollstein/images/blob/master/res_1.jpg)
![result 1](https://github.com/hollstein/images/blob/master/res_2.jpg)
![result 1](https://github.com/hollstein/images/blob/master/res_3.jpg)
