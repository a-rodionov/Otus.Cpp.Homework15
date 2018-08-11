import argparse
import matplotlib
import numpy
import matplotlib.pyplot  
import os

parser = argparse.ArgumentParser()
parser.add_argument('filename', help='csv file with 3 columns')
args = parser.parse_args()

path, file = os.path.split(args.filename)
filename, file_extension = os.path.splitext(file)
outFilename = path + '/' +filename + '.png'

data = numpy.genfromtxt(args.filename, delimiter=';', names=['x', 'y', 'z'])

matplotlib.pyplot.scatter(data['x'], data['y'], c=data['z'], lw = 0.7, s = 50)
matplotlib.pyplot.show()
