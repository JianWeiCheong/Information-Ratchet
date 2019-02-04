#!/bin/bash
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

from tqdm import tqdm
from finitestate import Tape, Ratchet

# Initialisation
k_B = 1
T = 1

tapeLength = 5000
bias = 0.9
ps = np.linspace(1E-8, 1 - 1E-8, 50)
qs = np.linspace(1E-8, 1 - 1E-8, 50)


def main():

    # Create Tape and generate sequence
    inputTape = Tape(tapeLength)
    inputTape.generateSequenceBias(bias)

    aveWorks = []
    infos = []

    # Looping through ps and qs
    for p in tqdm(ps):
        for q in qs:

            # Define transition matrix
            transMat = np.matrix([[0, 1 - p, 0, 0], [1, 0, q, 0], [0, p, 0, 1],
                                  [0, 0, 1 - q, 0]])

            # Create Ratchet
            ratchet = Ratchet(transMat)

            # Start single transition run
            outputTape, inStates = ratchet.startFixedTransitionRun(inputTape, 1)

            # Calculate Entropy Rate
            inputTape.calcEntropyRate()
            outputTape.calcEntropyRate()

            info = k_B * T * (outputTape.entropyRate - inputTape.entropyRate)

            # Calculate Work
            totalWork = ratchet.calcWork(inStates, outputTape.markovChain)
            aveWork = totalWork / inputTape.length

            aveWorks.append(aveWork)
            infos.append(info)

    # Plot regime spectrum
    plotRegimeSpectrum(aveWorks, infos)


def plotRegimeSpectrum(aveWorks, infos):

    # Converting to numpy arrays and reshaping
    infos = np.array(infos)
    aveWorks = np.array(aveWorks)

    infos = infos.reshape(len(ps), len(qs))
    aveWorks = aveWorks.reshape(len(ps), len(qs))

    # Plotting with different colours for different regimes
    plt.figure()

    for i in tqdm(range(len(ps))):
        for j in range(len(qs)):
            if (infos[i][j] >= aveWorks[i][j]) and (aveWorks[i][j] > 0):
                colour = "red"
            elif (0 > infos[i][j]) and (infos[i][j] >= aveWorks[i][j]):
                colour = "blue"
            elif (infos[i][j] >= 0) and (0 >= aveWorks[i][j]):
                colour = "green"
            else:
                colour = "black"

            plt.plot(j / len(qs), i / len(ps), '.', color=colour)

    plt.xlabel('q')
    plt.ylabel('p')
    plt.title('Numerical Solution')

    plt.savefig('numericalspectrum.png')


if __name__ == '__main__':
    main()
