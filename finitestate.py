#!/bin/bash
# -*- coding: utf-8 -*-

import random
import numpy as np
import scipy.linalg


class Tape:
    """Tape object"""

    def __init__(self, length):
        if length < 1:
            raise ValueError("The length of the tape must be at least 1.")

        self.length = length
        self.sequence = []
        self.entropyRate = 0
        self.biasSeq = 0
        self.correlation = 0
        self.hmm = 0
        self.markovChain = 0

    def generateSequenceBias(self, bias):
        """Generate sequence as a bias coin flip."""

        if bias < 0 or bias > 1:
            raise ValueError("Bias must be a value between 0 and 1.")
        else:
            for i in range(self.length):
                self.sequence.append(0 if random.random() < bias else 1)
                self.biasSeq = 1
                self.bias = bias

    def generateSequenceIndiv(self, bit):
        if bit != 0 and bit != 1:
            raise ValueError("Bit must be either 0 or 1.")
        else:
            self.sequence.append(bit)

    def generateSequenceHMM(self, startState, transMat, transBit):
        outStates = []
        states = np.arange(0, len(startState), 1)
        newState = 0

        for i in range(self.length):
            if i == 0:
                oldState = np.random.choice(states, p=startState)
            else:
                oldState = newState

            startState = np.zeros(len(startState))
            startState[oldState] = 1

            startState = startState * transMat.T
            startState = np.squeeze(np.asarray(startState))

            newState = np.random.choice(states, p=startState)

            outStates.append(newState)

            self.generateSequenceIndiv(transBit[newState, oldState])

        self.hmm = 1
        self.correlation = 1
        self.hmmMatrix = transMat

    def calcEntropy(self):
        if self.sequence == []:
            raise TypeError(
                "Tape sequence is empty, make sure you have called a generation method."
            )

        if self.length <= 1:
            self.entropyRate = 0
            return

        if self.biasSeq == 1:
            ent = -self.bias * np.log(self.bias) / np.log(2) - (
                1 - self.bias) * np.log(1 - self.bias) / np.log(2)
            self.entropyRate = ent
            return

        counts = np.bincount(self.sequence)
        probs = counts / self.length
        n_classes = np.count_nonzero(probs)

        if n_classes <= 1:
            self.entropyRate = 0
            return

        ent = 0.
        for i in probs:
            ent -= i * np.log(i) / np.log(n_classes)

        self.entropyRate = ent
        return

    def calcEntropyRate(self):
        if self.sequence == []:
            raise TypeError("Tape sequence is empty, make sure you have called a generation method.")

        # If generated with bias
        if self.correlation == 0:
            self.calcEntropy()
            return

        # If generated with HMM
        if self.hmm == 1:
            evals, evecs = scipy.linalg.eig(self.hmmMatrix)

            vecIndex = np.argmin(abs(evals - 1.0))
            statDist = evecs[:, vecIndex].real
            statDist /= statDist.sum()

            h = 0
            for i in range(len(statDist)):
                for j in range(len(statDist)):
                    if self.hmmMatrix[j, i] == 0:
                        pass
                    else:
                        h -= statDist[i] * self.hmmMatrix[j, i] * np.log(
                            self.hmmMatrix[j, i]) / np.log(2)

            self.entropyRate = h
            return

        # Non-generated tape
        dim = max(self.markovChain) + 1
        matrix = np.zeros([dim, dim])
        normMatrix = np.zeros([dim, dim])

        for i in range(self.length - 1):
            matrix[self.markovChain[i + 1], self.markovChain[i]] += 1

        sums = matrix.sum(axis=0, keepdims=1)

        for i in range(len(matrix)):
            if sums[:, i] != 0:
                normMatrix[i] = matrix[:, i] / sums[:, i]
            else:
                normMatrix[i] = 0

        normMatrix = normMatrix.T
        evals, evecs = scipy.linalg.eig(normMatrix)

        vecIndex = np.argmin(abs(evals - 1.0))
        statDist = evecs[:, vecIndex].real
        statDist /= statDist.sum()

        h = 0
        for i in range(len(statDist)):
            for j in range(len(statDist)):
                if normMatrix[j, i] == 0:
                    pass
                else:
                    h -= statDist[i] * normMatrix[j, i] * np.log(
                        normMatrix[j, i]) / np.log(2)

        self.entropyRate = h
        return


class Ratchet:
    """Ratchet object"""

    def __init__(self, transMat):
        if transMat.shape[0] != transMat.shape[1]:
            raise ValueError("Transition matrix must be a square matrix.")

        evals, evecs = scipy.linalg.eig(transMat)

        if not 1 in evals.round(8):
            raise ValueError(
                "Transition matrix must fulfils detailed balance.")

        numStates = len(transMat) / 2

        vecIndex = np.argmin(abs(evals - 1.0))
        statDist = evecs[:, vecIndex].real
        statDist /= statDist.sum()

        self.states = np.arange(0, numStates * 2, 1)
        self.transMat = transMat
        self.statDist = statDist
        self.numStates = numStates

        self.dynamicStatesIn = []
        self.dynamicEnergiesIn = []
        self.dynamicStatesOut = []
        self.dynamicEnergiesOut = []
        self.transStates = []
        self.transEnergies = []
        self.work = 0

    def startStationaryDistributionRun(self, inTape, initRatchetState):
        outTape = Tape(inTape.length)
        inStates = []
        outStates = []

        for i in range(inTape.length):
            if i == 0:
                ratchetState = initRatchetState

            stateOld = ratchetState + inTape.sequence[i] * self.numStates
            stateNew = np.random.choice(self.states, p=self.statDist)

            inStates.append(stateOld)
            outStates.append(stateNew)
            outTape.generateSequenceIndiv(
                1 if stateNew >= self.numStates else 0)

            ratchetState = stateNew - outTape.sequence[i] * self.numStates
            outTape.markovChain = outStates

        return outTape, inStates

    def startFixedTransitionRun(self,
                                inTape,
                                initRatchetState,
                                randTrans=False,
                                numTrans=1):
        outTape = Tape(inTape.length)
        inStates = []
        outStates = []

        for i in range(inTape.length):
            if i == 0:
                ratchetState = initRatchetState

            stateOld = ratchetState + inTape.sequence[i] * self.numStates

            states = np.zeros(2 * self.numStates)
            states[stateOld] = 1

            if randTrans == True:
                numLoops = np.random.randit(numTrans[0], numTrans[1])
            else:
                numLoops = numTrans

            for j in range(numLoops):
                states = states * self.transMat.T

            states = np.squeeze(np.asarray(states))
            stateNew = np.random.choice(self.states, p=states)

            inStates.append(stateOld)
            outStates.append(stateNew)
            outTape.generateSequenceIndiv(
                1 if stateNew >= self.numStates else 0)

            ratchetState = stateNew - outTape.sequence[i] * self.numStates
            outTape.markovChain = outStates

        return outTape, inStates

    def addEnergyDynamics(self, state, energy, inOut=None):
        if inOut == 'IN':
            self.dynamicStatesIn.append(state)
            self.dynamicEnergiesIn.append(energy)
        elif inOut == 'OUT':
            self.dynamicStatesOut.append(state)
            self.dynamicEnergiesOut.append(energy)
        else:
            self.transStates.append(state)
            self.transEnergies.append(energy)

    def calcEnergyDynamics(self, inStates, outStates):

        transStates = np.array(self.transStates)

        for i in range(len(inStates)):
            if inStates[i] in self.dynamicStatesIn:
                self.work += self.dynamicEnergiesIn[self.dynamicStatesIn.index(
                    inStates[i])]
            if outStates[i] in self.dynamicStatesOut:
                self.work += self.dynamicEnergiesOut[self.dynamicStatesOut.
                                                     index(outStates[i])]

            for j in range(len(transStates)):
                if inStates[i] == transStates[j, 0] and outStates[
                        i] == transStates[j, 1]:
                    self.work += self.transEnergies[j]
                elif inStates[i] == transStates[j, 1] and outStates[
                        i] == transStates[j, 0]:
                    self.work -= self.transEnergies[j]
        return self.work
