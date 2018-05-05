#Resources used https://github.com/LinuxIsCool/756project/blob/master/CartPole/CartPoleES.py 
# https://nathanrooy.github.io/posts/2017-11-30/evolving-simple-organisms-using-a-genetic-algorithm-and-deep-learning/
#and https://gist.github.com/wingedsheep/426e847e193d79c67e052a856d495338
# For theory on https://blog.openai.com/evolution-strategies/ and https://dzone.com/articles/beating-atari-games-with-openais-evolutionary-stra
# Implementation if fairly simplified because of keras, I have modeified most of the code and kept it as simple as possible
from keras.models import Sequential
from keras import optimizers
from keras.layers.core import Dense, Activation
from keras.layers.advanced_activations import LeakyReLU
import random
import numpy as np
import gym
from gym import wrappers




class GeneticEvolution:
    def __init__(self, inputs, outputs, nr_rounds_per_epoch, env, steps, epochs, scoreTarget):
        self.input_size = inputs
        self.output_size = outputs
        self.nr_rounds_per_epoch = nr_rounds_per_epoch
        self.env = env
        self.steps = steps
        self.epochs = epochs
        self.scoreTarget = scoreTarget
   
    def initAgents(self, nr_agents, hiddenLayers):
        self.nr_agents = nr_agents
        self.agents = [None] * nr_agents
        self.hiddenLayers = hiddenLayers

        for i in range(self.nr_agents):
            agent = Agent()
            model = self.create(self.input_size, self.output_size, hiddenLayers, "relu")
            agent.setNetwork(model)
            self.agents[i] = agent

    def create(self, inputs, outputs, hiddenLayers, activationType):
        model = Sequential()
        if len(hiddenLayers) == 0: 
            model.add(Dense(self.output_size, input_shape=(self.input_size,), init='lecun_uniform'))
            model.add(Activation("linear"))
        else:
            model.add(Dense(hiddenLayers[0], input_shape=(self.input_size,), init='lecun_uniform'))
            if (activationType == "LeakyReLU") :
                model.add(LeakyReLU(alpha=0.01))
            else:
                model.add(Activation('relu'))
            
            for index in range(1, len(hiddenLayers)):
                layerSize = hiddenLayers[index]
                model.add(Dense(layerSize, init='lecun_uniform'))
                if (activationType == "LeakyReLU") :
                    model.add(LeakyReLU(alpha=0.01))
                else:
                    model.add(Activation('relu'))
            model.add(Dense(self.output_size, init='lecun_uniform'))
            model.add(Activation("linear"))
        optimizer = optimizers.Adam(lr=1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss="mse", optimizer=optimizer)
        # model.summary()
        return model
    def crossover(self, agent1, agent2):
        weightLayerList = []
        for i in range(len(self.hiddenLayers)):
            weightLayerList.append(i * 2)
        
        new_weights = agent1.network.get_weights()
        agent2_weights = agent2.network.get_weights()
        for i in weightLayerList:
            layer2 = agent2_weights[i]
            for j in range(len(layer2)):
                rand = random.random()
                if (rand < 0.5):
                    new_weights[i][j] = layer2[j]
        child = Agent()
        newNetwork = self.create(self.input_size, self.output_size, self.hiddenLayers, "c")
        newNetwork.set_weights(new_weights)
        child.setNetwork(newNetwork)
        return child
    
    def getMaxValue(self, values):
        return np.max(values)


    # select the action with the highest value
    def selectAction(self, qValues):
        action = np.argmax(qValues)
        return action


    def simulate(self, env, agent, steps, render = False):
        """
        Simply run the simulation and return the total reward
        """
        observation = env.reset()
        totalReward = 0
        for t in range(steps):
            if (render):
                env.render()
            #Using our ANN to predict the action to be selected depending on the input state
            values=agent.network.predict(observation.reshape(1,len(observation)))[0]
            action = np.argmax(values)#changes
            newObservation, reward, done, info = env.step(action)
            totalReward += reward
            observation = newObservation
            if done:
                break
        env.close()
        
        return totalReward



    
        
    def mutate(self, agent1):
        new_weights = agent1.network.get_weights()
        weightLayerList = []
        for i in range(len(self.hiddenLayers)):
            weightLayerList.append(i * 2)
        
        for i in weightLayerList:
            layer = new_weights[i]
            for j in range(len(layer)):
                neuronConnectionGroup = layer[j]
                for k in range(len(neuronConnectionGroup)):
                    weight = neuronConnectionGroup[k]
                    rand = random.random()
                    if (rand < 0.1):
                        rand2 = (random.random() - 0.5) * 0.1
                        new_weights[i][j][k] = weight + rand2
        agent1.network.set_weights(new_weights)

    def selectBest(self):
        self.agents.sort(key=lambda x: x.fitness, reverse=True)
        selectionNr = int(self.nr_agents / 2)
        selectedAgents = self.agents[:selectionNr]
        return selectedAgents

    def createNewPopulation(self, bestAgents):
        print ("create new pop")
        newPopulation = bestAgents
        while len(newPopulation) < self.nr_agents:
            rand = random.random()
            #Probability values are pretty arbit here and that is what we want
            if rand < 0.5:
                parents = random.sample(bestAgents, 2) 
                child = self.crossover(parents[0], parents[1])
        
            elif rand < 0.7:
                parent = random.sample(bestAgents, 1)[0]
                child = Agent()
                newNetwork = self.create(self.input_size, self.output_size, self.hiddenLayers, "relu")
                newNetwork.set_weights(parent.network.get_weights())
                child.setNetwork(newNetwork)
                self.mutate(child)
            else:
                parents = random.sample(bestAgents, 2) 
                child = self.crossover(parents[0], parents[1])
                self.mutate(child)
            newPopulation.append(child)
        self.agents = newPopulation
        print('new population created')

    def tryAgent(self, agent, nr_episodes):
        total = 0
        for i in range(nr_episodes):
            total += self.simulate(self.env, agent, self.steps)
        return total / nr_episodes

    def evolve(self):
        """
        heart of the code, first compute the fitness of our current model,Terminate if satisfactory
        
        else kill all those who are in the bottom half of fitness.And repopulate the population
        """
        for e in range(self.epochs):
            self.calculateFitness()
            averageFitness = self.calculateAverageFitness()
            print ("Epoch",e,"average fitness: ",averageFitness)
            bestAgents = self.selectBest()
            bestAgentAverage = self.tryAgent(bestAgents[0] , 100)
            print(bestAgentAverage)
            if bestAgentAverage >= self.scoreTarget:
                print("bestAgentAverage: "+str(bestAgentAverage)+" >= self.scoreTarget "+str(self.scoreTarget))
                self.simulate(self.env, bestAgents[0] , self.steps*5, render = True)
                print("Finishing")

                break
            else:
                print ("Best agent average: ",bestAgentAverage)
            bestAgents[0].setFitness(bestAgentAverage)
            self.simulate(self.env, bestAgents[0] , self.steps, render = True)
            if(bestAgentAverage<averageFitness):
                self.mutate2(bestAgents[0])
            self.createNewPopulation(bestAgents)

    def calculateAverageFitness(self):
        """
        Simply iterate over th agents and return the average fitness va
        """
        total = 0
        count = 0
        for index, agent in enumerate(self.agents):
            total += agent.fitness
            count += 1
        return total / count

    def calculateFitness(self):
        """
        run a number of simulations (nr_rounds_per_epoch) for each agent and determine their fitness (average score)
        by simply running the simulation.
        Learning occurs here
        """        
        agentScores = [0] * self.nr_agents
        for r in range(self.nr_rounds_per_epoch):
            for a in range(self.nr_agents):
                agent = self.agents[a]
                score = self.simulate(self.env, agent , self.steps)
                agentScores[a] += score

        for index, value in enumerate(agentScores):
            value /= self.nr_rounds_per_epoch
            self.agents[index].setFitness(value)

class Agent:
    """
    Each agent represents an artificial neural network.
    """
    def __init__(self, network=None, fitness=None):
        self.network = network
        self.fitness = fitness

    def setNetwork(self, network):
        self.network = network

    def setFitness(self, fitness):
        self.fitness = fitness
env = gym.make('CartPole-v1')
#env=gym.make('Assault-ram-v0')

#env=gym.make('Acrobot-v1')
env=wrappers.Monitor(env,'Video5')
#env.close()
epochs = 10
steps = 800

scoreTarget = 500
# inputs, outputs, nr_rounds to determine fitness, env, max steps per episode, nr epochs, score target
evolution = GeneticEvolution(len(env.observation_space.high), env.action_space.n, 5, env, steps, epochs, scoreTarget)
# number of agents and hidden layer sizes in array
evolution.initAgents(100, [30, 30])#30 30 works for cartpole
evolution.evolve()
