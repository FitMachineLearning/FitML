import numpy as np
import keras
import gym
import roboschool

from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers

from Lib.Individual import Individual

ENVIRONMENT_NAME = "RoboschoolHopper-v1"
OBSERVATION_SPACE = 15
ACTION_SPACE = 3

B_DISCOUNT = 0.98

POPULATION_SIZE = 10
NETWORK_WIDTH = 512
NUM_TEST_EPISODES = 3
NUM_SELECTED_FOR_REPRODUCTION = 2
NOISE_SIGMA = 0.06

MAX_GENERATIONS = 20000

CLIP_ACTIONS = True
MAX_STEPS = 996

all_individuals = []
generations_count = 0
total_population_counter = 0





'''---------ENVIRONMENT INITIALIZATION--------'''

env = gym.make(ENVIRONMENT_NAME)
#env.render(mode="human")
env.reset()

print("-- Observations",env.observation_space)
print("-- actionspace",env.action_space)


#initialize training matrix with random states and actions
apdataX = np.random.random(( 5,OBSERVATION_SPACE ))
apdataY = np.random.random((5,ACTION_SPACE))


'''---------------------'''

def GetRememberedOptimalPolicy(targetModel,qstate):
    predX = np.zeros(shape=(1,OBSERVATION_SPACE))
    predX[0] = qstate

    #print("trying to predict reward at qs_a", predX[0])
    pred = targetModel.predict(predX[0].reshape(1,predX.shape[1]))
    r_remembered_optimal_policy = pred[0]
    return r_remembered_optimal_policy


def create_model(network_width, observation_space, action_space):
    action_predictor_model = Sequential()
    action_predictor_model.add(Dense(network_width, activation='relu', input_dim=observation_space))
    action_predictor_model.add(Dense(action_space))
    return action_predictor_model

def initialize_population(population_size,network_width, observation_space, action_space, environment_name,total_population_counter):
    initial_population = []
    for i in range (population_size):
        action_predictor_model = create_model(network_width, observation_space, action_space)
        indiv = Individual(generationID=0, indivID=total_population_counter , network = action_predictor_model)
        total_population_counter += 1
        initial_population.append(indiv)
    return initial_population, total_population_counter

def test_individual(indiv,num_test_episodes):
    indiv.lifeScore = 0
    allRewards = []
    for i in range(num_test_episodes):
        episodeRewards = []
        #print("episode "+str(i)+" performing test for indiv ",indiv.printme())
        qs = env.reset()
        for step in range (5000):
            a = GetRememberedOptimalPolicy(indiv.network, qs)
            if CLIP_ACTIONS:
                for i in range (np.alen(a)):
                    if a[i] < -1: a[i]=-0.99999999999
                    if a[i] > 1: a[i] = 0.99999999999
            qs,r,done,info = env.step(a)
            episodeRewards.append(r)
            #indiv.lifeScore += r
            env.render()
            if step > MAX_STEPS:
                done = True
            if done:
                episodeRewards.reverse()
                for j in range(len(episodeRewards)):
                    #if j ==0:
                    #    print("last reward ",episodeRewards[j])
                    if j > 0:
                        episodeRewards[j] = episodeRewards[j] + B_DISCOUNT * episodeRewards[j-1]
                #avg = sum(episodeRewards)/len(episodeRewards)
                #print("episode average ", avg)
                for j in range(len(episodeRewards)):
                    allRewards.append(episodeRewards[j])
                #allRewards = allRewards + episodeRewards
                break
        epAvg = sum(episodeRewards) / len(episodeRewards)
        print("generationID",indiv.generationID,"IndivID",indiv.indivID,"episodeRewards rewards ",epAvg)

    avg = sum(allRewards) / len(allRewards)
    indiv.lifeScore = avg
    #indiv.lifeScore = np.random.rand(1)[0]*50
    print("indivID - ",indiv.indivID,"lifeScore =",indiv.lifeScore)


def test_all_individuals(num_test_episodes):
    for i in range(len(all_individuals)):
        test_individual(all_individuals[i],NUM_TEST_EPISODES)


def select_top_individuals(num_selected,population_size):
    scores = np.zeros(population_size)
    for i in range(np.alen(scores)):
        scores[i] = all_individuals[i].lifeScore

    print( scores )
    topScores = scores[ scores.argsort()[-num_selected:][::-1] ]
    #print ("Top Scores ", topScores)
    selected_individuals = []
    for i in range(len(all_individuals)):
        if all_individuals[i].lifeScore >= topScores.min():
            #print("Selecting individual",i," with score ", all_individuals[i].lifeScore,"cuttoff ", topScores.min())
            selected_individuals.append(all_individuals[i])

    for i in range (len(selected_individuals)):
        print(selected_individuals[i].printme())

    return selected_individuals

# --- Parameter Noising
def add_noise_simple(mu,noiseSigma, largeNoise=False):
    x =   np.random.rand(1) - 0.5 #probability of doing x
    if not largeNoise:
        x = x*noiseSigma
    else:
        x = x*noiseSigma   #Sigma = width of the standard deviaion
    #print ("x/200",x,"big_sigma",big_sigma)
    return mu + x


add_noise_simple = np.vectorize(add_noise_simple,otypes=[np.float])


def add_noise_to_model(targetModel,noiseSigma=NOISE_SIGMA,largeNoise = True):

    sz = len(targetModel.layers)
    #if largeNoise:
    #    print("Setting Large Noise!")
    for k in range(sz):
        w = targetModel.layers[k].get_weights()
        if np.alen(w) >0 :
            #print("k==>",k)
            w[0] = add_noise_simple(w[0],noiseSigma,largeNoise)

        targetModel.layers[k].set_weights(w)
    return targetModel

def add_mutations(individuals,noiseSigma=NOISE_SIGMA):
    for i in range (len(individuals)):
        individuals[i].network = add_noise_to_model(individuals[i].network,noiseSigma,True)


def populate_next_generation(generationID,top_individuals,population_size, network_width, observation_space, action_space,total_population_counter):
    newPop = top_individuals
    for i in range( population_size - len(top_individuals)):
        newModel = create_model(network_width, observation_space, action_space)
        model1 = top_individuals[0].network
        model2 = top_individuals[1].network
        sz = len(newModel.layers)
        #if largeNoise:
        #    print("Setting Large Noise!")
        for k in range(sz):
            w = newModel.layers[k].get_weights()
            w1 = model1.layers[k].get_weights()
            w2 = model2.layers[k].get_weights()

            if np.alen(w) >0 :
                #print("k==>",k)
                #w[0][0] = combine_weights(w[0][0],w1[0][0],w2[0][0])
                for j in range(np.alen(w[0])):
                    y=w[0][j]
                    y1 = w1[0][j]
                    y2 = w2[0][j]
                    for l in range (np.alen(y)):
                        z=y[l]
                        z1=y1[l]
                        z2=y2[l]
                        if np.random.rand(1)>0.5:
                            z=z1+0.0
                        else:
                            z=z2+0.0
                        y[l]=z
                    w[0][j]=y

            newModel.layers[k].set_weights(w)
        top_individuals.append( Individual(generationID,total_population_counter,newModel) )
        total_population_counter+=1
    return top_individuals,total_population_counter




''' ------------------'''

all_individuals,total_population_counter = initialize_population(population_size=POPULATION_SIZE,
    network_width=NETWORK_WIDTH,
    observation_space=OBSERVATION_SPACE,
    action_space=ACTION_SPACE,
    environment_name=ENVIRONMENT_NAME,
    total_population_counter=total_population_counter)


for gens in range (MAX_GENERATIONS):
    test_all_individuals(NUM_TEST_EPISODES)
    top_individuals = select_top_individuals(NUM_SELECTED_FOR_REPRODUCTION,POPULATION_SIZE)
    generations_count += 1
    print("Generating next Gen ",generations_count)
    all_individuals,total_population_counter = populate_next_generation(generations_count,top_individuals,
        POPULATION_SIZE,NETWORK_WIDTH,
        OBSERVATION_SPACE,
        ACTION_SPACE,
        total_population_counter)
    print("@@@@ Adding Noise @@@@")
    add_mutations(all_individuals)



#for i in range (len(all_individuals)):
#    all_individuals[i].printNetwork()
