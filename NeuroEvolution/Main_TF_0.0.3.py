'''
Neuro Evolution Algorithm by Michel Aka author of FitML github blog and repository
https://github.com/FitMachineLearning/FitML/
See the agents in action at
https://www.youtube.com/channel/UCi7_WxajoowBl4_9P0DhzzA/featured
'''


import numpy as np
import keras
import gym
from random import gauss
import roboschool
import math


from random import randint
import tensorflow as tf

from Lib.Individual import IndividualTF
'''
ENVIRONMENT_NAME = "RoboschoolAnt-v1"
OBSERVATION_SPACE = 28
ACTION_SPACE = 8
'''

ENVIRONMENT_NAME = "LunarLanderContinuous-v2"
OBSERVATION_SPACE = 8
ACTION_SPACE = 2

B_DISCOUNT = 0.98

POPULATION_SIZE = 10
NETWORK_WIDTH = 2
NETWORK_HIDDEN_LAYERS = 0
NUM_TEST_EPISODES = 1
NUM_SELECTED_FOR_REPRODUCTION = 3
NOISE_SIGMA = 0.5
MUTATION_PROB = 0.85

MAX_GENERATIONS = 200000

USE_GAUSSIAN_NOISE = True
HAS_EARLY_TERMINATION_REWARD = False
EARLY_TERMINATION_REWARD = -50
CLIP_ACTIONS = True
MAX_STEPS = 950

all_individuals = []
generations_count = 0
total_population_counter = 0
#numLandings = 0





'''---------ENVIRONMENT INITIALIZATION--------'''

env = gym.make(ENVIRONMENT_NAME)
#env.render(mode="human")
env.reset()

print("-- Observations",env.observation_space)
print("-- actionspace",env.action_space)


#initialize training matrix with random states and actions
apdataX = tf.placeholder("float", [None, OBSERVATION_SPACE])
#apdataY = np.random.random((5,num_env_actions))
apdataY = tf.placeholder("float", [None, ACTION_SPACE])

sess = tf.Session()

'''---------------------'''

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def apModel(X, apw_h, apw_o):
    h = tf.nn.leaky_relu(tf.matmul(X, apw_h)) # this is a basic mlp, think 2 stacked logistic regressions
    return tf.matmul(h, apw_o) # note that we dont take the softmax at the end because our cost fn does that for us





def GetRememberedOptimalPolicy(indiv,qstate):
    predX = np.zeros(shape=(1,OBSERVATION_SPACE))
    predX[0] = qstate

    #print("trying to predict reward at qs_a", predX[0])
    #pred = action_predictor_model.predict(predX[0].reshape(1,predX.shape[1]))

    inputVal = predX[0].reshape(1,predX.shape[1])
    pred = sess.run(indiv.appy_x, feed_dict={apdataX: inputVal})
    r_remembered_optimal_policy = pred[0]
    return r_remembered_optimal_policy


def create_individualTF(network_width, network_hidden_layers, observation_space, action_space):
    ''' apModel '''
    apw_h = init_weights([OBSERVATION_SPACE, 32]) # create symbolic variables
    apw_h2 = init_weights([32, 32]) # create symbolic variables
    apw_h3 = init_weights([32, 32]) # create symbolic variable
    apw_o = init_weights([32, ACTION_SPACE])

    appy_x = apModel(apdataX, apw_h, apw_o)

    apcost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(apdataY, appy_x))))
    apOptimizer = tf.train.AdadeltaOptimizer(1.,0.9,1e-6)
    aptrain_op = apOptimizer.minimize(apcost)

    ''' end apModel '''

    sess.run(tf.global_variables_initializer())
    #action_predictor_model = create_model(network_width,network_hidden_layers, observation_space, action_space)
    indiv = IndividualTF(generationID=0, indivID=total_population_counter ,
            apw_h=apw_h,apw_h2=apw_h2,
            apw_h3=apw_h3,apw_o=apw_o,appy_x=appy_x)
    return indiv




def initialize_population(population_size,network_width,network_hidden_layers, observation_space, action_space, environment_name,total_population_counter):
    initial_population = []
    for i in range (population_size):
        print("kk", network_width,network_hidden_layers,observation_space,action_space)
        indiv = create_individualTF( network_width, network_hidden_layers, observation_space, action_space)
        total_population_counter += 1
        initial_population.append(indiv)
    return initial_population, total_population_counter

def test_individual(indiv,num_test_episodes):
    indiv.lifeScore = 0
    allRewards = []
    terminated_early = False
    for i in range(num_test_episodes):
        episodeRewards = []
        cumulativeRewards = 0
        #print("episode "+str(i)+" performing test for indiv ",indiv.printme())
        qs = env.reset()
        for step in range (5000):
            a = GetRememberedOptimalPolicy(indiv, qs)
            if CLIP_ACTIONS:
                for i in range (np.alen(a)):
                    if a[i] < -1: a[i]=-0.99999999999
                    if a[i] > 1: a[i] = 0.99999999999
            qs,r,done,info = env.step(a)
            if HAS_EARLY_TERMINATION_REWARD and done and step<MAX_STEPS-3:
                r = EARLY_TERMINATION_REWARD
                terminated_early = True

            cumulativeRewards = cumulativeRewards + r
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
                epAvg = sum(episodeRewards) / len(episodeRewards)
                allRewards.append(epAvg)
                #f epAvg >0:
                #    numLandings = numLandings+1

                break
        #print("generationID",indiv.generationID,"IndivID",indiv.indivID,"episodeRewards rewards ",epAvg)

        avg = sum(allRewards) / len(allRewards)
        indiv.lifeScore = avg
    #indiv.lifeScore = math.fabs(float(env.unwrapped.walk_target_dist) - 1001.0)
    #if terminated_early:
    #    print("Terminated early")
    #    indiv.lifeScore = math.fabs(float(env.unwrapped.walk_target_dist) - 1001.0) - ( - EARLY_TERMINATION_REWARD)
    print("generationID",indiv.generationID,"indivID - ",indiv.indivID,"numLandings ",0,"lifeScore =",indiv.lifeScore)


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


    print("Selected individuals ")
    for i in range (len(selected_individuals)):
        print(selected_individuals[i].printme())

    return selected_individuals

# --- Parameter Noising


def add_noise_simple(mu,noiseSigma, largeNoise=False):
    x =   np.random.rand(1) - 0.5 #probability of doing x
    if np.random.rand(1) < MUTATION_PROB:
        #print("mutating")
        if not largeNoise:
            x = x*noiseSigma
        else:
            x = x*noiseSigma   #Sigma = width of the standard deviaion
    else:
        x = 0
        #print ("x/200",x,"big_sigma",big_sigma)
    return mu + x

def add_gaussian_noise(mu,noiseSigma,largeNoise=False):
    #print ( gauss(mu, noiseSigma) )
    return gauss(mu, noiseSigma)

add_noise_simple = np.vectorize(add_noise_simple,otypes=[np.float])
add_gaussian_noise = np.vectorize(add_gaussian_noise,otypes=[np.float])



def add_noise_to_model_TF(indiv,noiseSigma,largeNoise = False):
    print ("ADDING NOISE TF")
    variables_names =[v.name for v in tf.trainable_variables()]
    values = sess.run(variables_names)
    #for k,v in zip(variables_names, values):
    #    if(k==naw_h.name):
    #        print(k, v)
    for k,v in zip(variables_names, values):
        if(k==indiv.naw_h.name):
            if USE_GAUSSIAN_NOISE:
                v2=add_gaussian_noise(v,noiseSigma,True)
            else:
                v2=add_noise_simple(v,noiseSigma,True)

            #v2 = v+0.001
    #print("Noise added. showing res v2",v2)
    assign_op = tf.assign(indiv.naw_h,v2)
    sess.run(assign_op)

    #for k,v in zip(variables_names, values):
    #    if(k==naw_o.name):
    #        print(k, v)
    for k,v in zip(variables_names, values):
        if(k==indiv.naw_o.name):
            if USE_GAUSSIAN_NOISE:
                v2=add_gaussian_noise(v,noiseSigma,True)
            else:
                v2=add_noise_simple(v,noiseSigma,True)
            #v2 = v+0.001
    #print("Noise added. showing res v2",v2)
    assign_op2 = tf.assign(indiv.naw_o,v2)
    sess.run(assign_op2)
    '''
    variables_names =[v.name for v in tf.trainable_variables()]
    values = sess.run(variables_names)
    for k,v in zip(variables_names, values):
        if(k==naw_h.name):
            print(k, v)
    '''
    return None


def add_noise_to_model(targetModel,noiseSigma=NOISE_SIGMA,largeNoise = True):

    sz = len(targetModel.layers)
    #if largeNoise:
    #    print("Setting Large Noise!")
    for k in range(sz):
        w = targetModel.layers[k].get_weights()
        if np.alen(w) >0 :
            #print("k==>",k)
            if USE_GAUSSIAN_NOISE:
                w[0] = add_gaussian_noise(w[0],noiseSigma,largeNoise)
            else:
                w[0] = add_noise_simple(w[0],noiseSigma,largeNoise)


        targetModel.layers[k].set_weights(w)
    return targetModel


''' MUTATIONS '''
def add_mutations(individuals,noiseSigma=NOISE_SIGMA):
    for i in range (len(individuals)):
        if i >NUM_SELECTED_FOR_REPRODUCTION :
            individuals[i].network = add_noise_to_model_TF(individuals[i],noiseSigma,True)



def randomSelect(m1,m2):
    if np.random.rand(1) > 0.5 :
        return m1+0.0
    else:
        return m2+0.0

randomSelect = np.vectorize(randomSelect,otypes=[np.float])


def populate_next_generation(generationID,top_individuals,population_size, network_width,network_hidden_layers, observation_space, action_space,total_population_counter):
    newPop = top_individuals
    num_selected = len(top_individuals)
    for i in range( population_size - len(top_individuals)):
        indiv = create_individualTF(network_width, network_hidden_layers, observation_space, action_space)
        indiv1 = top_individuals[0]
        indiv2 = top_individuals[1]

        indiv.apw_h = randomSelect(indiv1.apw_h,indiv2.apw_h)

        top_individuals.append( indiv )
        total_population_counter+=1
    return top_individuals,total_population_counter




''' ------------------'''

all_individuals,total_population_counter = initialize_population(population_size=POPULATION_SIZE,
    network_width=NETWORK_WIDTH,
    network_hidden_layers = NETWORK_HIDDEN_LAYERS,
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
        POPULATION_SIZE,NETWORK_WIDTH, NETWORK_HIDDEN_LAYERS,
        OBSERVATION_SPACE,
        ACTION_SPACE,
        total_population_counter)
    for i in range (len(all_individuals)):
        all_individuals[i].printNetwork()
    #print("@@@@ Adding Noise @@@@")
    add_mutations(all_individuals)



#for i in range (len(all_individuals)):
#    all_individuals[i].printNetwork()
