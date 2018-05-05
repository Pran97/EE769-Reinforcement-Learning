#Resources used- https://github.com/lazyprogrammer/machine_learning_examples/blob/master/rl2/cartpole/q_learning.py
#I have taken the body of the code from here modified and implemented a custom back propoagation(this is the brain of the code) in order to implement TD gammon with state action
#Value function Q(s,a) (Note in the paper they had implemented it on state value function V(s) I have made a modification here)
#
import gym
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
import tensorflow as tf
#Visualising the rewards after training is complete
def plot_running_avg(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()
#Brain of our AI
class TemporalDifference:
    #custom back propogation algorithm
    def __init__(self, D):
        lr = 0.1
        lb=0.1
        # crlbeate inputs, targets, params
        # matmul doesn't like when w is 1-D
        # so we make it 2-D and then flatten the prediction
        self.w = tf.Variable(tf.random_normal(shape=(D, 1)), name='w')
        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.Y = tf.placeholder(tf.float32, shape=(None,), name='Y')
        old_grad=[0]*D
        self.et = tf.Variable(tf.zeros(shape=(D, 1)), name='w')
        # make prediction and cost
        Y_hat = tf.reshape( tf.matmul(self.X, self.w), [-1] )
        delta = self.Y - Y_hat
        cost = tf.reduce_sum(delta * delta)
        grad_w  = tf.gradients(xs=[self.w], ys=cost)
        self.et=tf.add(lb*self.et,tf.divide(grad_w,delta))
        #Custom back propogation as stated in the paper
        new_w = self.w.assign(self.w- tf.reshape(tf.convert_to_tensor([lr*x for x in grad_w],dtype=tf.float32),[D,1])-0.9*tf.reshape(tf.convert_to_tensor(old_grad,dtype=tf.float32),[D,1]))
        old_grad=grad_w
        self.w=new_w
        # ops we want to call later
        #self.train_op = tf.train.GradientDescentOptimizer(lr).minimize(cost)
        self.predict_op = Y_hat
        
        # start the session and initialize params
        init = tf.global_variables_initializer()
        self.session = tf.InteractiveSession()
        self.session.run(init)
    def train(self, X, Y):
        self.session.run(self.w, feed_dict={self.X: X, self.Y: Y})
    def predict(self, X):
        return self.session.run(self.predict_op, feed_dict={self.X: X})
class FeatureTransformer:
    def __init__(self, env):
        observation_examples = np.random.random((20000, 4))*2 - 1
        scaler = StandardScaler()
        scaler.fit(observation_examples)
        l=[]
        for i in range(4):
            l.append((str(i),RBFSampler(gamma=np.random.rand(),n_components=1000)))
        # Used to converte a state to a featurizes represenation.
        # We use RBF kernels with different variances to cover different parts of the space
        featurizer = FeatureUnion(l)
        feature_examples = featurizer.fit_transform(scaler.transform(observation_examples))
        
        self.dimensions = feature_examples.shape[1]
        self.scaler = scaler
        self.featurizer = featurizer

    def transform(self, observations):
        scaled = self.scaler.transform(observations)
        return self.featurizer.transform(scaled)
    
class Model:
    def __init__(self, env, feature_transformer):
        self.env = env
        self.models = []
        self.feature_transformer = feature_transformer
        for i in range(env.action_space.n):
            #model = SGDRegressor(feature_transformer.dimensions) #This is a linear approximation also works very well
            model=TemporalDifference(feature_transformer.dimensions)
            self.models.append(model)

    def predict(self, s):
        X = self.feature_transformer.transform(np.atleast_2d(s))
        result = np.stack([m.predict(X) for m in self.models]).T
        return result
    def update(self, s, a, G):
        X = self.feature_transformer.transform(np.atleast_2d(s))
        self.models[a].train(X, [G])
    
    def sample_action(self, s, eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))
class SGDRegressor:
    def __init__(self, D):
        self.w = np.random.randn(D) / np.sqrt(D)
        self.lr = 0.1
    def train(self, X, Y):
        self.w += self.lr*(Y - X.dot(self.w)).dot(X)
    def predict(self, X):
        return X.dot(self.w)
def play_one(env, model, eps, gamma):
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0
    while not done and iters < 2000:
        # if we reach 2000, just quit, don't want this going forever
        # the 200 limit seems a bit early
        action = model.sample_action(observation, eps)
        prev_observation = observation
        observation, reward, done, info = env.step(action)
        next = model.predict(observation)
        # print(next.shape)
        assert(next.shape == (1, env.action_space.n))
        G = reward + gamma*np.max(next)
        model.update(prev_observation, action, G)
        
        totalreward += reward
        iters += 1
    
    return totalreward
if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    #env=wrappers.Monitor(env,'RL5Cartpole') # uncomment to save videos
    ft = FeatureTransformer(env)
    model = Model(env, ft)
    gamma = 0.99
    
    
    N = 500
    totalrewards = np.empty(N)
    for n in range(N):
        eps = 1.0/np.sqrt(n+1)
        totalreward = play_one(env, model, eps, gamma)
        totalrewards[n] = totalreward
        if n % 100 == 0:
            print("episode:", n, "total reward:", totalreward, "eps:", eps, "avg reward (last 100):", totalrewards[max(0, n-100):(n+1)].mean())

    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
    print("total steps:", totalrewards.sum())
    
    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()
    
    plot_running_avg(totalrewards)
