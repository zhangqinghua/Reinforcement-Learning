"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(self, n_actions, n_features, learning_rate=0.1, batch_size=10, memory_size=20, reward_decay=0.9,
                 epsilon=0.9):
        # 特征数
        self.n_features = n_features
        # 动作数
        self.n_actions = n_actions
        # 学习率
        self.learning_rate = learning_rate
        # 奖励衰减
        self.gamma = reward_decay
        # 离线学习内存大小
        self.memory_size = memory_size
        # 分批学习
        self.batch_size = batch_size
        # 随机行为
        self.epsilon = epsilon

        # consist of [target_net, evaluate_net]
        self._build_net()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):

        # 输入的特征
        self.s = tf.placeholder(tf.float32, [None, self.n_features])
        # 输出的结果
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions])

        # 神经网络预测的结果
        l1 = self._add_layer(self.s, self.n_features, 10, tf.nn.sigmoid)
        self.q_eval = self._add_layer(l1, 10, self.n_actions, tf.nn.softplus)

        loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        self.train_op = tf.train.RMSPropOptimizer(1).minimize(loss)

    @staticmethod
    def _add_layer(inputs, in_size, out_size, activation_function=None):
        weights = tf.Variable(tf.zeros([in_size, out_size]))
        biases = tf.Variable(tf.zeros([1, out_size]))
        wx_plus_b = tf.matmul(inputs, weights) + biases
        if activation_function is None:
            outputs = wx_plus_b
        else:
            outputs = activation_function(wx_plus_b)
        return outputs

    def choose_action(self, s):
        # to have batch dimension when feed into tf placeholder
        s = np.array(s)[np.newaxis, :]

        # forward feed the observation and get q value for every actions
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: s})

        return self.dis_rand(actions_value[0])

    def store_transition(self, s, a, r, s_):
        transition = np.append(np.append(s, [a, r]), s_)
        if not hasattr(self, "memory"):
            self.memory = transition
        self.memory = np.vstack((self.memory, transition))

        if len(self.memory) >= self.memory_size:
            self.learn()
            self.__delattr__('memory')

    def learn(self):
        mini_batches = [self.memory[k:k + self.batch_size] for k in range(0, len(self.memory), self.batch_size)]
        for mini_batche in mini_batches:
            # 当前环境
            s = mini_batche[:, :self.n_features]
            # 作出选择后的环境
            s_ = mini_batche[:, -self.n_features:]
            # 环境反馈激励
            reward = mini_batche[:, self.n_features + 1]

            # 要更改的行
            batch_index = np.arange(len(mini_batche), dtype=np.int32)
            # 要更改的列
            eval_act_index = mini_batche[:, self.n_features].astype(int)

            # 计算当前环境的预测值
            q_eval = self.sess.run(self.q_eval, feed_dict={self.s: s})
            print("\n", q_eval)
            # 计算下一步环境的预测值
            q_target = self.sess.run(self.q_eval, feed_dict={self.s: s_})

            # 将下一步环境的激励赋值给当前环境
            q_eval[batch_index, eval_act_index] = reward + self.gamma * np.max(q_target, axis=1)

            # 反向传播学习
            self.sess.run(self.train_op, feed_dict={self.q_target: q_eval, self.s: s})

            print(s, ': \n', self.sess.run(self.q_eval, feed_dict={self.s: s}))

            return

    @staticmethod
    def dis_rand(actions_value):
        sum_value = np.sum(actions_value)
        action_probability = []
        for v in actions_value:
            pre = 0 if len(action_probability) == 0 else action_probability[-1]
            action_probability.append(pre + v / sum_value)
        random = np.random.uniform()

        for i in range(len(action_probability)):
            if random <= action_probability[i]:
                return i

    @staticmethod
    def get_action(action):
        if action == 0:
            action = 'up'
        elif action == 1:
            action = 'down'
        elif action == 2:
            action = 'right'
        else:
            action = 'left'

        return action


s = tf.placeholder(tf.float32, [None, 1])
print(s)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


print(sess.run(s, {s: [[1]]}))

s = [[1]]


