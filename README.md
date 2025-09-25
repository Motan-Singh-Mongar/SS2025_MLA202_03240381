Name: Motan Singh Mongar
student_number: 03240381




Reflection:

#Q.A brief summary of the tasks you completed.
I have modified given code of Frozenlake ,printed the action_space and observation_space as guided by question in exercise 1. In exercise 2 I have created outer loop to execute the code for 1000 times. I have used for loop as we know how many times the loop is going to execute. And I have created variable called num_episode to keep the track on which track is code executing. And I have removed     env.render(), print() and time.sleep() to execute the code faster. Is this statement correct


#Q.The answers to the questions from Exercise 1 about the CartPole environment's action and observation spaces.
What type of space is the action space? How many actions are there?
Ans=>It's a discrete(2): There are two possible action(0:Left, 1:Right).


#Q.What type of space is the observation space? The output is Box(4,). This represents a continuous space with 4 numbers. Based on the problem, what could these four numbers possibly represent?
Ans=>Observation Space: Box(4)
This is a continuous space with 4 numbers.
These 4 numbers represent:
                        1. Cart position (how far the cart is from the center)
                        2. Cart velocity (how fast the cart is moving left/right)
                        3. Pole angle (how tilted the pole is)
                        4. Pole angular velocity (how fast the pole is falling/rotating)


##Q.Run the random agent for one episode. What does the reward seem to represent in this environment? (Hint: you get a reward for every step the pole remains balanced).
Ans=>reward represent that every time pole is balance reward is equal to 1.

#Q.The final average reward you calculated for the random agent in Exercise 2.
Ans=>Average reward over 1000 episodes: 0.0140.
c:\Users\Ideapad Gaming\OneDrive\Pictures\Screenshots\Screenshot (34).png


#Q.A section on challenges: What was the most difficult part of this practical for you? (e.g., setting up the environment, understanding the step function's return values, structuring the loops).
Ans=>The main challenges for me was to structure the loop. I have used for loop as questions already guided us that we have to run the code for 1000 time, so we know the number of episode to run(1000 times).

#A section on key takeaways: What is the most important or surprising thing you learned?
Ans=>The most surprising thing was that how low the performance of agent is, even after executing the code for 1000 times the total_reward seems to be very low. 