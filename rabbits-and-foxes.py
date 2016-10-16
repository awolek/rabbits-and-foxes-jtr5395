
# coding: utf-8

# # Objectives:
# First, use `odeint` from the `scipy.integrate` package to plot the graph and find the location of the second peak in fox population (eg. 425 days, 2800 foxes).
# 
# Then, implement a Kinetic Monte Carlo algorithm to simulate the same situation. See https://en.wikipedia.org/wiki/Kinetic_Monte_Carlo for details
# 
# Determine
# 
# 1. The expected location of the second peak in foxes (eg. 425 days, 2800 foxes), on occasions that there is one (eg. a peak that's  >200 days and >100 foxes)
# 
# 2. The interquartile range of the second peak in foxes (eg. 411-443 days, 2700-3120 foxes).
# 
# 3. The probability that the foxes die out before 600 days are complete
# 
# 
# Make sure you've done enough simulations to be suitably confident in your answers (given the precision you think appropriate).
# 
# Finally, list some things that you learned from this assignment.

# # Rabbits and foxes
# 
# There are initially 400 rabbits and 200 foxes on a farm (but it could be two cell types in a 96 well plate or something, if you prefer bio-engineering analogies). Plot the concentration of foxes and rabbits as a function of time for a period of up to 600 days. The predator-prey relationships are given by the following set of coupled ordinary differential equations:
# 
# \begin{align}
# \frac{dR}{dt} &= k_1 R - k_2 R F \tag{1}\\
# \frac{dF}{dt} &= k_3 R F - k_4 F \tag{2}\\
# \end{align}
# 
# * Constant for growth of rabbits $k_1 = 0.015$ day<sup>-1</sup>
# * Constant for death of rabbits being eaten by foxes $k_2 = 0.00004$ day<sup>-1</sup> foxes<sup>-1</sup>
# * Constant for growth of foxes after eating rabbits $k_3 = 0.0004$ day<sup>-1</sup> rabbits<sup>-1</sup>
# * Constant for death of foxes $k_4 = 0.04$ day<sup>-1</sup>
# 
# Also plot the number of foxes versus the number of rabbits.
# 
# Then try also with 
# * $k_3 = 0.00004$ day<sup>-1</sup> rabbits<sup>-1</sup>
# * $t_{final} = 800$ days
# 
# *This problem is based on one from Chapter 1 of H. Scott Fogler's textbook "Essentials of Chemical Reaction Engineering".*
# 

# # Solving ODEs
# 
# *Much of the following content reused under Creative Commons Attribution license CC-BY 4.0, code under MIT license (c)2014 L.A. Barba, G.F. Forsyth. Partly based on David Ketcheson's pendulum lesson, also under CC-BY. https://github.com/numerical-mooc/numerical-mooc*
# 
# Let's step back for a moment. Suppose we have a first-order ODE $u'=f(u)$. You know that if we were to integrate this, there would be an arbitrary constant of integration. To find its value, we do need to know one point on the curve $(t, u)$. When the derivative in the ODE is with respect to time, we call that point the _initial value_ and write something like this:
# 
# $$u(t=0)=u_0$$
# 
# In the case of a second-order ODE, we already saw how to write it as a system of first-order ODEs, and we would need an initial value for each equation: two conditions are needed to determine our constants of integration. The same applies for higher-order ODEs: if it is of order $n$, we can write it as $n$ first-order equations, and we need $n$ known values. If we have that data, we call the problem an _initial value problem_.
# 
# Remember the definition of a derivative? The derivative represents the slope of the tangent at a point of the curve $u=u(t)$, and the definition of the derivative $u'$ for a function is:
# 
# $$u'(t) = \lim_{\Delta t\rightarrow 0} \frac{u(t+\Delta t)-u(t)}{\Delta t}$$
# 
# If the step $\Delta t$ is already very small, we can _approximate_ the derivative by dropping the limit. We can write:
# 
# $$\begin{equation}
# u(t+\Delta t) \approx u(t) + u'(t) \Delta t
# \end{equation}$$
# 
# With this equation, and because we know $u'(t)=f(u)$, if we have an initial value, we can step by $\Delta t$ and find the value of $u(t+\Delta t)$, then we can take this value, and find $u(t+2\Delta t)$, and so on: we say that we _step in time_, numerically finding the solution $u(t)$ for a range of values: $t_1, t_2, t_3 \cdots$, each separated by $\Delta t$. The numerical solution of the ODE is simply the table of values $t_i, u_i$ that results from this process.
# 

# # Euler's method
# *Also known as "Simple Euler" or sometimes "Simple Error".*
# 
# The approximate solution at time $t_n$ is $u_n$, and the numerical solution of the differential equation consists of computing a sequence of approximate solutions by the following formula, based on Equation (10):
# 
# $$u_{n+1} = u_n + \Delta t \,f(u_n).$$
# 
# This formula is called **Euler's method**.
# 
# For the equations of the rabbits and foxes, Euler's method gives the following algorithm that we need to implement in code:
# 
# \begin{align}
# R_{n+1} & = R_n + \Delta t \left(k_1 R_n - k_2 R_n F_n \right) \\
# F_{n+1} & = F_n + \Delta t \left( k_3 R_n F_n - k_4 F_n \right).
# \end{align}
# 

# Kinetic Monte Carlo Notes from Class
# 
# Events:
# 
# Rabbit Born k1xR
# 
# Rabbit Dies k2xRxF
# 
# Fox Born 
# 
# Fox Dies
# 
# sum of rates is the expected number of events per time
# 
# 1/sum is the average wait time
# 
# exponential time distribution can be used, and events chosen based on probability

# R = 400  # rabbits
# F = 200  # foxes
# 
# 
# def pops(pops0, t):
#     k1 = .015  # 1/days
#     k2 = .00004  # 1/(days*foxes)
#     k3 = .00004  # 1/(days*rabbits)
#     k4 = .04  # 1/days
#     dxdt = k1*pops0[0]-k2*pops0[0]*pops0[1]
#     dydt = k3*pops0[0]*pops0[1]-k4*pops0[1]
#     ddt = [dxdt, dydt]
#     return ddt
# 
# t = np.linspace(0, 800, 10000)
# pops0 = [R, F]
# 
# populations = odeint(pops, pops0, t)
# 
# plt.plot(1)
# plt.title('k3 = 0.00004')
# plt.plot(t, populations[:, 0], 'b-')
# plt.plot(t, populations[:, 1], 'r--')
# plt.xlabel('Time (days)')
# plt.ylabel('Population')
# plt.legend(['Rabbits', 'Foxes'], loc=0)
# plt.show()

# In[1]:

#get_ipython().magic(u'matplotlib inline')

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from matplotlib import cm as CM
from matplotlib import mlab as ML


# In[2]:

def second_max(Time, Var):
# Returns the second maximum of the given 1-D Array (as an int) and the corresponding time (as a float)
# This function only works if the second peak is in the second half of the data set by itself

    x = Var[:int(np.floor(len(Var)/2))]
    y = Var[int(np.ceil(len(Var)/2)):] # Splits the given array in two
    index = np.argmax(y)+(len(x)-1) # gives the index of the maximum value with respect to original array
    
    return ([(round(Time[index], 2)), int(round(Var[index]))]) # round function added because int is floor


# In[3]:

R = 400  # initial rabbits
F = 200  # initial foxes
TimeStep = .01 #desired timestep for odeint feed

def pops(pops0, t):
    #Returns the derivative of the rabbit and fox populations to be fed into odeint
    k1 = .015  # 1/days
    k2 = .00004  # 1/(days*foxes)
    k3 = .0004  # 1/(days*rabbits)
    k4 = .04  # 1/days
    dxdt = k1*pops0[0]-k2*pops0[0]*pops0[1]
    dydt = k3*pops0[0]*pops0[1]-k4*pops0[1]
    return [dxdt, dydt]

t = np.arange(0, 600, TimeStep) #create a timespan for odeint

populations = odeint(pops, [R,F], t) #run odeint

#Make plot of odeint output and adust axis label's and legend settings
plt.plot(1)
plt.title('k3 = 0.0004')
plt.plot(t, populations[:,0], 'b-', label='Rabbits')
plt.plot(t, populations[:,1], 'r--', label='Foxes')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.legend(loc=0)
plt.show()

#runs second peak function on odeint output
ans = second_max(t, populations[:,1])

#print action
print("The second peak of foxes occurs at %.3f days and there are %d foxes." % (ans[0], ans[1]))


# In[4]:

# Set initial conditions
Ri = 400 # Initial number of rabbits
Fi = 200 # Initial number of foxes

CountRDeath = 0 # Number of times KMC resulted in all rabbits dying
CountFDeath = 0 # Number of times KMC resulted in all foxes dying

FPeak2 = [] # List of lists for second peak locations and times
trials = 10000 # number of trials

#define rates
k1 = .015  # 1/days
k2 = .00004  # 1/(days*foxes)
k3 = .0004  # 1/(days*rabbits)
k4 = .04  # 1/days

# for loop runs multiple simulations
for i in range(trials):
# reset lists
    R = [Ri]  #Resets number of Rabbits
    F = [Fi] #Resets number of Foxes
    t = [0] #Resets time index
    
    RDead = False # Keeps track of status of rabbit population
    FDead = False # Keeps track of status of fox population
    

    # while loop runs individual KMC simulations
    while t[-1] <= 600: #runs until t > 600 days

        # Sets all rates and sums of rates to avoid inefficient calculations in every if statement
        Q = k1*R[-1] + k2*R[-1]*F[-1] + k3*R[-1]*F[-1] + k4*F[-1]
        r1 = k1*R[-1]
        r2 = k1*R[-1] + k2*R[-1]*F[-1]
        r3 = k1*R[-1] + k2*R[-1]*F[-1] + k3*R[-1]*F[-1]
        
        # Determines which action will be taken
        val = np.random.rand() * Q
        
        # this section checks if the populations are non-zero
        if F[-1] < 1:
            # print("All foxes disappeared after %.3f days." % (t))
            CountFDeath += 1
            FDead = True
            break
            
        elif R[-1] < 1:
            # print("All rabbits disappeared after %.3f days." % (t))
            CountRDeath += 1
            FDead = True
            break

        # This section determines the correct action to take and appends the lists accordingly
        if val < r1:
            # Rabbit is born
            R.append(R[-1] + 1)
            F.append(F[-1])
            t.append(t[-1] + 1/Q * np.log(1/np.random.rand()))

        elif val < r2:
            # Rabbit is killed by a fox
            R.append(R[-1] - 1)
            F.append(F[-1])
            t.append(t[-1] + 1/Q * np.log(1/np.random.rand()))

        elif val < r3:
            # Fox is born
            F.append(F[-1] + 1)
            R.append(R[-1])
            t.append(t[-1] + 1/Q * np.log(1/np.random.rand()))

        elif val < Q:
            # Fox dies
            F.append(F[-1] - 1)
            R.append(R[-1])
            t.append(t[-1] + 1/Q * np.log(1/np.random.rand()))

        else:
            #breaks loop when an error occurs in the while loop
            print("Error in while loop")
            break

    
    # If neither of the populations died in a given KMC run, finds the second peak and saves it
    if FDead == False and RDead == False:
        FPeak2.append(second_max(t, np.array(F)))


# In[7]:

aFPeak2 = np.array(FPeak2) #converts list output to an array
tq75, tq25 = np.percentile(aFPeak2[:,0], [75 ,25]) #finds the quartiles of the time
fq75, fq25 = np.percentile(aFPeak2[:,1], [75 ,25]) #finds the quartiles of the fox population

#print output
print("The second peak of foxes occurs, on average, at %.3f days and there are %d foxes." % (np.mean(aFPeak2[:, 0]), np.mean(aFPeak2[:, 1])))
print("The interquartile range of the time was from %.3f to %.3f days." % (tq25, tq75))
print("The interquartile range of the number of foxes was from %.3f to %.3f. \n" % (fq25, fq75))

print("The probability that all of the rabbits would die was %.1f%%. " % (CountRDeath/trials*100))
print("The probability that all of the foxes would die was %.1f%%. " % (CountFDeath/trials*100))
print("Therefore, the total number of second peaks found was %d. " % (trials - CountRDeath - CountFDeath))


# This assignment was actually pretty helpful. It gave me an idea of how modeling works beyond simply using differential equations. Actually implementing the KMC simulation for this system without looking up code forced me to make a lot of mistakes that ended up helping me understand it better. I am sure there are better ways to do it, but this works decently well for the problem. 
# 
# Anything after this markdown is not required, I just wanted to try it out.

# In[8]:

plt.hexbin(aFPeak2[:, 0], aFPeak2[:, 1], gridsize=20, cmap=CM.gist_heat_r, bins=None)
plt.axis([0, 600, 0, 5000])

cb = plt.colorbar()
cb.set_label('Occurances')
plt.xlabel('Time (days)')
plt.ylabel('Foxes')
plt.show()
print("This plot shows the location of the maxima.")


# In[9]:

Average = [aFPeak2[0, 1]]
Quartile75 = [np.percentile(aFPeak2[0,1], 75)]
Quartile25 = [np.percentile(aFPeak2[0,1], 75)]

for i in range(1, len(aFPeak2)):
    Average.append(np.mean(aFPeak2[:i, 1]))
    Quartile75.append(np.percentile(aFPeak2[:i,1], 75))
    Quartile25.append(np.percentile(aFPeak2[:i,1], 25))
    
plt.plot(Average, 'b', label='Average')
plt.plot(Quartile75, 'r--', label='Quartile')
plt.plot(Quartile25, 'r--')
plt.xlabel('Total Peaks')
plt.ylabel('Foxes')
plt.legend(loc=0)
plt.show()
print("This plot tracks the average amplitude of the second peak of foxes.")


# In[ ]:



