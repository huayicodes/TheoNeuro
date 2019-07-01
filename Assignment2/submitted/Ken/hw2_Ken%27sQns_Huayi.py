
# coding: utf-8

# Problem 1: The inhibition-stabilized network (ISN). 
# 
# 1a. Now for a steady-state input perturbation δi, write down the equation for the
# steady-state response $δr$. 
# 
# Answer: the equation can be written as $T(1 - J)^{-1} \frac{d}{dt}δr = -δr + δi(1 - J)^{-1}. $ 
# Solving by exponential: $ δr(t) = δi(1 - J)^{-1} + (δr(0) - δi(1 - J)^{-1}) e^{\frac{-t}{T(1 - J)^{-1}}}$
# At steady state, $t \rightarrow ∞$, $ δr = -(J-1)^{-1}δi. \\ $
# 
# 
# 1b. Show that, for a stable fixed point, if a perturbation is given only to the I cells (δi ∝ [0 1]), the steady-state response is “paradoxical” (response to a positive input is negative; response to a negative input is positive) if and only if $j_{EE} > 1$. 
# 
# Answer: $ δr = \frac{1}{Det(J-I)} \begin{bmatrix} -j_{II}-1 & j_{EI} \\ -j_{IE} & j_{EE}-1 \end{bmatrix} \begin{bmatrix} 0 \\  δi\end{bmatrix} \\ = -  \frac{δi}{Det(J-I)}\begin{bmatrix}  j_{EI} \\ j_{EE}-1 \end{bmatrix}$
# 
# To let the term $-  \frac{δi}{Det(J-I)}$ determines the paradoxical sign of δr (i.e. when δi >0, δr < 0 and vice versa.), $j_{EE}-1 > 0$, thus $j_{EE} > 1 \\$.
# 
# 
# 1c. Show that $j_{EE} > 1 $ is precisely the condition that the E subpopulation alone is unstable: that is, if $r_I$ were a fixed constant, fixed to its steady state value, so that the only dynamics were the E equation with the fixed $r_I$, then the fixed point would be unstable.
# 
# Answer: $Det(J-I) = -(j_{EE}-1)(j_{II}+1) + j_{EI}j_{IE}$. For Det(J-I) > 0, $ (j_{EE}-1) < \frac{j_{EI}j_{IE}}{(j_{II}+1)}$
# When $r_I$ is fixed, this means that the excitatory population can no longer affect the inhibitory population, thus $j_{IE} = 0$. For the population to be stable, $ j_{EE}-1 < 0, or \  j_{EE} < 1$. Thus, if $j_{EE} > 1$, then $Det(J-I) < 0$, and the system becomes unstable.
# 
# 
# 2a. For the I nullcline, compute its slope, $dr_I/dr_E$; you should find that it is given by $\frac{j_{IE}}{1+j_{II}}$. This means that the nullcline always has positive slope.
# 
# Answer: Compute the I nucline $T\frac{dr_I}{dt} = -r_I + j_{IE}r_E - j_{II}r_I + i_I. \\ Set \ \frac{dr_I}{dt} = 0, \ then \  r_I = \frac{j_{IE}}{1+j_{II}}r_E - \frac{i_I}{1+j_{II}}.\\ Thus \ the \ slope \ is \ \frac{j_{IE}}{1+j_{II}}. \\$
# 
# 2b. Now for the E nullcline, compute the inverse of its slope, $dr_E/dr_I$; you should find that this inverse slope is $\frac{j_{EI}}{j_{EE} -1}$. This means that the slope is positive if the E subnetwork is unstable, and negative if the E subnetwork is stable.
# 
# Answer: Compute the E nucline $T\frac{dr_E}{dt} = -r_E + j_{EE}r_E - j_{EI}r_I + i_E. \\ Set \ \frac{dr_E}{dt} = 0, \ then \  r_E = \frac{j_{EI}}{j_{EE} -1}r_I - \frac{i_E}{j_{EE}-1}.\\ Thus \ the \ slope \ is \ \frac{j_{EI}}{j_{EE} -1}. \\$
#  
# 
# 2c. Show that the condition that Det (J − 1) > 0, which is necessary for stability, is equivalent to the I nullcline having a larger slope than the E nullcline. So for a fixed point to be stable, it is necessary that the I nullcline have a larger slope than the E nullcline at their crossing that defines the fixed point.
# 
# Answer: The I nullcline has a positive slope and is always stable, but for E nullcline it needs a negative slope to be stable. Thus, the I nullcline will always have a larger slope than the E nullcline. For $\frac{j_{EI}}{j_{EE} -1} < 0, \ it \ requires\ j_{EE} -1 < 0, \ thus \ j_{EE} < 1.$ This is the same condition for Det (J − 1) > 0.
# 
# 2d. Graph two versions of the nullclines. 
# Answer: see below. 
# 
# 2e. Now, suppose you add a positive input to the I cells. Show that the resulting change in the I nullcline is to reduce rE by the same amount for any given rI, that is, to move the I nullcline leftward.
# 
# Answer: When input current is zero, the I nullcline is effectively $r_I = \frac{j_{IE}}{1+j_{II}}r_E$. With positive input current, the I nullcline becomes $r_I = \frac{j_{IE}}{1+j_{II}}r_E + \frac{i_I}{1+j_{II}}$. Thus, the curve shifts upwards, and for a sinusoidal curve, that's the same as shifting leftwards. 
# 
# 2d. Show on the plot that, for a stable fixed point, if the network is an ISN, the result is to decrease both rE
# and rI in moving to the new fixed point; while for a non-ISN, the result is to decrease rE but increase rI.
# 
# Answer: As shown below by the position of the new fixed point, rE is reduced for both cases, but ISN has smaller rI, while non-ISN has bigger rI. 
# 

# In[547]:

import numpy as np
import matplotlib.pyplot as plt

# ISN
# creating fake data that looks like nullclines 
t = np.arange(0,1,0.001)
rI = 3*np.cos(np.pi*1*t+np.pi) +3
rE = - np.sin(np.pi*2.5*t-np.pi/4) +3

rI2 = 3*np.cos(np.pi*1*(t+0.06)+np.pi) +3

# find new fix point by looking for interception
idx = np.argwhere(np.diff(np.sign(rI2 - rE)) != 0).reshape(-1) + 0

plt.plot(t,rI,'b')
plt.plot(t,rE,'r')
plt.plot(t,rI2,'b--')
plt.plot(0.5, 3, 'o', mfc='none', mew = 1.5) #initial fixed point
plt.plot(t[idx], rI2[idx], 'o', mfc='none', mew = 1.5, mec = '0.3') # fixed point with input current

#Draw the arrows indicating the direction of flow in the different regions of the nullcline plane
plt.arrow(0.6, 3.8, 0.2, 1, head_width=0.02, head_length=0.1, fc='k', ec='k', linestyle=':')
plt.arrow(0.4, 2.2, -0.2, -1, head_width=0.02, head_length=0.1, fc='k', ec='k', linestyle=':')
plt.arrow(0.25, 4.2, 0.2, -1, head_width=0.02, head_length=0.1, fc='k', ec='k', linestyle=':')
plt.arrow(0.75, 1.8, -0.2, 1, head_width=0.02, head_length=0.1, fc='k', ec='k', linestyle=':')

# plotting to show: in negative-sloping regions of the E nullcline, if rI is kept fixed, small perturbations off the E nullcline will flow back to the nullcline;
plt.arrow(0, 3, 0.07, 0, head_width=0.2, head_length=0.01, fc='k', ec='k')
plt.arrow(0.21, 3, -0.07, 0, head_width=0.2, head_length=0.01, fc='k', ec='k')
# plotting to show: while in positive-sloping regions, it will flow away.
plt.arrow(0.45, 2.7, -0.07, 0, head_width=0.2, head_length=0.01, fc='k', ec='k')
plt.arrow(0.55, 3.3, 0.07, 0, head_width=0.2, head_length=0.01, fc='k', ec='k')

# plot the dynamic path
## Dear TA: I'm not sure how to plot arcs,so I made straight arrows. Please imagine the green arrows linked with curved corners. 
plt.arrow(0.5, 3, 0.15, 1.25, head_width=0.02, head_length=0.1, fc='g', ec='g' )
plt.arrow(0.65, 4.25, -0.11, -0, head_width=0.1, head_length=0.02, fc='g', ec='g')
plt.arrow(0.65-0.12, 4.25, -0.18, -2, head_width=0.02, head_length=0.1, fc='g', ec='g')


plt.xlim([0,1.0])
plt.xlabel('rE')
plt.ylabel('rI')
plt.title('ISN')
plt.legend(['I-nullcline','E-nullcine','I-nullcline with input'],loc='best')
plt.show()


# non-ISN
plt.figure()
# creating fake data that looks like nullclines 
t = np.arange(0,1,0.001) 
rI = 3*np.cos(np.pi*1*t+np.pi) +3
rE = -3*(t-0.5) +3

rI2 = 3*np.cos(np.pi*1*(t+0.06)+np.pi) +3 # shifted inhibitory nullcline if curent is injected into the inhibitory cells. 

# find new fix point by looking for interception
idx = np.argwhere(np.diff(np.sign(rI2 - rE)) != 0).reshape(-1) + 0

plt.plot(t,rI,'b')
plt.plot(t,rE,'r')
plt.plot(t,rI2,'b--')
plt.plot(0.5, 3, 'ko', mew = 1.5) # initial fixed point
plt.plot(t[idx], rI2[idx], 'o', mfc= '0.3', mew = 1.5, mec = '0.3' )  # fixed point with input current

#Draw the arrows indicating the direction of flow in the different regions of the nullcline plane
plt.arrow(0.7, 3.8, -0.15, -0.6, head_width=0.02, head_length=0.1, fc='k', ec='k', linestyle=':')
plt.arrow(0.3, 2.2, 0.15, 0.6, head_width=0.02, head_length=0.1, fc='k', ec='k', linestyle=':')
plt.arrow(0.35, 4.5, 0.13, -1.3, head_width=0.02, head_length=0.1, fc='k', ec='k', linestyle=':')
plt.arrow(0.65, 1.5, -0.13, 1.3, head_width=0.02, head_length=0.1, fc='k', ec='k', linestyle=':')

plt.xlim([0,1.0])
plt.xlabel('rE')
plt.ylabel('rI')
plt.title('Non-ISN')
plt.legend(['I-nullcline','E-nullcine','I-nullcline with input'],loc='best')
plt.show()


# Problem 2: Non-normal dynamics
# 
# 2a. Verify that the (unnormalized) eigenvectors are $ \begin{bmatrix} 1\\ 1 \end{bmatrix}$ with eigenvalue $λ_1 = -x$, and $ \begin{bmatrix} \frac{w_1+x}{w_2} \\ 1 \end{bmatrix}$ eigenvalue $λ_2 = w_1 - w_2$.
# 
# Answer: $T \frac{d}{dt}r = -r + Wr +i.$ For Eigen value λ, Wr = λIr. (W- λI)r = 0. If (W- λI) is invertible, then there's no solution. Thus (W- λI) needs to be non-invertible, i.e. det(W- λI) = 0. $λ = \frac{w_1 - w_2 -x \pm \sqrt{(w_1 - (w_2+x))^2 - 4(w_1w_2-w_1x+w_2w_1+w_2x)}}{2}.  λ = \frac{w_1 - w_2 -x \pm (w_2-w_1-x)}{2}, \ λ_1 = -x, \ λ_2 = w_1 - w_2. \\ 
# λ_1 \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} -x \\ -x \end{bmatrix}. \ W\begin{bmatrix} 1\\ 1 \end{bmatrix} = \begin{bmatrix} w_1-(w_1+x)\\ w_2 - (w_2+ x) \end{bmatrix} = \begin{bmatrix} -x\\ -x \end{bmatrix}. Verfitied \\ λ_2\begin{bmatrix} \frac{w_1+x}{w_2} \\ 1 \end{bmatrix} = \begin{bmatrix} \frac{(w_1+x)(w_1 - w_2)}{w_2} \\ (w_1 - w_2)\end{bmatrix}. \ W\begin{bmatrix} \frac{w_1+x}{w_2} \\ 1 \end{bmatrix} = \begin{bmatrix} \frac{(w_1+x)w_1}{w_2} -(w_1+x) \\ \frac{(w_1+x)w_2}{w_2} -(w_2+x) \end{bmatrix} = \begin{bmatrix} \frac{(w_1+x)(w_1 - w_2)}{w_2} \\ (w_1 - w_2)\end{bmatrix}. \  Verified \\ $
# 
# 2b. Schur transformation.We'll choose our (normalized) Schur basis vectors to be $s_1 = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ 1 \end{bmatrix}$, and a vector orthogonal to it, $s_2 = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ -1 \end{bmatrix}$.Since the component of a vector along an orthonormal basis vector is just given by the dot product of the vector with the basis vector, show that r has components $ \begin{bmatrix} r_1 \\ r_2 \end{bmatrix} = \begin{bmatrix} \frac{r_E + r_I}{\sqrt{2}} \\ \frac{r_E - r_I}{\sqrt{2}} \end{bmatrix}$ in the s1, s2 basis, that is, the components r1 and r2 represent the sum and difference of E and I activities, respectively.
# 
# Answer: $r = (r_E \ r_I), \ r_1 = r \cdot s_1 = (r_E \ r_I) \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ 1 \end{bmatrix} = \frac{r_E + r_I}{\sqrt{2}} \\ r_2 = r \cdot s_2 = (r_E \ r_I) \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ -1 \end{bmatrix} = \frac{r_E - r_I}{\sqrt{2}}.$ 
# 
# Thus, $ \begin{bmatrix} r_1 \\ r_2 \end{bmatrix} = \begin{bmatrix} \frac{r_E + r_I}{\sqrt{2}} \\ \frac{r_E - r_I}{\sqrt{2}} \end{bmatrix}$
# 
# 
# 2c. You already know that $Ws_1 = -xs_1$. Show that $Ws_2 = (w_1 +w_2 +x)s_1 +(w_1 -w_2)s_2$. 
# 
# Answer: $Ws_2 = \frac{1}{\sqrt{2}} \begin{bmatrix} w_1+w_1+x \\ w_2+w_2+x \end{bmatrix} = \frac{1}{\sqrt{2}} \begin{bmatrix} w_1 +w_2 +x + (w_1-w_2) \\ w_1 +w_2 +x - (w_1-w_2)  \end{bmatrix} = (w_1 +w_2 +x)s_1 +(w_1 -w_2)s_2. \ Shown$ 
# 
# 2d. Show this also in the rE; rI basis: if wFF = w1 + w2 + x = 0, then the matrix W in this basis becomes a symmetric matrix, $W = W^T$, and therefore is normal; and the eigenvectors become orthogonal, i.e. the second eigenvector becomes $ \begin{bmatrix} 1 \\ 1 \end{bmatrix}$. 
# 
# Answer: $if \ w_FF = w_1 + w_2 + x = 0, \ then \ W = \begin{bmatrix} w_1 & -(w_1+x) \\ w_2 & -(w_2+x) \end{bmatrix} = \begin{bmatrix} w_1 & w_2 \\ w_2 & w_1 \end{bmatrix} = W^T. \ Shown.$
# 2e.Solve the linear dynamics $τ \frac{d}{dt} r = −r + Wr + i$ for constant i in the $s_1, s_2$ basis. You will have to first solve for $r_2(t)$, then solve for $r_1(t)$ with $w_{FF}r_2(t)$ as one of the inputs. 
# 
# Answer:  $τ \frac{d}{dt} r_2 = −r_2 + Wr_2 + i_2 = −r_2 + λ_2r_2 + i_2. \\ \frac{τ}{1-λ_2} \frac{d}{dt} r_2 = −r_2 + \frac{i_2}{1-λ_2}. \\ Solving \ the \ equation: \ r_2(t) = \frac{i_2}{1-λ_2} + (r_2(0) - \frac{i_2}{1-λ_2})e^{-\frac{(1-λ_2)t}{τ}} = r_2(0)e^{-\frac{(1-λ_2)t}{τ}} + \frac{i_2}{1-λ_2}(1-e^{-\frac{(1-λ_2)t}{τ}}). \\ τ \frac{d}{dt} r_1 = −r_1 + Wr_1 + i_1 = −r_1 + λ_1r_1 +w_{FF}r_2 + i_1. \\ \frac{τ}{1-λ_1} \frac{d}{dt} r_1 = −r_1 + \frac{w_{FF}r_2 + i_1}{1-λ_1}. \\ Solving \ the \ equation: \ r_1(t) = \frac{w_{FF}r_2(t) + i_1}{1-λ_1} + (r_1(0) - \frac{w_{FF}r_2(t) + i_1}{1-λ_1})e^{-\frac{(1-λ_1)t}{τ}} =  r_1(0)e^{-\frac{(1-λ_1)t}{τ}} + \frac{w_{FF}r_2(t) + i_1}{1-λ_1}(1-e^{-\frac{(1-λ_1)t}{τ}}) . \\ Substitute \ r_2(t) \ into \ the \ equation:r_1 (t) = r_1(0)e^{-\frac{(1-λ_1)t}{τ}} + \frac{i_1 + w_{FF}i_2}{(1 − λ_2)(1 − λ_1)}(1 - e^{-\frac{(1-λ_1)t}{τ}}) + w_{FF}(r_2(0) - \frac{i_2}{1 − λ_2}) \frac{e^{-\frac{(1-λ_1)t}{τ}}-e^{-\frac{(1-λ_2)t}{τ}}}{λ_1 -λ_2}$
# 
# 
# 2e: Graph the function $ \frac{e^{-\frac{(1-λ_1)t}{τ}}-e^{-\frac{(1-λ_2)t}{τ}}}{λ_1 -λ_2}$ for some choice of $λ_1$ and $λ_2$ as real numbers less than 1 (they can be negative).
# 
# Anwser: see below
# 
# Dear TA, I stopped here as the rest is optional. 
# 

# In[548]:

# plotting for 2E.  

lamda1 = 0.1
lamda2 = 0.5
tau = 1.0

t = np.arange(1,10,0.1)
ftrst = (np.exp(-(1-lamda1)*t/tau) -  np.exp(-(1-lamda2)*t/tau))/(lamda1-lamda2)

plt.plot(t,ftrst)
plt.xlabel('t')
plt.show()

