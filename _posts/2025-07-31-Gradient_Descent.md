---
layout: post
title: "Gradient Descent from Scratch"
date: 2025-07-31
tags: []
categories: [From Scratch]
description: ""
featured: true
thumbnail: 
---

Hell yeah, Here we go again! It's been some time, that's for sure.

I'm thinking of starting a new series of blog posts where we'll be trying to implement things from scratch; Let's kick start things with gradient descent algorithm(from scratch) with only using NumPy(Numerical Python Library). I'll be writing the code alongside the theory to better understand this super awesome algorithm. 

There are plenty of guides over the internet on this, you can name it protege effect or anything, but teaching others only enhances your own clarity on the topic. Okay lets cut to the chase and jump right into the topic. 

## Introduction

With the onset of big data- Neural nets have found there way into the mainstream. To train these mammoth neural nets, we are always looking for ways to save our expenses while not compromising on the quality of the training. 

### What's our approach? 

We are trying to minimize a "decided" Loss function(with loads of parameters, like loads and loads), i.e., looking out for suitable values for those parameters which minimize the above decided loss function. 

Note: I'm using *decided*, as the loss depends on the problem we are trying to solve. 

More mathematically, we have an optimization problem as, 
$$\min_{w\in\mathbb{R}^D}E(w)$$
$$w^*=\text{arg} \min_{w\in\mathbb{R}^D}E(w)$$

$w^*$ is the vector, why we are doing all this. It will be our gateway to awesome predictions. 

The Objective function [$E(w)$ above] is usually a super complicated function; by complicated, i mean some function which cannot be solved in closed form($w^*$= some function involving the independent variables), and needs some deep introspection to find its global minima.

There are plenty of iterative methods to solve for the above optimisation problems. Like the Newton Method and all, but the one which has took the world by storm is- Gradient Descent and Stochastic Gradient descent, a clever extension which serves as the basis for training super large neural nets. 


## Gradient Descent

It's basically an Algorithm, if followed with the right-set of hyperparamters will fetch you the golden $w^*$ that you are looking out for.  

Getting into math of it, its an iterative first-order algorithm for minimizing a differentiable multivariate function.

Let's get a few mechanical details out of the way; 

Gradient Descent $\iff$ used for finding the local minima of multivariate functions

Gradient Ascent $\iff$ used for finding the local maxima of multivariate functions 

Let me walk you through a short lecture on gradients, 

For a Multi-Variable function $F(x)$ defined and differentiable in a neighborhood of a point $a$, then $F(x)$ increases the fastest if one goes from $a$ in the direction of the gradient of $F$ at $a$ i.e., $\frac{\partial F}{\partial x}{\Large |}_{x=a}$

In simple English, a recursive operation to look for the local minimum of seemingly daunting functions. 

Idea is that we use iterative methods to solve the optimization problem, 
$$w^{(0)}\rightarrow w^{(1)}... \rightarrow w^{(\infty)}=w^\star$$

We'll be initializing the weight vector $w^{(0)}$ usually with small random numbers; I'm using the zero index to signify that as our initial guess of optimal parameters. We'll continue using our recursive function to get hold of the minimum. 

### When to stop?

This is a million-dollar question and the stopping criteria can vary- It can be limited to a set number of iterations. The difference between the updates is quite small and many others. 

Which one to choose is simply upto the user and the problem that their trying to solve. 

Let's jump back to the algorithm, 

<!-- For Example we can initialize them as $w_d^{(0)}\sim$ uniform$[-0.01,0.01]$ : $\forall d$ -->


*Notation Alert: The number of training samples  ${(x^{(i)},y^{(i)})}$ is $m$*


> **Example**
> Eg: $w_d^{(0)}\sim$ uniform$[-0.01,0.01]$ : $\forall d$
> Initial weight vector $w^{(0)}$: usually small random numbers. 
> 
> For Demonstration purpose, lets assume the squared loss function (the el classico of all loss functions), and later I will work out the details for $L^1$ Loss function as well.  
> - We are working with a supervised learning algorithm, so there will be a training set of the form,  
>    $T=\{(x^{(1)},y^{(1)}),(x^{(2)},y^{(2)}),...,(x^{(m)},y^{(m)})\}$  
>    where each $x^{(i)}$ is a $p$ dimensional vector $: \forall i$  
> - Consider the ordinary least square regression model, then the hypothesis function is: $h_{\theta}(x)=\sum_{i=0}^{d}\theta_i x_i=\theta^Tx$ where $x_0=1$  
> - The $L^2$ Loss function is  $\mathbb{L}(y_i,\hat{y_i})=(y_i-\hat{y_i})^2$  
> - if we take the sum of all the loss functions over the training set, we get the cost function  
> - $J(\theta)=\frac{1}{2}\sum_{i=1}^{m}L(y_i,\hat{y_i})=\frac{1}{2}\sum_{i=1}^{m} L(y_i,h_{\theta}(x_i))=\frac{1}{2}\sum_{i=1}^{m} (h_{\theta}(x^{(i)})-y^{(i)})^2$  
>
> Now this is Cost Function, we are looking for the values which minimize this function. 


Detail to ponder over: Here the half($\frac{1}{2}$) in cost function is added to simplify the calculation and nothing else. Any positive monotonic transformation of the objective function won't change the points of minima($arg \hspace{2 mm}min$). 

>Gradient Descent Algorithm 
>- Update rule
	$\theta_j \leftarrow \theta_j +\Delta \theta_j$ with an update $\Delta \theta_j =-\alpha \frac{\partial J(\theta)}{\partial \theta_j}$, element wise for $d=1,...,m$
>
>for the above $L^2$ Loss function, upon working out the derivative, >we get it as the following, 
>$$
\frac{\partial J(\theta)}{\partial \theta}= \sum_{i=1}^{m} (\theta^T x^{(i)}-y^{(i)})x^{(i)}
>$$
>
>Therefore, the update will become, 
>$$
\theta:= \theta + \alpha \sum_{i=1}^m (y^{i}-\theta^T x^{i})x^{i}
>$$
>
>To sum it up in vector notation, it will be(for a generic cost >function $J(\theta)$)
>$$
\theta :=\theta -\alpha \nabla_\theta J(\theta)
$$


Here is the code implementation of it,
I'll briefly discuss the idea of it, so basically we are building a class which will take in the training data and 


```python
import numpy as np
```
Thats all the library imports we'll be using

```python
class GradientDescent:

    def __init__(self,training_data,start,learning_rate,num_iterations):
        """
        training_data: The data that is to be considered for the supervised learning problem \n
        start: Initial guess of the parameters \n
        learning_rate: Rate which determines the step size for every iteration \n
        num_iterations: Total Number of Iterations
        """
        # Convert training data to matrices
        X = []
        y = []
        for x_vec, y_val in training_data:
            X.append([1] + list(x_vec))  # Add bias term
            y.append(y_val)
        self.X = np.array(X)  # Shape: (m, n+1) where m=samples, n=features
        self.y = np.array(y)  # Shape: (m,)
        self.m=len(training_data) # number of training examples
        self.start = np.array(start) # Inital starting point
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.points = [tuple(start)]

    def compute_gradient_loop(self, current):
        current = np.array(current)
        grad = np.zeros_like(current)
        for i in range(self.m):
            x_i = self.X[i]  # i-th training example
            y_i = self.y[i]  # i-th target
            residual = y_i - np.dot(current, x_i)
            grad += residual * x_i
        return grad

    def run(self):
        current=np.array(self.start)
        for _ in range(self.num_iterations):
            grad=self.compute_gradient_loop(current)
            current = current + self.learning_rate * grad
            self.points.append(tuple(current))

    def predict(self,query_point):
        query_point_with_dummy=[1.0]+list(query_point)
        return np.dot(np.array(self.points[-1]),np.array(query_point_with_dummy))

```

## Stochastic Gradient Descent

Now lets build another class which will run the stochastic gradient descent, as the name suggest, it is stochastic- the parameter value update is now based on the gradient of a single training example(choosen randomly from the training set) and this greatly reduces the run-time(as now you wont be using the all the training samples to calculate the gradient). 

I'll show the results.  Particularly when the training set is large, the Stochastic Gradient Descent is preferred over the Batch Gradient Descent. 

```python
import numpy as np 

class StochasticGradientDescent:
    def __init__(self,training_data,start,learning_rate,num_iterations):
        """
        training_data: The data that is to be considered for the supervised learning problem \n
        start: Initial guess of the parameters \n
        learning_rate: Rate which determines the step size for every iteration \n
        num_iterations: Total Number of Iterations
        """
        self.training_data=[(tuple([1]+list(x)),y) for x,y in training_data]
        self.start=start
        self.learning_rate=learning_rate
        self.num_iterations=num_iterations
        self.points=[start]
    
    def compute_gradient(self,point,i):
        grad=np.zeros_like(point)
        point=np.array(point)
        x_i=np.array(self.training_data[i][0])
        y_i=self.training_data[i][1]
        residual=y_i-np.dot(point,x_i)
        grad+=(residual)*x_i
        return tuple(grad)
    
    def run(self):
        for _ in range(self.num_iterations):
            i=np.random.randint(0,len(self.training_data))
            current=self.points[-1]
            gradient=self.compute_gradient(current,i)
            update = tuple(current[j] + self.learning_rate * gradient[j] for j in range(len(current)))
            self.points.append(update)
    
    def predict(self,query_point):
        query_point_with_dummy=[1.0]+list(query_point)
        return np.dot(np.array(self.points[-1]),np.array(query_point_with_dummy))
```

Now let's put these two algorithms to a race,  

```python
#importing the time module to compare the run-time

import time
training_data = [([1.0, 2.0,4.0,5.0], 12.0),([1.0, 8.0,4.0,6.0], 19.0), ([2.0,4.0, 4.0,5.0], 15.0)]

gd = GradientDescent(start=(0.0,0.0,0.0,0.0,0), learning_rate=0.005, num_iterations=int(10e5), training_data=training_data)
start=time.time()
gd.run()
end=time.time()

print(f'parameters from Batch GD: {gd.points[-1]}\n')
print(f'time taken: {(end-start)}\n')
print(f'prediction for (3,3,3,3): {gd.predict([3,3,3,3])}\n')

sgd = StochasticGradientDescent(start=(0.0,0.0,0.0,0.0,0), learning_rate=0.005, num_iterations=int(10e5), training_data=training_data)
start=time.time()
sgd.run()
end=time.time()

print(f'parameters from Stochastic GD: {sgd.points[-1]}\n')
print(f'time taken: {(end-start)}\n')
print(f'prediction for (3,3,3,3): {sgd.predict([3,3,3,3])}\n')

```

Let's take a look at the output,

*Red Alert: I have a pretty normal Laptop*

```output
parameters from Batch GD: (0.2341891045710697, 1.0012523481527535, 0.9993738259236024, 0.9367564182842788, 1.003757044458375)

time taken: 28.767800092697144

prediction for (3,3,3,3): 12.057608015028098

parameters from Stochastic GD: (0.23418910457018247, 1.001252348153782, 0.9993738259231028, 0.9367564182807299, 1.003757044461383)

time taken: 25.743896484375

prediction for (3,3,3,3): 12.057608015027176
```

We can see that for a mere $1,00,000$ iterations, we get the parameter estimates from both the algorithms to be very very close. Another point worth noting is the fact that, the Stochastic Gradient Descent was faster than Gradient Decent by more than 4 seconds on my laptop. This could be exploited to a great extent when the training set is huge. 

Literature also has the mention of Stochastic Batch Gradient Descent(SBGD), which is a slight ugrade over the vanilla SGD as here in SBGD, we sample n(Batch Size) without replacement and go about following the same steps of SGD. SBGD is proven to yield us better results but just a tad bit noiser in the training. 

---
### Conclusion


In summary, these two algorithms are commonly used to training large neural nets(or, some convex combination of them). The underlying mathematical  foundations underlying these methods are truly elegant! 

If you are interested in experimenting with the code, you can access the accompanying Python notebook for hands-on practice.

Happy Learning! 
