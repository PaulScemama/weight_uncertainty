# Preface
A probabilistic model is an approximation of nature. In particular, it approximates the process by which our observed data was created. While this approximation may be causally and scientifically inaccurate, it can still provide utility based on the goals of the practitioner. For instance, finding associations through our model can be useful for prediction even when the underlying generative assumptions don't mirror reality. From the perspective of probabilistic modeling, our data $y_{1:N}$ are viewed as realizations of a random process that involves hidden quantities -- termed *hidden variables*. 

Hidden variables are quantities which we believe played a role in generating our data, but unlike our data which is observed, their particular values are unknown to us. The goal of *inference* is to use our observed data to uncover the likely values of the hidden variables in our model in the form of posterior distributions over those hidden variables. Hidden variables are partitioned into two categories: *global* hidden variables and *local* hidden variables, which we denote $\theta$ and $z_{1:N}$, respectively. Most people are familiar with global hidden variables. These are variables that we assume govern all $N$ elements of our observed data. Models containing local hidden variables are often called "Latent Variable Models". This entire exposition is all to say that this implementation (inspired by the paper "Weight Uncertainty in Neural Networks") only deals with global variable models. For example, the supervised setting where we map inputs $x_{1:N}$ to outputs $y_{1:N}$ with a single neural network. Each "weight" in the neural network is a global variable because it governs all $y_{1:N}$ in the same way.

The reason for this lengthy preface is that a lot of resources on mean-field variational inference (the focus of this repository) speak in terms of both local and global hidden variables, and how we treat them during inference is different.



# Mean-field Variational Inference: Motivation

We have a probabilistic model of our data $y_{1:N}$:

$$
p(\theta, y | x) = \underbrace{p(y|\theta, x)}_{\text{observation model}} \overbrace{p(\theta)}^{\text{prior}}
$$

We'd like to compute the posterior $p(\theta|y_{1:N}, x_{1:N})$. To do so we use Bayes' theorem,

$$
p(\theta | y_{1:N}, x_{1:N}) = \frac{p(\theta, y_{1:N}|x_{1:N})}{\underbrace{p(y_{1:N}|x_{1:N})}_{\text{normalizing constant}}}
$$

The normalizing constant involves a large multi-dimensional integration which is generally intractable. Mean-field variational inference turns this "integration problem" into an "optimization problem" which is much easier (computing derivatives is fairly easy, integration if very hard). 


# Mean-field Variational Inference: How It's Done

From now on I will suppress the $(\cdot)_{1:N}$ to unclutter notation. 

The basic premise of variational inference is to propose a _variational distribution_ $q(\theta)$ (that we can deal with easily) and minimize the KL divergence between this distribution and the true posterior $p(\theta|y, x)$:

$$
\underset{q}{\text{argmax }} \text{KL}[q(\theta)  ||  p(\theta|y, x)] \tag{1}
$$

We will see that we cannot directly do this because it ends up involving the computation of the "evidence" $p(y|x)$ (the quantity for which we appeal to approximate inference in the first place!). We instead optimize a related quantity termed the *evidence lower bound* (ELBO). To see why, we expand the KL divergence $(1)$:...








