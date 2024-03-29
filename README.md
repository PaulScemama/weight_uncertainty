# Disclaimer

This repository is inspired by [Blackjax's](https://github.com/blackjax-devs/blackjax). In particular, the interface. I have also contributed to Blackjax and used some of the things I've learned here to incorporate in there. This was an exercise for me to implement an inference algorithm in JAX. It also allowed me to ramble a bit about variational inference and how Bayesian neural networks fit into the variational inference framework. 

- Be sure to check the amazing [Blackjax](https://github.com/blackjax-devs/blackjax) library out!
- Be sure to checkout the seminal paper from [(Blundell et. al, 2015)](https://arxiv.org/pdf/1505.05424.pdf) upon which this repository is also inspired by.


# Preface
A probabilistic model is an approximation of nature. In particular, it approximates the process by which our observed data was created. While this approximation may be causally and scientifically inaccurate, it can still provide utility based on the goals of the practitioner. For instance, finding associations through our model can be useful for prediction even when the underlying generative assumptions don't mirror reality. From the perspective of probabilistic modeling, our data $y_{1:N}$ are viewed as realizations of a random process that involves hidden quantities -- termed *hidden variables*. 

Hidden variables are quantities which we believe played a role in generating our data, but unlike our data which is observed, their particular values are unknown to us. The goal of *inference* is to use our observed data to uncover the likely values of the hidden variables in our model in the form of posterior distributions over those hidden variables. Hidden variables are partitioned into two categories: *global* hidden variables and *local* hidden variables, which we denote $\theta$ and $z_{1:N}$, respectively. Most people are familiar with global hidden variables. These are variables that we assume govern all $N$ elements of our observed data. Models containing local hidden variables are often called "Latent Variable Models". This entire exposition is all to say that this implementation (inspired by the paper "Weight Uncertainty in Neural Networks") only deals with global variable models. For example, the supervised setting where we map inputs $x_{1:N}$ to outputs $y_{1:N}$ with a single neural network. Each "weight" in the neural network is a global variable because it governs all $y_{1:N}$ in the same way.

The reason for this lengthy preface is that a lot of resources on variational inference (the focus of this repository) speak in terms of both local and global hidden variables, and how we treat them during inference is different.

# Variational Inference: Motivation

We have a probabilistic model of our data $y_{1:N}$:

$$
p(\theta, y | x) = \underbrace{p(y|\theta, x)}_{\text{observation model}} \overbrace{p(\theta)}^{\text{prior}}
$$

We'd like to compute the posterior $p(\theta|y_{1:N}, x_{1:N})$. To do so we use Bayes' theorem,

$$
p(\theta | y_{1:N}, x_{1:N}) = \frac{p(\theta, y_{1:N}|x_{1:N})}{\underbrace{p(y_{1:N}|x_{1:N})}_{\text{normalizing constant}}}
$$

The normalizing constant involves a large multi-dimensional integration which is generally intractable. Variational inference turns this "integration problem" into an "optimization problem" which is much easier (computing derivatives is fairly easy, integration if very hard). 


# Variational Inference: How It's Done

From now on I will suppress the $(\cdot)_{1:N}$ to unclutter notation. 

The basic premise of variational inference is to first propose a _variational family_ $\mathcal{Q}$ of _variational distributions_ $q$ over the hidden variables -- in our case just the global hidden variable $\theta$. In *mean-field* variational inference this variational distribution is indexed by _variational parameters_ $\gamma$, so we have $q_{\gamma}(\theta)$. We then minimize the KL divergence between this distribution and the true posterior $p(\theta|y, x)$ to learn the variational parameters $\gamma$:

$$
\underset{\gamma}{\text{argmax }} \text{KL}[q_{\gamma}(\theta)  ||  p(\theta|y, x)] \tag{1}
$$

We will see that we cannot directly do this because it ends up involving the computation of the "evidence" $p(y|x)$ (the quantity for which we appeal to approximate inference in the first place!). We instead optimize a related quantity termed the *evidence lower bound* (ELBO). To see why, we expand the KL divergence $(1)$, noting that we suppress $\gamma$ for conciseness,

$$
\begin{aligned}
\text{KL}[q(\theta)||p(\theta|y, x)] & = \mathbb{E}_{q(\theta)}[\text{log}  \\, q(\theta)] - \mathbb{E} _{q(\theta)} [\text{log}  \\, p(\theta|y, x)] \\
&= \mathbb{E} _{q(\theta)}[\text{log}  \\, q(\theta)] - \mathbb{E} _{q(\theta)} [\text{log}  \\, p(\theta, y| x)] + \underbrace{\text{log}  \\, p(y|x)} _{\text{intractable}} 
\end{aligned}
$$

So instead we optimize the ELBO, which is equivalent to the KL divergence term up to a constant. It is simply the KL divergence term without the intractable evidence, and then negated since we maximize the ELBO while we would minimize the KL divergence.

$$
\text{ELBO}(q(\theta)) = \mathbb{E} _{q(\theta)} [\text{log}  \\, p(\theta, y| x)] - \mathbb{E} _{q(\theta)}[\text{log}  \\, q(\theta)] 
$$

Further manipulation of the ELBO allows us to gather intuitive insights into how it will lead $q(\theta)$ to behave,

$$
\begin{aligned}
\text{ELBO}(q(\theta)) &= \mathbb{E} _{q(\theta)} [\text{log}  \\, p(\theta, y| x)] - \mathbb{E} _{q(\theta)}[\text{log}  \\, q(\theta)]  \\
&= \mathbb{E} _{q(\theta)}[\text{log}  \\, p(y|x, \theta)] + \mathbb{E} _{q(\theta)}[\text{log} \\, p(\theta)]  - \mathbb{E} _{q(\theta)}[\text{log}  \\, q(\theta)] \\
& = \underbrace{\mathbb{E} _{q(\theta)}[\text{log}  \\, p(y|x, \theta)]} _{\text{Expected data loglikelihood}} - \underbrace{\text{KL}[q(\theta)||p(\theta)]} _{\text{Relative entropy}}
\end{aligned}
$$

The expected data log likelihood term encourages $q(\theta)$ to place its probability so as to explain the observed data, while the relative entropy term encourages $q(\theta)$ to be close to the prior $p(\theta)$; it keeps $q(\theta)$ from collapsing to a distribution with a single point mass. For most $p$ and $q$, the KL divergence in the relative entropy term is analytically intractable, in which case we can resort to the approximation 

$$
\text{KL}[q(\theta)||p(\theta)] = \mathbb{E} _{q(\theta)}[q(\theta) - p(\theta)].
$$

# Mean-field Variational Inference

This repository implements a particular form of variational inference, often referred to as mean-field variational inference. *But be careful!* The formulation of the mean-field family and how one optimizes the variational parameters depends on whether the variational distribution is over the local hidden variables or global hidden variables. For the local hidden variable formulation, see [(Margossian et. al, 2023)](https://arxiv.org/abs/2307.11018). For the global variable case, however, mean-field variational inference is often referred to as selecting the following variational family (of distributions) over the global hidden variables ([(Coker et. al, 2021)](https://arxiv.org/pdf/2106.07052.pdf) & [(Foong et. al, 2020)](https://proceedings.neurips.cc/paper_files/paper/2020/file/b6dfd41875bc090bd31d0b1740eb5b1b-Paper.pdf)):

$$
\mathcal{Q} = \\{ q  :  q(\theta) = \prod_{i=1}^{|\theta|} \mathcal{N}(\theta_i  |  \mu_i, \sigma^2_i) \\} .
$$ 

In other words, the family of multivariate Gaussians with diagonal covariance (also called a fully factorized Gaussian). Some have questioned the expressivity of the mean-field family, and whether it can capture the complex dependencies in a high-dimensional target posterior distribution. For instance, [(Foong et. al, 2020)](https://proceedings.neurips.cc/paper_files/paper/2020/file/b6dfd41875bc090bd31d0b1740eb5b1b-Paper.pdf) look at the failure modes of mean-field variational inference in shallow neural networks. On the other hand, [(Farquhar et. al)](https://oatml.cs.ox.ac.uk/blog/2020/11/29/liberty_or_depth.html) argue that with large neural networks, mean-field variational inference is sufficient. 

# The Reparameterization Trick

We would like to take the gradient of the ELBO with respect to the variational parameters $\gamma$,

$$
\nabla_\gamma \mathbb{E}_ {q_ \gamma(\theta)}[\overbrace{\text{log} \\, p(\theta, y|x) - \text{log} \\, q_ \gamma(\theta)}^{f_ \gamma(\theta)}]. \tag{2}
$$ 



## Preliminary: Monte Carlo Integration

[Monte Carlo integration](https://en.wikipedia.org/wiki/Monte_Carlo_method) allows us to get an unbiased approximation of an expectation of a function by sampling from the distribution the expectation is with respect to:

$$
\mathbb{E}_ {p_ \theta} [f(x)] \approx \frac{1}{N} \sum_{n=1}^N f(x^{(n)}) \\; \text{ with } \\; x^{(n)} \sim p_ \theta(x)
$$

## The General Way (Not Reparameterization Trick)

Let us expand $(2)$,

$$
\begin{align}
\nabla_\gamma \mathbb{E}_ {q_ \gamma(\theta)}[\overbrace{\text{log} \\, p(\theta, y|x) - \text{log} \\, q_ \gamma(\theta)}^{f_ \gamma(\theta)}] &= \nabla_ {\gamma} \int \text{d}\theta \\, q_ {\gamma}(\theta) f_ \gamma(\theta) \\
&\overset{(i)}{=} \int \text{d}\theta \\,  \\{ (\nabla_\gamma q_ \gamma(\theta)) f_ \gamma(\theta) + q_ \gamma(\theta)(\nabla \gamma f_ \gamma(\theta) \\} \\
&\overset{(ii)}{=} \underbrace{\int \text{d}\theta \\, ( \overbrace{\nabla_\gamma q_ \gamma(\theta)}^{\text{not a density}})f_ \gamma(\theta)}_ {\text{cannot monte carlo}} + \underbrace{\int \text{d}\theta \\, q_ \gamma(\theta)(\nabla_\gamma \\, f_ \gamma(\theta)}_{\text{can monte carlo}} \tag{3}
\end{align}
$$

where in $(i)$ we use the chain rule as well as the [Leibniz rule](https://en.wikipedia.org/wiki/Leibniz_integral_rule) to push the derivative inside the integral and in $(ii)$ we simply split the integral. As noted, the first integral in the last line cannot be approximated via Monte Carlo so we have a problem. We need to somehow mold the expression so that we can express it as $\mathbb{E}_ {q_ \gamma(\theta)} [\dots]$. Here is how we do it:

1. Note the identity: $\nabla_\gamma \\, q_ \gamma(\theta) = q_ \gamma(\theta) \nabla_\gamma \\, \text{log} \\, q_ \gamma(\theta)$
2. Plug the new expression for $\nabla_\gamma \\, q_ \gamma(\theta)$ into $(3)$, factor and rearrange to get $\mathbb{E}_ {q_ \gamma(\theta)} [\dots]$...

$$
\begin{align}
\int \text{d}\theta \\, (\nabla_\gamma \\, q_ \gamma(\theta))f_ \gamma(\theta) + \int \text{d}\theta \\, q_ \gamma(\theta)(\nabla_\gamma f_ \gamma(\theta) &= \int \text{d}\theta \\, q_ \gamma(\theta)(\nabla_\gamma \text{log} \\, q_ \gamma(\theta) ) \\, f_ \gamma(\theta) + \int \text{d}\theta \\, q_ \gamma(\theta)(\nabla_\gamma f_ \gamma(\theta) \\
&= \int \text{d}\theta \\, q_ \gamma(\theta) \\{  (\nabla_\gamma \text{log} \\, q_ \gamma(\theta) ) \\, f_ \gamma(\theta) + \nabla_\gamma f_ \gamma(\theta)   \\} \\
& = \mathbb{E}_ {q_ \gamma(\theta)} [(\nabla_\gamma \text{log} \\, q_ \gamma(\theta) ) \\, f_ \gamma(\theta) + \nabla_\gamma f_ \gamma(\theta)] \tag{4}
\end{align}
$$

We can use Monte Carlo to approximate $(4)$ now! This type of estimator for the gradient of the ELBO is called by many names: the score function estimator, the REINFORCE estimator, the likelihood ratio estimator. Unfortunately, this estimator can have severely high variance and in some cases is even unusable. Fortunately, for some types of variational distributions (e.g. Gaussian), we can use the *reparameterization trick* to come up with an estimator with drastically better variance. 

## The Reparameterization Trick (For Real This Time)
Let's recall what we want to do: like in $(2)$ we want to take the following derivative

$$
\nabla_ \gamma \mathbb{E}_ {q_ \gamma(\theta)} [f_ \gamma (\theta)]. \tag{5}
$$

We just saw how we can find a Monte Carlo estimator for this but the variance of such an estimator can be unusably high. The reparameterization trick ends up making the estimator have much less variance; however, it requires more assumptions (i.e. specific forms of variational distribution). The idea of reparameterization is that we can come up with an equivalent representation of a random quantity but this new representation allows us to do cool and good things. So suppose we can express $\theta \sim q_ \gamma(\theta)$ as a *deterministic* variable $\theta = g_ \gamma(\epsilon)$ where $\epsilon$ is an auxiliary variable with independent marginal distribution $p(\epsilon)$ and $g_ \gamma(\cdot)$ is some deterministic function parameterized by $\gamma$. For example, we can actually do this with a normally distributed random variable: 

$$
\begin{align}
&\theta \sim q_ \gamma(\theta) = \text{Normal}(\theta|\gamma) = \text{Normal}(\theta|\mu, \sigma^2) \\
&\downarrow\\
&\theta = \mu + \sigma \epsilon \\; \text{ where } \\; \epsilon \sim p(\epsilon) = \text{Normal}(\epsilon|0, 1). \tag{6}
\end{align}
$$ 

Now why is this useful? Well let us go through taking the derivative in $(5)$ but with the assumption that we can reparameterize $\theta$ as a deterministic function of an auxiliary variable governed by independent marginal distribution:

$$
\begin{align}
\nabla_ \gamma \mathbb{E}_ {q_ \gamma(\theta)} [f_ \gamma (\theta)] &\overset{(i)}{=} \nabla_ \gamma \mathbb{E}_ {p(\epsilon)} [f_ \gamma(g_ \gamma(\epsilon))] \\
&= \nabla_ \gamma \int \text{d}{\epsilon} \\, p(\epsilon) \\{ f_ \gamma(g_ \gamma(\epsilon)) \\} \\
&= \int \text{d}{\epsilon} \\, p(\epsilon) \\{\nabla_ \gamma f_ \gamma(g_ \gamma(\epsilon)) \\} \\
&= \mathbb{E}_ {p(\epsilon)}[\nabla_ \gamma f_ \gamma(g_ \gamma(\epsilon))] \tag{7}
\end{align}
$$

We can use Monte Carlo to approximate $(7)$ and has [smaller variance](https://arxiv.org/pdf/1809.10330.pdf) than the score function estimator we saw earlier! For our use case, because we have a variational family of diagonal multivariate Gaussian distributions we can indeed employ the reparameterization trick. 

# Practical Framing of Implementation
We use the reparameterization trick in this repository because we can; namely, we choose $q(\theta)$ to be a multivariate normal distribution which is amenable to the reparameterization trick. Here we frame all the work we have done above that best resembles the implementation in this repository. First, we formulate the ELBO such that it has the KL divergence term in it. We have written this already in $(2)$, but now we make the variational parameters explicit:


$$
\text{ELBO}(q_ \gamma(\theta)) = \underbrace{\mathbb{E} _{q _\gamma(\theta)}[\text{log}  \\, p(y|x, \theta)]} _{\text{Expected data loglikelihood}} - \underbrace{\text{KL}[q _\gamma(\theta)||p(\theta)]} _{\text{Relative entropy}}
$$

From the reparameterization trick we have $\theta = g _\gamma(\epsilon)$, and as a consequence the ELBO becomes:

$$
\begin{align}
\text{ELBO}(q_ \gamma(\theta)) &= \mathbb{E} _{q _\gamma(\theta)}[\text{log}  \\, p(y|x, \theta)] - \text{KL}[q _\gamma(\theta)||p(\theta)] \\
\overset{\text{reparam trick}}{\rightarrow} \\; \\; \\; &= \mathbb{E} _{p(\epsilon)} [\text{log}  \\, p(y|x, g _\gamma(\epsilon)] - \text{KL}[q(g _\gamma(\epsilon))||p(g _\gamma(\epsilon))]
\end{align}
$$

Sometimes, given the form chosen for $q(\theta)$ and $p(\theta)$, we can analytically compute the KL divergence term and takes its derivative. In other cases, we cannot. 

## Case 1: KL term is not analytically tractable



# Resources

[Monte Carlo Gradient Estimation in Machine Learning](https://arxiv.org/pdf/1906.10652.pdf)






