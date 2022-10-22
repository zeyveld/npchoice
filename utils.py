import jax.numpy as jnp
import lcc_logit.config as config
from jax import random, jit
from jax.ops import segment_prod
from functools import partial
from gc import collect


class bind(partial):
    """
    An improved version of partial which accepts ellipsis (...) as a placeholder
    """

    def __call__(self, *args, **keywords):
        keywords = {**self.keywords, **keywords}
        iargs = iter(args)
        args = (next(iargs) if arg is ... else arg for arg in self.args)
        return self.func(*args, *iargs, **keywords)


def gen_true_coeffs(key, num_agents):
    """Randomly generate true coefficients of each agent from mixture of two
    multivariate normal distributions.
    Args:
        key (Jax PRNGKey): pseudo-random number generator (PRNG) key
    Returns:
        Array: True coefficients of each agent. Has dimension
            (config.num_agents, config.num_random_coeffs)
    """

    def gen_normal_dist(key):
        """Generate a multivariate normal distribution
        Args:
            key (PRNGKey): pseudo-random number generator (PRNG) key
        Returns:
            Dict: Mean and covariance of multivariate normal distribution
        """
        mean_coeffs = random.uniform(
            key, (1, config.num_random_coeffs), minval=-3, maxval=3
        )
        runiform_square = random.uniform(
            key,
            (config.num_random_coeffs, config.num_random_coeffs),
            minval=-0.9,
            maxval=0.9,
        )
        cov_coeffs = jnp.dot(runiform_square, runiform_square.T)
        return {"mean_coeffs": mean_coeffs, "cov_coeffs": cov_coeffs}

    mixture_components_dict = {
        component: gen_normal_dist(key) for component in range(2)
    }
    mixture_weight0 = random.uniform(key)
    if_mixture0 = jnp.ravel(random.uniform(key, (num_agents,)) > mixture_weight0)
    potential_coeffs_dict = {
        component: random.multivariate_normal(
            key,
            mixture_components_dict[component]["mean_coeffs"],
            mixture_components_dict[component]["cov_coeffs"],
            shape=(num_agents,),
        )
        for component in range(2)
    }
    return (
        if_mixture0[:, None] * potential_coeffs_dict[0]
        + (1 - if_mixture0)[:, None] * potential_coeffs_dict[1]
    )


def synthesize_data(
    key,
    num_agents,
    num_choices_per_agent,
    num_alternatives,
    num_random_coeffs,
    prob_out_of_stock,
):
    """Sythesize explanatory and explained variables, as well as product
    availability
    Args:
        key (PRNGKey): pseudo-random number generator (PRNG) key
        mean_coeffs (array): Vector of
    """
    config.num_alternatives = num_alternatives
    config.num_random_coeffs = num_random_coeffs
    availability = (
        random.uniform(
            key, (num_agents * num_choices_per_agent, config.num_alternatives)
        )
        > prob_out_of_stock
    )
    meaningful_choices = jnp.count_nonzero(availability, axis=1) > 1
    availability = availability[meaningful_choices]
    explanatory_vars = random.normal(
        key,
        (
            num_agents * num_choices_per_agent,
            config.num_alternatives,
            config.num_random_coeffs,
        ),
    )[meaningful_choices]
    agent_ids_by_choice = jnp.repeat(jnp.arange(num_agents), num_choices_per_agent)[
        meaningful_choices
    ]
    agent_ids, indices_of_agent_ids, config.num_choices_of_each_agent = jnp.unique(
        agent_ids_by_choice, return_index=True, return_counts=True
    )
    config.num_choices, config.num_agents = jnp.sum(meaningful_choices), agent_ids.size

    true_agent_coeffs = gen_true_coeffs(key, num_agents)
    true_agent_coeffs_by_choice = jnp.repeat(
        true_agent_coeffs, config.num_choices_of_each_agent, 0
    )
    true_agent_coeffs = true_agent_coeffs_by_choice[indices_of_agent_ids]

    linear_index = jnp.einsum(
        "njk,nk -> nj", explanatory_vars, true_agent_coeffs_by_choice
    )
    utility = linear_index + random.normal(
        key, (config.num_choices, config.num_alternatives)
    )
    explained_var = (
        jnp.zeros(utility.shape, dtype=int)
        .at[jnp.arange(len(utility)), jnp.nanargmax(utility, axis=1)]
        .set(1)
    )
    availability = availability[explained_var == 0].reshape(
        config.num_choices, config.num_alternatives - 1
    )
    return {
        "explanatory_vars": explanatory_vars,
        "explained_var": explained_var,
        "availability": availability,
        "agent_ids_by_choice": agent_ids_by_choice,
        "agent_ids": agent_ids,
        "true_agent_coeffs": true_agent_coeffs,
    }


def bfgs_alg(
    loglik_fn,
    individual_class_coeffs,
    args,
    **options,
):
    """BFGS optimization routine."""
    res, g, grad_n = loglik_fn(individual_class_coeffs, *args)
    Hinv = jnp.linalg.pinv(jnp.dot(grad_n.T, grad_n))
    step_tol_failed = False
    nit, nfev, njev = 0, 1, 1
    while True:
        old_g = g
        d = -Hinv.dot(g)
        step = 2
        while True:
            step = step / 2
            s = step * d
            resnew, *_ = loglik_fn(individual_class_coeffs + s, *args)
            nfev += 1
            if step > options["step_tol"]:
                if resnew <= res or step < options["step_tol"]:
                    individual_class_coeffs += s
                    resnew, gnew, grad_n = loglik_fn(
                        individual_class_coeffs,
                        *args,
                    )
                    njev += 1
                    break
            else:
                step_tol_failed = True
                break
        nit += 1

        if step_tol_failed:
            break
        old_res = res
        res = resnew
        g = gnew
        gproj = jnp.abs(jnp.dot(d, old_g))

        if (
            (gproj < options["gtol"])
            or (jnp.abs(res - old_res) < options["tol"])
            or (nit > options["clogit_maxiter"])
        ):
            break
        delta_g = g - old_g

        Hinv = (
            Hinv
            + (
                (
                    (s.dot(delta_g) + (delta_g[None, :].dot(Hinv)).dot(delta_g))
                    * jnp.outer(s, s)
                )
                / (s.dot(delta_g)) ** 2
            )
            - (
                (jnp.outer(Hinv.dot(delta_g), s) + (jnp.outer(s, delta_g)).dot(Hinv))
                / (s.dot(delta_g))
            )
        )
    Hinv = jnp.linalg.pinv(jnp.dot(grad_n.T, grad_n))
    return individual_class_coeffs


@jit
def individual_class_ccprobs_etc(
    individual_class_coeffs, delta_explanatory_vars, availability
):
    """Compute conditional choice probabilities for an individual class.
    Args:
        individual_class_coeffs (array): Vector of coefficients (for an
            individual class). Has dimension (config.num_random_coeffs,)
        delta_explanatory_vars (array): Matrix of differences in explanatory
            variables between the alternative in question and the one ACTUALLY
                CHOSEN. Has dimension
                (config.num_agents, config.num_alternatives - 1, config.num_random_coeffs)
        availability (array): Vector of product availability. Has dimension
            (config.num_agents, config.num_alternatives - 1)
    Returns:
        Array: exponentiated sums of linear index differences. Has dimension
            (config.num_agents, config.num_alternatives-1)
        Array: Conditional choice probabilities of chosen alternatives. Has
            dimension (config.num_agents, config.num_alternatives - 1)
        Array: Log of conditional choice probabilities of chosen alternatives.
            Has dimension (config.num_agents, config.num_alternatives - 1)
    """
    delta_linear_index = jnp.einsum(
        "njk,k -> nj", delta_explanatory_vars, individual_class_coeffs
    )
    exponentiated_delta_linear_index = (
        jnp.exp(delta_linear_index) * availability
    )  # Exponentiated differences in representative utilities
    ccprobs = 1 / (
        1 + exponentiated_delta_linear_index.sum(axis=1)
    )  # Chosen alts' conditional choice probs
    return exponentiated_delta_linear_index, ccprobs, jnp.log(ccprobs)


@jit
def individual_class_loglik_and_grad(
    individual_class_coeffs,
    individual_class_weights,
    delta_explanatory_vars,
    availability,
):
    """Compute log-likelihood, gradient, and hessian.
    Args:
        individual_class_coeffs (array): Vector of coefficients for an individual class
        individual_class_weights (array): Vector of conditional class
        membership probabilities, one per choice instance. Has dimension (config.num_agents * T,)
        delta_explanatory_vars (array):  Matrix of differences in explanatory variables between the
            alternative in question and the one ACTUALLY CHOSEN. Has dimension
            (config.num_agents, config.num_alternatives - 1, config.num_random_coeffs)
        availability (array): Vector of product availability. Has dimension (config.num_agents, config.num_alternatives - 1)
    Returns:
        Float: Negative log likelihood value
        Array: Gradient of log likelihood function. Has dimension (config.num_random_coeffs,)
        Array: Gradient of log likelihood function by agent. Has dimension
            (config.num_agents, config.num_random_coeffs)
    """
    (
        exponentiated_delta_linear_index,
        ccprobs,
        log_ccprobs,
    ) = individual_class_ccprobs_etc(
        individual_class_coeffs, delta_explanatory_vars, availability
    )
    # Log likelihood
    loglik = jnp.sum(log_ccprobs * individual_class_weights)
    output = (-loglik,)
    # Individual contribution to the gradient
    grad_n = (
        -jnp.einsum(
            "njk,nj -> nk", delta_explanatory_vars, exponentiated_delta_linear_index
        )
        * ccprobs[:, None]
        * individual_class_weights[:, None]
    )
    grad = jnp.sum(grad_n, axis=0)  # Average gradient across agents
    output += (-grad.ravel(),)
    output += (grad_n,)
    return output
    # return output if len(output) > 1 else output[0]


@jit
def compute_kernels(coeffs, delta_explanatory_vars, availability, agent_ids_by_choice):
    """Compute conditional choice probabilities for an individual class.
    Args:
        coeffs: Matrix containing ALL classes' coefficients from
            the preceding round of the EM algorithm. Has dimension
            (config.num_latent_classes, config.num_random_coeffs)
        shares (array): Vector of aggregate class shares. Has dimension (config.num_latent_classes,)
        delta_explanatory_vars (array): Matrix of differences in explanatory variables between the
            alternative in question and the one ACTUALLY CHOSEN. Has dimension
            (config.num_agents, config.num_alternatives - 1, config.num_random_coeffs)
        availability (array): Vector of product availability. Has dimension (config.num_agents, config.num_alternatives - 1)
    Returns:
        Array: Kernels by agent and latent class. Has dimension (config.num_agents, config.num_latent_classes)
    """
    delta_linear_index = jnp.einsum("njk,kc->njc", delta_explanatory_vars, coeffs.T)
    exponentiated_delta_linear_index = jnp.einsum(
        "njc,nj->njc", jnp.exp(delta_linear_index), availability
    )
    ccprobs = 1 / (
        1 + exponentiated_delta_linear_index.sum(axis=1)
    )  # Chosen alts' conditional choice probs
    return jit(segment_prod, static_argnums=2)(
        ccprobs, agent_ids_by_choice, config.num_agents
    )


@jit
def update_weights(
    coeffs,
    shares,
    delta_explanatory_vars,
    availability,
    agent_ids_by_choice,
):
    """Update conditional class membership probabilities of all classes.
    Args:
        coeffs: Matrix containing ALL classes' coefficients from
            the preceding round of the EM algorithm. Has dimension (config.num_latent_classes, config.num_random_coeffs)
        shares (array): Vector of aggregate class shares. Has dimension (config.num_latent_classes,)
        delta_explanatory_vars (array): Matrix of differences in explanatory variables between the
            alternative in question and the one ACTUALLY CHOSEN. Has dimension
            (config.num_agents, config.num_alternatives - 1, config.num_random_coeffs)
        availability (array): Vector of product availability. Has dimension (config.num_agents, config.num_alternatives - 1)
    Returns:
        array: Matrix of conditional class membership probabilities (one
            per agent). Has dimension (config.num_agents, config.num_latent_classes)
        array: Matrix of conditional class membership probabilities (one
            per choice instance. Has dimension (config.num_agents * T, config.num_latent_classes)
    """
    kernels = compute_kernels(
        coeffs, delta_explanatory_vars, availability, agent_ids_by_choice
    )
    weighted_kernels = jnp.einsum("nc,c->nc", kernels, shares)
    weights_by_agent = jnp.einsum(
        "nc,n->nc", weighted_kernels, 1 / jnp.sum(weighted_kernels, axis=1)
    )
    print(f"In update_weights(): {weights_by_agent[:10]}")
    return weights_by_agent, jnp.repeat(
        weights_by_agent, config.num_choices_of_each_agent, 0
    )


@jit
def update_shares(weights_by_agent):
    """Update aggregate class membership probabilties.
    Args:
        weights_by_agent (array): Matrix of conditional class membership
            probabilities (one per agent). Has dimension (config.num_agents,config.num_latent_classes)
    Returns:
        array: Vector of aggregate class shares. Has dimension (config.num_latent_classes,)
    """
    return jnp.mean(weights_by_agent, 0)


def update_individual_class_coeffs(
    individual_class_init_coeffs,
    individual_class_weights_by_choice,
    delta_explanatory_vars,
    availability,
    **options,
):
    """Fit conditional logit model.
    Args:
        individual_class_init_coeffs (array): Vector containing the class's coefficients from the
            preceding round of the EM algorithm. Has dimension (config.num_random_coeffs,)
        individual_class_weights_by_choice (array): Vector of conditional class membership
            probabilities, one per choice instance. Has dimension (config.num_choices,)
        delta_explanatory_vars (array): Matrix of differences in explanatory variables between the
            alternative in question and the one ACTUALLY CHOSEN. Has dimension
            (config.num_choices, config.num_alternatives - 1, config.num_random_coeffs)
        availability (array): Vector of product availability. Has dimension (config.num_choices, config.num_alternatives - 1)
    Returns:
        Array: Vector of coefficients for given class. Has dimension (config.num_random_coeffs,)
    """
    fargs = (individual_class_weights_by_choice, delta_explanatory_vars, availability)
    return bfgs_alg(
        individual_class_loglik_and_grad,
        individual_class_init_coeffs,
        args=fargs,
        **options,
    )


def update_coeffs(
    init_coeffs,
    weights_by_choice,
    delta_explanatory_vars,
    availability,
    **options,
):
    """Update the coefficients of each class (contained in a matrix)
    Args:
        init_coeffs (array): Matrix containing ALL classes' coefficients from
            the preceding round of the EM algorithm. Has dimension (config.num_latent_classes, config.num_random_coeffs)
        weights_by_choice (array): Vector of conditional class membership
            probabilities, one per choice instance. Has dimension (config.num_latent_classes, config.num_agents * T)
        delta_explanatory_vars: Matrix of explanatory variables. Has dimension (config.num_agents, config.num_alternatives - 1, config.num_random_coeffs)
        availability (array): Array of product availability. Has dimension (config.num_agents, config.num_alternatives - 1)
    Returns:
        Array: Matrix containing ALL classes' updated coefficients. Has
            dimension (config.num_latent_classes, config.num_random_coeffs)
    """
    updated_coeffs_list = [
        update_individual_class_coeffs(
            individual_class_init_coeffs=init_coeffs[latent_class],
            individual_class_weights_by_choice=weights_by_choice[:, latent_class],
            delta_explanatory_vars=delta_explanatory_vars,
            availability=availability,
            **options,
        )
        for latent_class in range(config.num_latent_classes)
    ]
    return jnp.array(updated_coeffs_list)


def compute_loglik_val(
    coeffs,
    shares,
    delta_explanatory_vars,
    availability,
    **options,
):
    """Compute log likelihood value, given present class coefficients and shares
    Args:
        coeffs (array): Matrix of coefficients by class. Has dimension (config.num_latent_classes, config.num_random_coeffs)
        shares (array): Vector of aggregate class shares. Has dimension (config.num_latent_classes,)
        delta_explanatory_vars (array): Matrix of explanatory variables. Has dimension (config.num_agents, config.num_alternatives - 1, config.num_random_coeffs)
        availability (array): Array of product availability. Has dimension (config.num_agents, config.num_alternatives - 1)
    Returns:
        Float: Log likelihood value
    """
    log_ccprobs_list = []
    for latent_class in range(config.num_latent_classes):
        log_ccprobs_list.append(
            individual_class_ccprobs_etc(
                coeffs[latent_class], delta_explanatory_vars, availability
            )[-1]
        )
    log_ccprobs = jnp.array(log_ccprobs_list)
    kernels = jnp.exp(log_ccprobs)
    class_shares_weighted_kernels = jnp.average(kernels, axis=0, weights=shares)
    return jnp.sum(jnp.log(class_shares_weighted_kernels))


def em_alg(
    coeffs,
    shares,
    delta_explanatory_vars,
    availability,
    agent_ids_by_choice,
    **options,
):
    """Run EM algorithm, which goes as follows:
        (1) update agents' conditional class membership probabilities;
        (2) update aggregate class shares;
        (3) update class coefficients;
    Args:
        coeffs (array): Matrix of coefficients by class. Has dimension (config.num_latent_classes, config.num_random_coeffs)
        shares (array): Vector of aggregate class shares. Has dimension (config.num_latent_classes,)
        delta_explanatory_vars (array): Matrix of explanatory variables. Has dimension (config.num_agents, config.num_alternatives - 1, config.num_random_coeffs)
        availability (array): Array of product availability. Has dimension (config.num_agents, config.num_alternatives - 1)
        loglik_val_lag5 (float): Log likelihood value from five iterations ago
    Returns:
        Array: Updated vector of aggregate class shares. Has dimension (config.num_latent_classes,)
        Array: Updated matrix of coefficients by class. Has dimension (config.num_latent_classes, config.num_random_coeffs)

    """
    updated_weights_by_agent, updated_weights_by_choice = update_weights(
        coeffs=coeffs,
        shares=shares,
        delta_explanatory_vars=delta_explanatory_vars,
        availability=availability,
        agent_ids_by_choice=agent_ids_by_choice,
    )
    print(f"In em_alg(): {updated_weights_by_agent[:10]}")
    updated_shares = update_shares(updated_weights_by_agent)
    updated_coeffs = update_coeffs(
        init_coeffs=coeffs,
        weights_by_choice=updated_weights_by_choice,
        delta_explanatory_vars=delta_explanatory_vars,
        availability=availability,
        **options,
    )
    loglik_val = compute_loglik_val(
        coeffs=updated_coeffs,
        shares=updated_shares,
        delta_explanatory_vars=delta_explanatory_vars,
        availability=availability,
    )
    return updated_coeffs, updated_shares, loglik_val
