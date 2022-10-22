import jax.numpy as jnp
import lcc_logit.config as config
from jax import random
from lcc_logit.utils import (
    update_weights,
    update_shares,
    em_alg,
)


def estimate(key, num_latent_classes, data_dict, **options):
    """Perform latent class conditional logit estimation.
    Args:
        key: (Jax PRNGKey): Pseudo-random number generator key
        num_latent_classes (int): Number of latent preference classes
        data_dict (dictionary): Dictionary containing choice data
    Returns:
        Array: Matrix of coefficients by latent class. Has dimension
            (config.num_random_coeffs, num_latent_classes)
        Array: Vector of aggregate latent class shares. Has dimension
            (num_latent_classes,)
        Float: Log-likelihood value
    """
    config.num_latent_classes = num_latent_classes
    default_options = {
        "em_maxiter": 2000,
        "em_loglik_tol": 0.0001,
        "clogit_maxiter": 2000,
        "tol": 1e-10,
        "gtol": 1e-6,
        "step_tol": 1e-10,
        "disp": False,
        "num_gpus": 1,
        # "num_gpus": local_device_count(),
    }
    options = {**default_options, **options}
    # Record differences in explanatory vars between non-chosen alternatives
    # and the chosen alternative
    delta_explanatory_vars = data_dict["explanatory_vars"][
        data_dict["explained_var"] == 0
    ].reshape(
        config.num_choices, config.num_alternatives - 1, config.num_random_coeffs
    ) - data_dict[
        "explanatory_vars"
    ][
        data_dict["explained_var"] == 1
    ].reshape(
        config.num_choices, 1, config.num_random_coeffs
    )
    coeffs = 5 * random.uniform(
        key, (config.num_latent_classes, config.num_random_coeffs)
    )
    weights_by_agent, _ = update_weights(
        coeffs,
        jnp.ones((config.num_latent_classes,)) / config.num_latent_classes,
        delta_explanatory_vars,
        data_dict["availability"],
        data_dict["agent_ids_by_choice"],
    )
    shares = update_shares(weights_by_agent)
    print(
        f"Absolute initial shares: {jnp.ones((config.num_latent_classes,)) / config.num_latent_classes}"
    )
    print(f"Starting Shares (from randomly picked initial coeffs):\n{shares}")
    loglik_val_list = []
    for em_recursion in range(options["em_maxiter"]):
        print(f"EM recursion: {em_recursion}")
        coeffs, shares, loglik_val = em_alg(
            coeffs=coeffs,
            shares=shares,
            delta_explanatory_vars=delta_explanatory_vars,
            availability=data_dict["availability"],
            agent_ids_by_choice=data_dict["agent_ids_by_choice"],
            **options,
        )
        print(f"Shares:\n{shares}")
        loglik_val_list.append(loglik_val)
        if em_recursion >= 4:
            if (loglik_val - loglik_val_list[-5]) / loglik_val_list[-5] <= options[
                "em_loglik_tol"
            ]:
                return coeffs, shares, loglik_val
    print("Convergence failed.")
    return coeffs, shares, loglik_val
