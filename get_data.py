import pandas as pd
import jax.numpy as jnp
import lcc_logit.config as config


def get_data(product_category, num_alternatives, num_random_coeffs):
    """Describe here"""

    def get_individual_dataset(
        product_category, desired_data, num_alternatives, num_random_coeffs
    ):
        """Import .csv file from 'data' folder as Jax Numpy array"""
        desired_data_df = pd.read_csv(
            f"/mnt/research/conlin-lab/andrewworkspace/state_dependence/data/{product_category}/{desired_data}.csv"
        )
        desired_data_array = jnp.array(desired_data_df.to_numpy())

        if desired_data in (
            "agent_ids_by_choice",
            "agent_ids",
            "num_choices_of_each_agent",
        ):
            return jnp.ravel(desired_data_array)
        elif desired_data == "explanatory_vars":
            long_explanatory_vars = jnp.nan_to_num(desired_data_array)
            return long_explanatory_vars.reshape(
                -1, num_alternatives, num_random_coeffs
            )
        elif desired_data == "explained_var":
            return desired_data_array.reshape(-1, num_alternatives)
        elif desired_data == "availability":
            return desired_data_array.reshape(-1, num_alternatives - 1)

    data_dict = {
        desired_data: get_individual_dataset(
            product_category, desired_data, num_alternatives, num_random_coeffs
        )
        for desired_data in (
            "explanatory_vars",
            "explained_var",
            "availability",
            "agent_ids_by_choice",
            "agent_ids",
            "num_choices_of_each_agent",
        )
    }
    config.num_alternatives = num_alternatives
    config.num_random_coeffs = num_random_coeffs
    config.num_choices = len(data_dict["agent_ids_by_choice"])
    config.num_agents = len(data_dict["agent_ids"])
    config.num_choices_of_each_agent = data_dict["num_choices_of_each_agent"]
    print(f"Num alternatives: {config.num_alternatives}")
    print(f"Num random coefficients: {config.num_random_coeffs}")
    print(f"Num choices: {config.num_choices}")
    print(f"Num agents: {config.num_agents}")
    print(f"Explanatory vars dim: {data_dict['explanatory_vars'].shape}")
    print(f"Explained var dim: {data_dict['explained_var'].shape}")
    print(f"Availability dim: {data_dict['availability'].shape}")
    print(f"Agent ids by choice dim: {data_dict['agent_ids_by_choice'].shape}")
    print(f"Agent ids dim: {data_dict['agent_ids'].shape}")
    print(f"Num choices of each agent: {data_dict['num_choices_of_each_agent'].shape}")
    return data_dict
