

import pandas as pd


def obtain_molecules_metrics(molecules):

    # Calculate number of molecules
    num_molecules = molecules['MOLECULE_ID'].nunique()

    # Calculate mean switching cycles per molecule
    mean_switching_cycles = molecules['#_TRACKS'].mean()

    # Calculate mean on time per molecule
    mean_on_time = molecules['TOTAL_ON_TIME'].mean()

    # Calculate how many molecules are bleached
    bleached = molecules[molecules['BLEACHED'] == True]
    survival_fraction = (num_molecules - len(bleached)) / num_molecules

    # Calculate Duty Cycle
    duty_cycle = mean_on_time / 10000

    # Create metrics dataframe
    metrics = pd.DataFrame({
        'Molecules': [num_molecules],
        'Switching Cycles per Mol': [mean_switching_cycles],
        'On Time per SC': [mean_on_time],
        'Survival Fraction': [survival_fraction],
        'Duty Cycle': [duty_cycle]
    })

    return metrics



