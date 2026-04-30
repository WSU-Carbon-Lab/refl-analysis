import ollama

def extract_parameters(obj, exclude_prefixes=("scale_", "theta_")):
    """
    Extract parameters from an objective object.

    Parameters
    ----------
    obj : Objective
        The objective object containing parameters
    exclude_prefixes : tuple
        Parameter name prefixes to exclude

    Returns
    -------
    dict
        Dictionary with parsed parameter information
    """
    params_dict = {}
    non_standard_params = []

    for p in obj.varying_parameters():
        name = p.name

        if any(name.startswith(prefix) for prefix in exclude_prefixes):
            continue

        value = p.value
        error = p.stderr if hasattr(p, 'stderr') and p.stderr else 0.0
        bounds = [p.bounds.lb, p.bounds.ub] if hasattr(p, 'bounds') and p.bounds else [None, None]

        parts = name.split('_')

        if len(parts) >= 3:
            layer = parts[0]
            energy = parts[1]
            param_name = '_'.join(parts[2:])

            key = (layer, energy, param_name)
            params_dict[key] = {
                'name': name,
                'layer': layer,
                'energy': energy,
                'param_name': param_name,
                'value': value,
                'error': error,
                'bounds': bounds
            }
        else:
            non_standard_params.append({
                'name': name,
                'value': value,
                'error': error,
                'bounds': bounds
            })

    return params_dict, non_standard_params


def format_parameters_for_ollama(params_dict, non_standard_params, model_name):
    """
    Format parameters into a readable string for Ollama.

    Parameters
    ----------
    params_dict : dict
        Dictionary of standard format parameters
    non_standard_params : list
        List of non-standard format parameters
    model_name : str
        Name of the model

    Returns
    -------
    str
        Formatted parameter string
    """
    lines = [f"Model: {model_name}", "=" * 80, ""]

    if non_standard_params:
        lines.append("Non-standard format parameters (do not follow <layer>_<energy>_<param> format):")
        for p in non_standard_params:
            lines.append(f"  {p['name']}: value={p['value']:.6e} +/- {p['error']:.6e}, bounds={p['bounds']}")
        lines.append("")

    layers = sorted(set(k[0] for k in params_dict.keys()))
    energies = sorted(set(k[1] for k in params_dict.keys()), key=float)

    for layer in layers:
        lines.append(f"\nLayer: {layer}")
        for energy in energies:
            layer_energy_params = {k: v for k, v in params_dict.items()
                                 if k[0] == layer and k[1] == energy}
            if layer_energy_params:
                lines.append(f"  Energy: {energy} eV")
                for (l, e, pname), param in sorted(layer_energy_params.items(), key=lambda x: x[0][2]):
                    lines.append(f"    {pname}: value={param['value']:.6e} +/- {param['error']:.6e}, "
                               f"bounds=[{param['bounds'][0]:.6e}, {param['bounds'][1]:.6e}]")

    return "\n".join(lines)


def create_comparison_prompt(params1_str, params2_str, model1_name, model2_name):
    """
    Create a comprehensive prompt for Ollama to compare two parameter sets.

    Parameters
    ----------
    params1_str : str
        Formatted parameters for model 1
    params2_str : str
        Formatted parameters for model 2
    model1_name : str
        Name of model 1
    model2_name : str
        Name of model 2

    Returns
    -------
    str
        Complete prompt for Ollama
    """
    prompt = f"""You are analyzing X-ray reflectivity fitting parameters from two different models. Compare and contrast the two parameter sets below.

Focus on:
1. What parameters are being varied in each model
2. Differences in parameter values, uncertainties, and bounds
3. Notable patterns in index of refraction parameters (diso, biso, bire, dichro)
4. Differences in structural parameters (density, thickness, roughness)
5. Any parameters that do not follow the standard <layer>_<energy>_<parameter> format
6. Physical significance of the differences
7. Which model appears more constrained or has better determined parameters

Parameter naming convention:
- Standard format: <layer name>_<energy in eV>_<parameter name>
- Parameters include:
  * diso: isotropic delta (real part of index of refraction)
  * biso: isotropic beta (imaginary part of index of refraction)
  * bire: birefringence (anisotropy in real part)
  * dichro: dichroism (anisotropy in imaginary part)
  * density: material density
  * thick: layer thickness
  * rough: surface roughness
  * rho: scattering length density
  * rotation: molecular rotation angle

MODEL 1: {model1_name}
{params1_str}

MODEL 2: {model2_name}
{params2_str}

Provide a detailed comparison covering:
- Summary of what differs between the models
- Do the models have the same number of parameters? Or are there extra parameters in
one model and not the other? What are the extra parameters?
- Are there any significant differences in the parameter values?
"""
    return prompt


def compare_parameters_with_ollama(obj1, obj2, model1_name, model2_name, model="llama3.2"):
    """
    Compare two parameter sets using Ollama.

    Parameters
    ----------
    obj1 : Objective
        First objective object
    obj2 : Objective
        Second objective object
    model1_name : str
        Name of first model
    model2_name : str
        Name of second model
    model : str
        Ollama model to use

    Returns
    -------
    str
        Comparison analysis from Ollama
    """
    params1_dict, non_std1 = extract_parameters(obj1)
    params2_dict, non_std2 = extract_parameters(obj2)

    params1_str = format_parameters_for_ollama(params1_dict, non_std1, model1_name)
    params2_str = format_parameters_for_ollama(params2_dict, non_std2, model2_name)

    prompt = create_comparison_prompt(params1_str, params2_str, model1_name, model2_name)

    response = ollama.chat(model=model, messages=[
        {
            'role': 'user',
            'content': prompt
        }
    ])

    return response['message']['content']
