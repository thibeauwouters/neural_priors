#!/usr/bin/env python3
"""
Critical inspection of the saved FlowJAX model to verify if bijections are actually saved.
"""

import os
import sys
import json
import jax
import jax.numpy as jnp
import jax.random as jr
from flowjax.flows import block_neural_autoregressive_flow
from flowjax.distributions import Normal, Transformed
from flowjax import bijections as bij
import equinox as eqx

def print_tree_structure(obj, name="", indent=0, max_depth=10):
    """
    Recursively print the structure of a PyTree object.
    """
    if indent > max_depth:
        print("  " * indent + "... (max depth reached)")
        return
        
    obj_type = type(obj).__name__
    print("  " * indent + f"{name}: {obj_type}")
    
    # Look for key attributes that would indicate bijections
    if hasattr(obj, '__dict__'):
        for attr_name, attr_value in obj.__dict__.items():
            if attr_name.startswith('_'):
                continue
            print_tree_structure(attr_value, f"{name}.{attr_name}", indent + 1, max_depth)
    
    # For FlowJAX objects, also check for specific patterns
    if hasattr(obj, 'bijection') or hasattr(obj, 'bijections'):
        print("  " * indent + f"*** FOUND BIJECTION ATTRIBUTE ***")
    
    # Check if this looks like a Sigmoid or Affine transformation
    if 'Sigmoid' in obj_type or 'Affine' in obj_type or 'SoftPlus' in obj_type:
        print("  " * indent + f"*** FOUND TRANSFORMATION: {obj_type} ***")

def examine_model_leaves(model_path):
    """
    Load model and examine all leaves to look for transformation components.
    """
    print(f"CRITICAL INSPECTION OF: {model_path}")
    print("=" * 80)
    
    # Load model kwargs
    kwargs_path = os.path.join(model_path, "model_kwargs.json")
    with open(kwargs_path, "r") as f:
        nf_kwargs = json.load(f)
    
    model_file = os.path.join(model_path, "model.eqx")
    
    print("1. EXAMINING RAW SAVED FILE STRUCTURE")
    print("-" * 40)
    
    # Create the base flow architecture to deserialize
    n_inputs = nf_kwargs["n_inputs"]
    base_dist = Normal(jnp.zeros(n_inputs))
    key = jr.key(42)
    
    # Try to load as different possible structures
    unbounded_flow = block_neural_autoregressive_flow(
        key=key,
        base_dist=base_dist,
        nn_depth=nf_kwargs["nn_depth"],
        nn_block_dim=nf_kwargs["nn_block_dim"],
        flow_layers=nf_kwargs["flow_layers"],
    )
    
    print(f"Base unbounded flow type: {type(unbounded_flow)}")
    
    # Load the actual saved model
    try:
        loaded_model = eqx.tree_deserialise_leaves(model_file, unbounded_flow)
        print(f"✓ Loaded model type: {type(loaded_model)}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    print("\n2. DEEP STRUCTURE ANALYSIS")
    print("-" * 40)
    print_tree_structure(loaded_model, "loaded_model", max_depth=8)
    
    print("\n3. SEARCHING FOR TRANSFORMATION COMPONENTS")
    print("-" * 40)
    
    # Use JAX tree utilities to find all leaves and their types
    leaves, treedef = jax.tree_util.tree_flatten(loaded_model)
    
    print(f"Total leaves in model: {len(leaves)}")
    print(f"Tree structure: {treedef}")
    
    # Look for any objects that might be transformations
    found_transformations = []
    for i, leaf in enumerate(leaves):
        leaf_type = type(leaf).__name__
        leaf_str = str(leaf)
        
        # Check for transformation keywords
        transform_keywords = ['Sigmoid', 'Affine', 'SoftPlus', 'Chain', 'Stack', 'bijection']
        if any(keyword.lower() in leaf_type.lower() or keyword.lower() in leaf_str.lower() 
               for keyword in transform_keywords):
            found_transformations.append((i, leaf_type, leaf_str[:100]))
    
    if found_transformations:
        print("*** FOUND POTENTIAL TRANSFORMATIONS ***")
        for i, leaf_type, leaf_str in found_transformations:
            print(f"  Leaf {i}: {leaf_type}")
            print(f"    String repr: {leaf_str}")
    else:
        print("*** NO TRANSFORMATIONS FOUND IN MODEL LEAVES ***")
        print("This strongly suggests the model is NOT constrained!")
    
    print("\n4. ATTEMPTING TO RECONSTRUCT EXPECTED CONSTRAINED MODEL")
    print("-" * 40)
    
    # Create what the constrained model SHOULD look like
    param_names = nf_kwargs.get("names", [])
    bijection_list = []
    
    for i, name in enumerate(param_names):
        if "mass_ratio" in name or "q" in name:
            scale_shift = bij.Affine(loc=0.1, scale=0.9)
            bijection = bij.Chain([bij.Sigmoid(shape=()), scale_shift])
            bijection_list.append(bijection)
            
        elif "lambda" in name and name != "delta_lambda_tilde":
            bijection = bij.SoftPlus(shape=())
            bijection_list.append(bijection)
            
        else:
            bijection = bij.Identity(shape=())
            bijection_list.append(bijection)
    
    to_constrained = bij.Stack(bijection_list)
    expected_constrained = Transformed(unbounded_flow, to_constrained)
    
    # Check if saved model matches expected structure
    try:
        expected_leaves, expected_treedef = jax.tree_util.tree_flatten(expected_constrained)
        print(f"Expected constrained model leaves: {len(expected_leaves)}")
        print(f"Actual loaded model leaves: {len(leaves)}")
        
        if len(expected_leaves) == len(leaves):
            print("✓ Leaf count matches - model MIGHT be constrained")
        else:
            print("✗ Leaf count mismatch - model is likely NOT constrained")
            print(f"  Expected: {len(expected_leaves)} leaves")
            print(f"  Actual: {len(leaves)} leaves")
            print("  This suggests bijections were NOT saved!")
            
    except Exception as e:
        print(f"Failed to create expected model: {e}")
    
    print("\n5. FINAL VERDICT")
    print("-" * 40)
    
    if found_transformations:
        print("✓ TRANSFORMATIONS FOUND - Model appears constrained")
    else:
        print("✗ NO TRANSFORMATIONS FOUND - Model is NOT constrained")
        print("  The training code may have a bug where bijections aren't saved")
        print("  Or the constrained training path wasn't actually executed")
    
    return loaded_model

def main():
    if len(sys.argv) != 2:
        print("Usage: python inspect_saved_model.py <model_path>")
        sys.exit(1)
        
    model_path = sys.argv[1]
    examine_model_leaves(model_path)

if __name__ == "__main__":
    main()