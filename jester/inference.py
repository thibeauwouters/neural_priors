"""
Full-scale inference: we will use jim as flowMC wrapper
"""

################
### PREAMBLE ###
################
import os 
import time
import shutil
import numpy as np
np.random.seed(43) # for reproducibility
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jimgw.prior import UniformPrior, CombinePrior
from jimgw.jim import Jim
import utils
import utils_plotting
import argparse

print(f"GPU found?")
print(jax.devices())

################
### Argparse ###
################

def parse_arguments():
    parser = argparse.ArgumentParser(description="Full-scale inference script with customizable options.")
    parser.add_argument("--make-cornerplot", 
                        type=bool, 
                        default=True, 
                        help="Whether to make the cornerplot. Turn off by default since can be expensive in memory.")
    parser.add_argument("--which-NEP-prior", 
                        type=str, 
                        default="default",
                        choices=["default", "small"],
                        help="Which NEP prior to sample from. If `small` then we do not include 3rd and 4th order and use a smaller range for L_sym.")
    parser.add_argument("--crust-name", 
                        type=str, 
                        default="DH",
                        choices=["DH", "BPS", "DH_fixed"],
                        help="Which filename for the crust to choose.")
    parser.add_argument("--which-nbreak-prior", 
                        type=str, 
                        default="normal", 
                        help="Which EOS prior to sample from. If set to 'broad', the broader nbreak prior will be used.")
    parser.add_argument("--sample-GW170817", 
                        type=bool, 
                        default=False, 
                        help="Whether to sample the GW170817 event")
    parser.add_argument("--sample-GW190425", 
                        type=bool, 
                        default=False, 
                        help="Whether to sample the GW190425 event")
    parser.add_argument("--sample-GW231109",
                        type=bool, 
                        default=False, 
                        help="Whether to sample the GW231109 event")
    parser.add_argument("--GW231109-NF-filepath",
                        type=str, 
                        default="./NFs/GW231109/model.eqx", 
                        help="The name of the NF file to load and use for GW231109 -- so essentially which posterior")
    parser.add_argument("--sample-GW170817-injection", 
                        type=bool, 
                        default=False, 
                        help="Whether to sample the GW170817-like injection")
    parser.add_argument("--use-GW170817-posterior-Hauke", 
                        type=bool, 
                        default=False, 
                        help="Whether to use the NF trained on the posterior samples of the GW170817 analysis by Koehn+")
    parser.add_argument("--use-GW170817-posterior-agnostic-prior", 
                        type=bool, 
                        default=False, 
                        help="Whether to use the NF trained on the posterior samples of the GW170817 analysis with an agnostic lambdas prior")
    parser.add_argument("--use-GW170817-posterior-eos-prior", 
                        type=bool, 
                        default=False, 
                        help="Whether to use the NF trained on the posterior samples of the GW170817 analysis with an EOS-informed lambdas prior")
    parser.add_argument("--use-binary-Love", 
                        type=bool, 
                        default=False, 
                        help="Whether to sample the GW170817 event for which we used the Binary Love relations")
    parser.add_argument("--sample-J0030", 
                        type=bool, 
                        default=False, 
                        help="Whether to sample the J0030 event")
    parser.add_argument("--sample-J0740", 
                        type=bool, 
                        default=False, 
                        help="Whether to sample the J0740 event")
    parser.add_argument("--sample-radio", 
                        type=bool, 
                        default=False, 
                        help="Whether to sample the radio timing mass measurement pulsars. Do all of them at once.")
    parser.add_argument("--sample-NICER-masses", 
                        type=bool, 
                        default=False, 
                        help="If set to True, then we sample the NICER masses as well instead of integrating up to MTOV")
    parser.add_argument("--sample-PREX", 
                        type=bool, 
                        default=False, 
                        help="Whether to sample PREX data")
    parser.add_argument("--sample-CREX", 
                        type=bool, 
                        default=False, 
                        help="Whether to sample CREX data")
    parser.add_argument("--sample-chiEFT", 
                        type=bool, 
                        default=False, 
                        help="Whether to sample chiEFT data")
    parser.add_argument("--use-zero-likelihood", 
                        type=bool, 
                        default=False, 
                        help="Whether to use a mock log-likelihood which constantly returns 0")
    parser.add_argument("--outdir", 
                        type=str, 
                        default="./outdir/", 
                        help="Directory to save output files (default: './outdir/')")
    parser.add_argument("--N-samples-EOS", 
                        type=int, 
                        default=10_000,
                        help="Number of samples for which the TOV equations are solved")
    parser.add_argument("--nb-cse", 
                        type=int, 
                        default=8, 
                        help="Number of CSE grid points (excluding the last one at the end, since its density value is fixed, we do add the cs2 prior separately.)")
    parser.add_argument("--sampling-seed", 
                        type=int, 
                        default=11,
                        help="Number of CSE grid points (excluding the last one at the end, since its density value is fixed, we do add the cs2 prior separately.)")
    ### flowMC/Jim hyperparameters
    parser.add_argument("--n-loop-training", 
                        type=int, 
                        default=60,
                        help="Number of flowMC training loops.)")
    parser.add_argument("--n-loop-production", 
                        type=int, 
                        default=30,
                        help="Number of flowMC production loops.)")
    parser.add_argument("--eps-mass-matrix", 
                        type=float, 
                        default=3e-5,
                        help="Overall scaling factor for the step size matrix for MALA.")
    parser.add_argument("--n-local-steps", 
                        type=int, 
                        default=50,
                        help="Number of local steps to perform.")
    parser.add_argument("--n-global-steps", 
                        type=int, 
                        default=50,
                        help="Number of global steps to perform.")
    parser.add_argument("--n-epochs", 
                        type=int, 
                        default=20,
                        help="Number of epochs for NF training.")
    parser.add_argument("--n-chains", 
                        type=int, 
                        default=1000,
                        help="Number of MCMC chains to evolve.")
    parser.add_argument("--train-thinning", 
                        type=int, 
                        default=1,
                        help="Thinning factor before feeding samples to NF for training.")
    parser.add_argument("--output-thinning", 
                        type=int, 
                        default=5,
                        help="Thinning factor before saving samples.")
    return parser.parse_args()

def main(args):
    
    NMAX_NSAT = 25
    NB_CSE = args.nb_cse

    ### NEP priors
    K_sat_prior = UniformPrior(150.0, 300.0, parameter_names=["K_sat"])
    Q_sat_prior = UniformPrior(-500.0, 1100.0, parameter_names=["Q_sat"])
    Z_sat_prior = UniformPrior(-2500.0, 1500.0, parameter_names=["Z_sat"])

    E_sym_prior = UniformPrior(28.0, 45.0, parameter_names=["E_sym"])
    if args.which_NEP_prior == "small":
        max_L_sym = 100.0
    else:
        max_L_sym = 200.0
    print(f"We are using the {args.which_NEP_prior} NEP prior, so max_L_sym = {max_L_sym}")
    L_sym_prior = UniformPrior(10.0, max_L_sym, parameter_names=["L_sym"])
    K_sym_prior = UniformPrior(-300.0, 100.0, parameter_names=["K_sym"])
    Q_sym_prior = UniformPrior(-800.0, 800.0, parameter_names=["Q_sym"])
    Z_sym_prior = UniformPrior(-2500.0, 1500.0, parameter_names=["Z_sym"])

    if args.which_NEP_prior == "small":
        prior_list = [
            E_sym_prior,
            L_sym_prior, 
            K_sym_prior,
            Q_sym_prior,
            Z_sym_prior,

            K_sat_prior,
            Q_sat_prior,
            Z_sat_prior,
        ]
    else:
        prior_list = [
            E_sym_prior,
            L_sym_prior, 
            K_sym_prior,

            K_sat_prior,
        ]

    ### CSE priors
    if NB_CSE > 0:
        if args.which_nbreak_prior == "broad":
            print(f"Using the broad nbreak prior: U[1.0, 4.0] * 0.16")
            nbreak_prior = UniformPrior(1.0 * 0.16, 4.0 * 0.16, parameter_names=[f"nbreak"])
        else:
            print(f"Using the regular nbreak prior: U[1.0, 2.0] * 0.16")
            nbreak_prior = UniformPrior(1.0 * 0.16, 2.0 * 0.16, parameter_names=[f"nbreak"])
        prior_list.append(nbreak_prior)
        for i in range(NB_CSE):
            # NOTE: the density parameters are sampled from U[0, 1], so we need to scale it, but it depends on break so will be done internally
            prior_list.append(UniformPrior(0.0, 1.0, parameter_names=[f"n_CSE_{i}_u"]))
            prior_list.append(UniformPrior(0.0, 1.0, parameter_names=[f"cs2_CSE_{i}"]))

        # Final point to end
        prior_list.append(UniformPrior(0.0, 1.0, parameter_names=[f"cs2_CSE_{NB_CSE}"]))

    # Construct the EOS prior and a transform here which can be used down below for creating the EOS plots after inference is completed
    eos_prior = CombinePrior(prior_list)
    eos_param_names = eos_prior.parameter_names
    all_output_keys = ["logpc_EOS", "masses_EOS", "radii_EOS", "Lambdas_EOS", "n", "p", "h", "e", "dloge_dlogp", "cs2"]
    name_mapping = (eos_param_names, all_output_keys)
    
    # This transform will be the same as my_transform, but with different output keys, namely, all EOS related quantities, for postprocessing
    if args.nb_cse > 0:
        keep_names = ["E_sym", "L_sym", "nbreak"]
    else:
        keep_names = ["E_sym", "L_sym"]
    my_transform_eos = utils.MicroToMacroTransform(name_mapping,
                                                   keep_names=keep_names,
                                                   nmax_nsat=NMAX_NSAT,
                                                   nb_CSE=NB_CSE,
                                                   crust_name=args.crust_name,
                                                )
    
    # Create the output directory if it does not exist
    outdir = args.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        
    # Copy this script to the output directory, for reproducibility later on
    shutil.copy(__file__, os.path.join(outdir, "backup_inference.py"))
    
    # First, add mass priors if toggled (GW170817 by default, for NICER we can choose)
    keep_names = ["E_sym", "L_sym"]
    
    # Sample GW170817 PE
    if args.sample_GW170817:
        m1_GW170817_prior = UniformPrior(1.5, 2.1, parameter_names=["mass_1_GW170817"])
        m2_GW170817_prior = UniformPrior(1.0, 1.5, parameter_names=["mass_2_GW170817"])

        prior_list.append(m1_GW170817_prior)
        prior_list.append(m2_GW170817_prior)
        
        keep_names += ["mass_1_GW170817", "mass_2_GW170817"]
    
    # Sample GW190425 PE
    if args.sample_GW190425:
        m1_GW190425_prior = UniformPrior(1.1, 2.0, parameter_names=["mass_1_GW190425"])
        m2_GW190425_prior = UniformPrior(1.2, 1.8, parameter_names=["mass_2_GW190425"])

        prior_list.append(m1_GW190425_prior)
        prior_list.append(m2_GW190425_prior)
        
        keep_names += ["mass_1_GW190425", "mass_2_GW190425"]
        
    # Sample GW231109 PE
    if args.sample_GW231109:
        if ("et" in args.GW231109_NF_filepath.lower()) or ("ce" in args.GW231109_NF_filepath.lower()):
            print("Sampling a 3G-like injection of GW231109")
            m1_GW231109_prior = UniformPrior(1.42, 1.62, parameter_names=["mass_1_GW231109"])
            m2_GW231109_prior = UniformPrior(1.26, 1.45, parameter_names=["mass_2_GW231109"])
        else:
            m1_GW231109_prior = UniformPrior(1.3, 1.9, parameter_names=["mass_1_GW231109"])
            m2_GW231109_prior = UniformPrior(1.1, 1.5, parameter_names=["mass_2_GW231109"])
        
        prior_list.append(m1_GW231109_prior)
        prior_list.append(m2_GW231109_prior)
        
        keep_names += ["mass_1_GW231109", "mass_2_GW231109"]
        
    # TODO: add G1124251 here
    if args.sample_J0030 and args.sample_NICER_masses:
        prior_list += [UniformPrior(1.0, 2.0, parameter_names=["mass_J0030"])]
        keep_names += ["mass_J0030"]
        
    if args.sample_J0740 and args.sample_NICER_masses:
        prior_list += [UniformPrior(1.5, 2.5, parameter_names=["mass_J0740"])]
        keep_names += ["mass_J0740"]
    
    ##################
    ### LIKELIHOOD ###
    ##################

    # Likelihood: choose which PSR(s) to perform inference on:
    if not args.use_zero_likelihood:
        
        # GW
        likelihoods_list_GW = []
        if args.sample_GW170817:
            # NOTE: this is a slightly older NF so the default kwargs are ok
            likelihoods_list_GW += [utils.GWlikelihood_with_masses("GW170817", "./NFs/GW170817/model.eqx")]
        
        if args.sample_GW190425:
            # NOTE: this is a slightly newer NF so use these updated kwargs FIXME: these are hardcoded -- this should change in future editions
            likelihoods_list_GW += [utils.GWlikelihood_with_masses("GW190425", "./NFs/GW190425/model.eqx", nn_depth=6, nn_block_dim=16)]
        
        if args.sample_GW231109:
            # NOTE: this is a slightly newer NF so use these updated kwargs FIXME: these are hardcoded -- this should change in future editions
            nn_depth = 6
            nn_block_dim = 16
            
            likelihoods_list_GW += [utils.GWlikelihood_with_masses("GW231109", args.GW231109_NF_filepath, nn_depth=nn_depth, nn_block_dim=nn_block_dim)]
            
        # NICER
        likelihoods_list_NICER = []
        if args.sample_J0030:
            if args.sample_NICER_masses:
                print(f"Loading data necessary for the event J0030 and sampling the with NICER masses")
                likelihoods_list_NICER += [utils.NICERLikelihood_with_masses("J0030")]
            
            else:
                print(f"Loading data necessary for the event J0030")
                likelihoods_list_NICER += [utils.NICERLikelihood("J0030")]
        
        if args.sample_J0740:
            if args.sample_NICER_masses:
                print(f"Loading data necessary for the event J0740 and sampling the with NICER masses")
                likelihoods_list_NICER += [utils.NICERLikelihood_with_masses("J0740")]
            
            else:
                print(f"Loading data necessary for the event J0740")
                likelihoods_list_NICER += [utils.NICERLikelihood("J0740")]

        # Radio timing mass measurement pulsars
        likelihoods_list_radio = []
        if args.sample_radio:
            likelihoods_list_radio += [utils.RadioTimingLikelihood("J1614", 1.94, 0.06)]
            # likelihoods_list_radio += [utils.RadioTimingLikelihood("J0348", 2.01, 0.08)]
            if not args.sample_J0740:
                likelihoods_list_radio += [utils.RadioTimingLikelihood("J0740", 2.08, 0.14)]
            else:
                print("NOTE: Not adding the radio timing for J0740 since we also sample the NICER result -- this already has this in the prior")

        # PREX and CREX
        likelihoods_list_REX = []
        if args.sample_PREX:
            print(f"Loading data necessary for PREX")
            likelihoods_list_REX += [utils.REXLikelihood("PREX")]
        if args.sample_CREX:
            print(f"Loading data necessary for CREX")
            likelihoods_list_REX += [utils.REXLikelihood("CREX")]
            
        if len(likelihoods_list_REX) == 0:
            print(f"Not sampling PREX or CREX data now")
            
        # Chiral EFT
        likelihoods_list_chiEFT = []
        # FIXME: only add chiEFT if we are sampling MM+CSE, ignore during MM-only
        if args.sample_chiEFT and args.nb_cse > 0:
            keep_names += ["nbreak"]
            print(f"Loading data necessary for the Chiral EFT")
            likelihoods_list_chiEFT += [utils.ChiEFTLikelihood()]

        # Total likelihoods list:
        likelihoods_list = likelihoods_list_GW + likelihoods_list_NICER + likelihoods_list_radio + likelihoods_list_REX + likelihoods_list_chiEFT
        print(f"Sanity checking: likelihoods_list = {likelihoods_list}\nlen(likelihoods_list) = {len(likelihoods_list)}")
        likelihood = utils.CombinedLikelihood(likelihoods_list)
        
    # Construct the transform object
    TOV_output_keys = ["masses_EOS", "radii_EOS", "Lambdas_EOS"]
    prior = CombinePrior(prior_list)
    sampled_param_names = prior.parameter_names
    name_mapping = (sampled_param_names, TOV_output_keys)
    my_transform = utils.MicroToMacroTransform(name_mapping,
                                               keep_names = keep_names,
                                               nmax_nsat = NMAX_NSAT,
                                               nb_CSE = NB_CSE,
                                               )
    
    if args.use_zero_likelihood:
        print("Using the zero likelihood:")
        likelihood = utils.ZeroLikelihood(my_transform)

    mass_matrix = jnp.eye(prior.n_dim)
    local_sampler_arg = {"step_size": mass_matrix * args.eps_mass_matrix}
    kwargs = {"n_loop_training": args.n_loop_training,
            "n_loop_production": args.n_loop_production,
            "n_chains": args.n_chains,
            "n_local_steps": args.n_local_steps,
            "n_global_steps": args.n_global_steps,
            "n_epochs": args.n_epochs,
            "train_thinning": args.train_thinning,
            "output_thinning": args.output_thinning,
            "local_sampler_name": "GaussianRandomWalk"
    }
    
    print("We are going to give these kwargs to Jim:")
    print(kwargs)
    
    print("We are going to sample the following parameters:")
    print(prior.parameter_names)

    # Define the Jim object here
    jim = Jim(likelihood,
              prior,
              local_sampler_arg = local_sampler_arg,
              likelihood_transforms = [my_transform],
              **kwargs)

    # Test case
    samples = prior.sample(jax.random.PRNGKey(0), 3)
    samples_transformed = jax.vmap(my_transform.forward)(samples)
    log_prob = jax.vmap(likelihood.evaluate)(samples_transformed, {})
    
    print("log_prob")
    print(log_prob)
    
    # Do the sampling
    print(f"Sampling seed is set to: {args.sampling_seed}")
    start = time.time()
    jim.sample(jax.random.PRNGKey(args.sampling_seed))
    jim.print_summary()
    end = time.time()
    runtime = end - start

    print(f"Sampling has been successful, now we will do some postprocessing. Sampling time: roughly {int(runtime / 60)} mins")

    ### POSTPROCESSING ###
        
    # Training (just to count number of samples)
    sampler_state = jim.sampler.get_sampler_state(training=True)
    log_prob = sampler_state["log_prob"].flatten()
    nb_samples_training = len(log_prob)

    # Production (also for postprocessing plotting)
    sampler_state = jim.sampler.get_sampler_state(training=False)

    # Get the samples, and also get them as a dictionary
    samples_named = jim.get_samples()
    samples_named_for_saving = {k: np.array(v) for k, v in samples_named.items()}
    samples_named = {k: np.array(v).flatten() for k, v in samples_named.items()}
    keys, samples = list(samples_named.keys()), np.array(list(samples_named.values()))

    # Get the log prob, also count number of samples from it
    log_prob = np.array(sampler_state["log_prob"])
    log_prob = log_prob.flatten()
    nb_samples_production = len(log_prob)
    total_nb_samples = nb_samples_training + nb_samples_production
    
    # Save the final results
    print(f"Saving the final results")
    np.savez(os.path.join(outdir, "results_production.npz"), log_prob=log_prob, **samples_named_for_saving)

    print(f"Number of samples generated in training: {nb_samples_training}")
    print(f"Number of samples generated in production: {nb_samples_production}")
    print(f"Number of samples generated: {total_nb_samples}")
    
    # Save the runtime to a file as well
    with open(os.path.join(outdir, "runtime.txt"), "w") as f:
        f.write(f"{runtime}")

    # Generate the final EOS + TOV samples from the EOS parameter samples
    idx_1 = np.random.choice(np.arange(len(log_prob)), size=args.N_samples_EOS, replace=False)
    idx_2 = np.random.choice(np.arange(len(log_prob)), size=args.N_samples_EOS, replace=False)
    
    chosen_samples_test = {k: jnp.array(v[idx_1]) for k, v in samples_named.items()}
    chosen_samples = {k: jnp.array(v[idx_2]) for k, v in samples_named.items()}
    # NOTE: jax lax map helps us deal with batching, but a batch size multiple of 10 gives errors, therefore this weird number
    # transformed_samples = jax.lax.map(jax.jit(my_transform_eos.forward), chosen_samples, batch_size = 4_999)
    
    # First do a single batch to jit compile, then do compiled vmap to get the timing right
    my_forward = jax.jit(my_transform_eos.forward)
    transformed_samples_test = jax.vmap(my_forward)(chosen_samples_test)
    
    TOV_start = time.time()
    transformed_samples = jax.vmap(my_forward)(chosen_samples)
    TOV_end = time.time()
    print(f"Time taken for TOV map: {TOV_end - TOV_start} s")
    chosen_samples.update(transformed_samples)

    log_prob = log_prob[idx_2]
    np.savez(os.path.join(args.outdir, "eos_samples.npz"), log_prob=log_prob, **chosen_samples)
    
    if args.make_cornerplot:
        try:    
            utils_plotting.plot_corner(outdir, samples, keys)
        except Exception as e:
            print(f"Could not make the corner plot, because of the following error: {e}")
    
    print("DONE entire script")
    
if __name__ == "__main__":
    args = parse_arguments()  # Get command-line arguments
    main(args)