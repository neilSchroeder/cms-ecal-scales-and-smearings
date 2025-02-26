import numpy as np

from python.classes.constant_classes import CategoryConstants as cc
from python.classes.constant_classes import DataConstants as dc


def target_function_wrapper(initial_guess, __ZCATS__, *args, **kwargs):
    """
    Wrapper for the target function. This is necessary to keep track of the previous guess, to eliminate redundant calculations.

    Args:
        __GUESS__ (iterable): iterable of floats, representing the initial guess for the scales and smearings
        __ZCATS__ (list): list of zcat objects, each representing a category
        **options: keyword arguments, which contain the following:
            num_scales (int): number of scales to be derived
            num_smears (int): number of smearings to be derived
    Returns:
        target_function (function): target function
    """

    previous_guess = [initial_guess]

    def wrapped_target_function(x, *args, **options):
        (previous, __ZCATS__, __num_scales__, __num_smears__) = args
        ret = target_function(
            x, previous_guess[0], __ZCATS__, __num_scales__, __num_smears__, **options
        )
        previous_guess[0] = x
        return ret

    def reset(x=None):
        previous_guess[0] = x if x is not None else initial_guess

    return wrapped_target_function, reset


def target_function(x, *args, verbose=False, **options):
    """
    This is the target function, which returns an event weighted -2*Delta NLL
    This function features a small verbose option for debugging purposes.
    target_function accepts an iterable of floats and uses them to evaluate the NLL in each category.
    Some 'smart' checks prevent the function from evaluating all N(N+1)/2 categories unless absolutely necessary.

    Args:
        x (iterable): iterable of floats, representing the scales and smearings chosen by the minimizer
        *args: a tuple of arguments, which contains the following:
            __GUESS__ (iterable): iterable of floats, representing the initial guess for the scales and smearings
            __ZCATS__ (list): list of zcat objects, each representing a category
            __num_scales__ (int): number of scales to be derived
            __num_smears__ (int): number of smearings to be derived
    """

    # unpack args
    (previous, __ZCATS__, __num_scales__, __num_smears__) = args

    # find where __GUESS__ and x differ
    # no use updating categories if they don't need to be updated
    updated_scales = [i for i in range(len(x)) if x[i] != previous[i]]

    # find all the categories that need to be updated
    mask = np.array(
        [
            cat.valid
            and (
                cat.lead_index in updated_scales
                or cat.sublead_index in updated_scales
                or cat.lead_smear_index in updated_scales
                or cat.sublead_smear_index in updated_scales
            )
            for cat in __ZCATS__
        ]
    )

    cats_to_update = np.array(__ZCATS__)[mask]

    active_indices = [
        (cat.lead_index, cat.sublead_index) for cat in __ZCATS__ if cat.valid
    ]

    # count how many times each scale is used
    scale_counts = np.zeros(len(x))
    for cat in cats_to_update:
        scale_counts[cat.lead_index] += 1
        scale_counts[cat.sublead_index] += 1

    if __num_smears__ > 0:
        for cat in cats_to_update:
            scale_counts[cat.lead_smear_index] += 1
            scale_counts[cat.sublead_smear_index] += 1

    # if any scale is not used at all, give the user a warning
    for i, count in enumerate(scale_counts):
        if count == 0:
            print(
                f"[WARNING][python/nll] scale {i} is not used in any category. This is likely a mistake."
            )

    # update the categories
    [
        (
            cat.update(x[cat.lead_index], x[cat.sublead_index])
            if __num_smears__ == 0
            else cat.update(
                x[cat.lead_index],
                x[cat.sublead_index],
                lead_smear=x[cat.lead_smear_index],
                sublead_smear=x[cat.sublead_smear_index],
            )
        )
        for cat in cats_to_update
    ]

    if verbose:
        print("------------- zcat info -------------")
        [cat.print() for cat in cats_to_update]
        print("-------------------------------------")
        print()

    tot = sum([cat.weight for cat in __ZCATS__ if cat.valid])
    ret = sum([cat.NLL * cat.weight for cat in __ZCATS__ if cat.valid])
    final_value = ret / tot if tot != 0 else 9e30

    if verbose:
        print("------------- total info -------------")
        # print("weighted nll:",ret/tot)
        print(
            "diagonal nll vals:",
            [
                cat.NLL * cat.weight / tot
                for cat in __ZCATS__
                if cat.lead_index == cat.sublead_index and cat.valid
            ],
        )
        print("using scales:", x)
        print("--------------------------------------")

    return final_value


def scan_nll(x, **options):
    """
    Performs the NLL scan to initialize the variables.

    Args:
        x (iterable): iterable of floats, representing the scales and smearings chosen by the minimizer
        **options: keyword arguments, which contain the following:
            __GUESS__ (iterable): iterable of floats, representing the initial guess for the scales and smearings
            __ZCATS__ (list): list of zcat objects, each representing a category
            _kFixScales (bool): whether or not to fix the scales
            num_scales (int): number of scales to be derived
            num_smears (int): number of smearings to be derived
            scan_min (float): minimum value for the scan
            scan_max (float): maximum value for the scan
            scan_step (float): step size for the scan
    Returns:
        guess (iterable): iterable of floats, representing the scales and smearings chosen by the minimizer
    """
    __ZCATS__ = options["zcats"]
    __GUESS__ = options["__GUESS__"]
    guess = x
    scanned = []

    # find most sensitive category and scan that first
    print("[INFO][python/helper_minimizer/scan_ll] scanning scales")
    weights = [
        (cat.weight, cat.lead_index)
        for cat in __ZCATS__
        if cat.valid and cat.lead_index == cat.sublead_index
    ]
    weights.sort(key=lambda x: x[0])
    loss_function, reset_loss_initial_guess = target_function_wrapper(
        guess, __ZCATS__, **options
    )

    if not options["_kFixScales"]:
        while weights:
            max_index = cc.empty
            tup = weights.pop(0)

            if tup[cc.i_eta_min] not in scanned:
                max_index = tup[cc.i_eta_min]
                scanned.append(tup[cc.i_eta_min])

            if max_index != cc.empty:
                x = np.arange(
                    options["scan_min"], options["scan_max"], options["scan_step"]
                )
                my_guesses = []

                # generate a few guesses
                for j, val in enumerate(x):
                    guess[max_index] = val
                    my_guesses.append(guess[:])

                # evaluate nll for each guess
                nll_vals = np.array(
                    [
                        loss_function(
                            g,
                            __GUESS__,
                            __ZCATS__,
                            options["num_scales"],
                            options["num_smears"],
                        )
                        for g in my_guesses
                    ]
                )
                mask = [
                    y > 0 for y in nll_vals
                ]  # addresses edge cases of scale being too large/small
                x = x[mask]
                nll_vals = nll_vals[mask]

                if len(nll_vals) > 0:
                    guess[max_index] = x[nll_vals.argmin()]
                    print(
                        "[INFO][python/nll] best guess for scale {} is {}".format(
                            max_index, guess[max_index]
                        )
                    )

    print("[INFO][python/helper_minimizer/scan_nll] scanning smearings:")
    scanned = []
    weights = [
        (cat.weight, cat.lead_smear_index)
        for cat in __ZCATS__
        if cat.valid and cat.lead_smear_index == cat.sublead_smear_index
    ]
    weights.sort(key=lambda x: x[0])

    low = 0.00025
    high = 0.025
    step = 0.00025
    x = np.arange(low, high, step)
    if options["num_smears"] > 0:
        while weights:
            max_index = cc.empty
            tup = weights.pop(0)

            if tup[cc.i_eta_min] not in scanned:
                max_index = tup[cc.i_eta_min]
                scanned.append(tup[cc.i_eta_min])

            # smearings are different, so use different values for low,high,step
            if max_index != cc.empty:
                my_guesses = []

                # generate a few guesses
                for j, val in enumerate(x):
                    guess[max_index] = val
                    my_guesses.append(guess[:])

                # evaluate nll for each guess
                nll_vals = np.array(
                    [
                        loss_function(
                            g,
                            __GUESS__,
                            __ZCATS__,
                            options["num_scales"],
                            options["num_smears"],
                        )
                        for g in my_guesses
                    ]
                )
                mask = [
                    y > 0 for y in nll_vals
                ]  # addresses edge cases of scale being too large/small
                x = x[mask]
                nll_vals = nll_vals[mask]
                if len(nll_vals) > 0:
                    guess[max_index] = x[nll_vals.argmin()]
                    print(
                        f"[INFO][python/nll] best guess for smearing {max_index} is {guess[max_index]}"
                    )

    print("[INFO][python/nll] scan complete")
    return guess
