import numpy as np


def validate_inputs(qi, di, b, Dlim, IBU, MBU):
    return all(x >= 0 for x in [IBU, MBU]) and all(x > 0 for x in [qi, di, b, Dlim])


def modified_hyperbolic(time_array, qi, di, b, Dlim, IBU, MBU,
                        buildup_method='Linear', epsilon=1e-10):
    """
    Calculate daily production and monthly volumes using the Modified Hyperbolic
    Decline Model with an optional buildup phase.

    Returns (daily_production, monthly_volumes).
    """
    if not validate_inputs(qi, di, b, Dlim, IBU, MBU):
        print(f"Invalid inputs detected: qi={qi}, di={di}, b={b}, Dlim={Dlim}, IBU={IBU}, MBU={MBU}")
        return np.zeros_like(time_array), np.zeros_like(time_array)

    # Convert decline rates to nominal rates
    try:
        if di > 100:
            di = 99.99
        di_nominal = ((1 - 0.01 * di) ** (-b) - 1) / (12 * b)
        Dlim_nominal = ((1 - 0.01 * Dlim) ** (-b) - 1) / (12 * b)
        if np.isnan(di_nominal) or np.isinf(di_nominal):
            return np.zeros_like(time_array), np.zeros_like(time_array)
    except (ZeroDivisionError, OverflowError, ValueError):
        return np.zeros_like(time_array), np.zeros_like(time_array)

    # Calculate switch point
    try:
        qlim = qi * (Dlim_nominal / (di_nominal + epsilon)) ** (1.0 / b)
        tlim = ((qi / qlim) ** b - 1.0) / (b * (di_nominal + epsilon))
    except (ZeroDivisionError, OverflowError, ValueError):
        return np.zeros_like(time_array), np.zeros_like(time_array)

    if MBU == 0:
        t_buildup = np.array([])
        buildup_production = np.array([])
        t_post_buildup = time_array
        t_hyp = t_post_buildup[t_post_buildup < tlim]
        t_exp = t_post_buildup[t_post_buildup >= tlim]
    else:
        t_buildup = time_array[time_array <= MBU]
        t_post_buildup = time_array[time_array > MBU]
        t_hyp = t_post_buildup[t_post_buildup < tlim]
        t_exp = t_post_buildup[t_post_buildup >= tlim]

        if buildup_method == 'Flat':
            buildup_production = IBU + np.zeros_like(t_buildup)
        elif buildup_method == 'Linear':
            slope = (qi - IBU) / (MBU + epsilon)
            buildup_production = IBU + slope * t_buildup
        elif buildup_method == 'Exp':
            slope = np.log(qi / (IBU + epsilon)) / (MBU + epsilon)
            buildup_production = IBU * np.exp(slope * t_buildup)
        else:
            raise ValueError(f"Unknown buildup method: {buildup_method}")

    # Hyperbolic decline
    try:
        q_model_hyp = qi * (1.0 + b * di_nominal * (t_hyp - MBU)) ** (-1.0 / b)
    except (ZeroDivisionError, OverflowError, ValueError):
        q_model_hyp = np.zeros_like(t_hyp)

    # Exponential decline
    try:
        q_model_exp = qlim * np.exp(-Dlim_nominal * (t_exp - tlim))
    except (ZeroDivisionError, OverflowError, ValueError):
        q_model_exp = np.zeros_like(t_exp)

    # Smooth transition
    if len(t_exp) > 0 and len(t_hyp) > 0:
        time_fraction = (tlim - t_hyp[-1]) / (t_exp[0] - t_hyp[-1] + epsilon)
        q_model_exp[0] = time_fraction * q_model_hyp[-1] + (1 - time_fraction) * q_model_exp[0]

    production_post_buildup = np.concatenate((q_model_hyp, q_model_exp))
    daily_production = np.concatenate((buildup_production, production_post_buildup))
    monthly_volumes = daily_production * 30

    return daily_production, monthly_volumes


def generate_production_rates(df, headers, time_array, resource_type='Oil'):
    """Generate production rate arrays for each row in df."""
    productions = []
    prefix = {'Oil': 'Oil_Params_P50_', 'Gas': 'Gas_Params_P50_', 'Water': 'Water_Params_P50_'}
    if resource_type not in prefix:
        raise ValueError("resource_type must be 'Oil', 'Gas', or 'Water'.")
    pfx = prefix[resource_type]

    for idx in range(len(df)):
        qi = df.iloc[idx][df.columns.get_loc(f'{pfx}InitialProd')]
        di = df.iloc[idx][df.columns.get_loc(f'{pfx}DiCoefficient')]
        b = df.iloc[idx][df.columns.get_loc(f'{pfx}BCoefficient')]
        IBU = 0
        MBU = 0
        Dlim = 7

        if any(x < 0 for x in [qi, di, b, IBU, MBU, Dlim]):
            productions.append(np.full_like(time_array, np.nan))
            continue
        production = modified_hyperbolic(time_array, float(qi), float(di), float(b),
                                         float(Dlim), float(IBU), float(MBU))[0]
        productions.append(production)

    return productions


def detect_spurious_curves(time_array, productions, discontinuity_threshold=0.5,
                           buildup_threshold=24):
    """Return indices of spurious production curves."""
    spurious_indices = []
    epsilon = 1e-10

    for i, production in enumerate(productions):
        production = np.atleast_1d(production)

        if np.any(np.isnan(production)):
            spurious_indices.append(i)
            continue

        if len(production) > 1:
            if np.all(np.diff(production) > 0):
                spurious_indices.append(i)
                continue
        else:
            spurious_indices.append(i)
            continue

        if np.argmax(production) > buildup_threshold:
            spurious_indices.append(i)
            continue

        if np.any(production <= 0) or np.any(production > 1e20):
            spurious_indices.append(i)
            continue

        diffs = np.diff(production)
        if len(diffs) > 1:
            max_diff_index = np.argmax(diffs)
            if 0 < max_diff_index < len(production) - 1:
                pre_transition = production[max_diff_index]
                post_transition = production[max_diff_index + 1]
                discontinuity_ratio = abs(post_transition - pre_transition) / max(pre_transition, epsilon)
                if discontinuity_ratio > discontinuity_threshold:
                    spurious_indices.append(i)
                    continue

    return spurious_indices


def remove_spurious_curves(df, headers, time_array, discontinuity_threshold=0.4,
                           buildup_threshold=24, resource_type='Oil'):
    """Remove rows with spurious production curves."""
    productions = generate_production_rates(df, headers, time_array, resource_type)
    spurious_indices = detect_spurious_curves(time_array, productions,
                                              discontinuity_threshold, buildup_threshold)
    spurious_indices = df.index[spurious_indices]
    df_cleaned = df.drop(spurious_indices).reset_index(drop=True)
    print(f"Removed {len(spurious_indices)} spurious {resource_type.lower()} curves.")
    return df_cleaned


def generate_production_rates_testing(y_pred_denormalized, headers, time,
                                      resource_type='Oil', use_baseline=False):
    """Generate production rates from predictions (for testing/sensitivity)."""
    productions = []
    prefix = {'Oil': 'Oil_Params_P50_', 'Gas': 'Gas_Params_P50_', 'Water': 'Water_Params_P50_'}
    if resource_type not in prefix:
        raise ValueError("resource_type must be 'Oil', 'Gas', or 'Water'.")
    pfx = prefix[resource_type]

    suffix = '_baseline' if use_baseline else ''

    for idx in range(len(y_pred_denormalized)):
        try:
            qi = y_pred_denormalized.iloc[idx, headers.index(f'{pfx}InitialProd{suffix}')]
            di = y_pred_denormalized.iloc[idx, headers.index(f'{pfx}DiCoefficient{suffix}')]
            b = y_pred_denormalized.iloc[idx, headers.index(f'{pfx}BCoefficient{suffix}')]
            IBU = 0
            MBU = 0
            Dlim = 7

            if not validate_inputs(qi, di, b, Dlim, IBU, MBU):
                productions.append(np.zeros_like(time))
                continue

            production = modified_hyperbolic(time, qi, di, b, Dlim, IBU, MBU)[1]
            productions.append(production)
        except Exception as e:
            print(f"Error generating production rates: {e}")
            productions.append(np.zeros_like(time))

    return productions


# ── TensorFlow Versions (for custom loss functions) ─────────────────────────


def _get_tf():
    """Lazy-import TensorFlow so the module can be loaded without it."""
    import tensorflow as tf
    return tf


def validate_inputs_tf(qi, di, b, Dlim, IBU, MBU):
    tf = _get_tf()
    valid_ibu_mbu = tf.reduce_all(tf.greater_equal([IBU, MBU], 0))
    valid_params = tf.reduce_all(tf.greater([qi, di, b, Dlim], 0))
    return tf.logical_and(valid_ibu_mbu, valid_params)


def modified_hyperbolic_tf(time_array, qi, di, b, Dlim, IBU, MBU,
                           buildup_method='Linear', epsilon=1e-10):
    """TF version of modified_hyperbolic for use in custom loss functions."""
    tf = _get_tf()
    if not validate_inputs_tf(qi, di, b, Dlim, IBU, MBU):
        return tf.zeros_like(time_array, dtype=tf.float32)

    try:
        di_nominal = ((1 - 0.01 * di) ** (-b) - 1) / (12 * b)
        Dlim_nominal = ((1 - 0.01 * Dlim) ** (-b) - 1) / (12 * b)
    except (ZeroDivisionError, OverflowError, ValueError):
        return tf.zeros_like(time_array, dtype=tf.float32)

    try:
        qlim = qi * (Dlim_nominal / (di_nominal + epsilon)) ** (1.0 / b)
        tlim = ((qi / qlim) ** b - 1.0) / (b * (di_nominal + epsilon))
    except (ZeroDivisionError, OverflowError, ValueError):
        return tf.zeros_like(time_array, dtype=tf.float32)

    if MBU == 0:
        t_buildup = tf.constant([], dtype=tf.float32)
        buildup_production = tf.constant([], dtype=tf.float32)
        t_post_buildup = time_array
        t_hyp = t_post_buildup[t_post_buildup < tlim]
        t_exp = t_post_buildup[t_post_buildup >= tlim]
    else:
        t_buildup = time_array[time_array <= MBU]
        t_post_buildup = time_array[time_array > MBU]
        t_hyp = t_post_buildup[t_post_buildup < tlim]
        t_exp = t_post_buildup[t_post_buildup >= tlim]

        if buildup_method == 'Flat':
            buildup_production = IBU + tf.zeros_like(t_buildup, dtype=tf.float32)
        elif buildup_method == 'Linear':
            slope = (qi - IBU) / (MBU + epsilon)
            buildup_production = IBU + slope * t_buildup
        elif buildup_method == 'Exp':
            slope = tf.math.log(qi / (IBU + epsilon)) / (MBU + epsilon)
            buildup_production = IBU * tf.exp(slope * t_buildup)
        else:
            raise ValueError(f"Unknown buildup method: {buildup_method}")

    try:
        q_model_hyp = qi * (1.0 + b * di_nominal * (t_hyp - MBU)) ** (-1.0 / b)
    except (ZeroDivisionError, OverflowError, ValueError):
        q_model_hyp = tf.zeros_like(t_hyp, dtype=tf.float32)

    try:
        q_model_exp = qlim * tf.exp(-Dlim_nominal * (t_exp - tlim))
    except (ZeroDivisionError, OverflowError, ValueError):
        q_model_exp = tf.zeros_like(t_exp, dtype=tf.float32)

    if tf.size(t_exp) > 0 and tf.size(t_hyp) > 0:
        time_fraction = (tlim - t_hyp[-1]) / (t_exp[0] - t_hyp[-1] + epsilon)
        q_model_exp = tf.tensor_scatter_nd_update(
            q_model_exp, [[0]],
            [time_fraction * q_model_hyp[-1] + (1 - time_fraction) * q_model_exp[0]])

    production_post_buildup = tf.concat([q_model_hyp, q_model_exp], axis=0)
    daily_production = tf.concat([buildup_production, production_post_buildup], axis=0)
    monthly_volumes = daily_production * 30

    return monthly_volumes
