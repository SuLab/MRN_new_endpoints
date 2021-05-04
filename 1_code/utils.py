import json
import pickle
from pathlib import Path, PosixPath
import pandas as pd

DEFAULT_PARAMS = '../0_data/manual/data_param.json'


def process_param_paths(params):
    """
    Converts paths in params to pathlib.Path objects. Paths are
    defined by keys ending in '_dir' or '_file'.
    """
    for k, v in params.items():
        if k.endswith('_dir') or k.endswith('_file'):
            params[k] = Path(v).resolve()

    return params


def validate_params(params):
    """Ensure required keys are present in params"""

    for k in ['nw_dir', 'hyperp_dir', 'model_dir', 'biolink', 'target_node',
              'target_edge', 'semantic_info_file']:
        assert k in params.keys(), "'{}' not in params".format(k)

    for k, v in params.items():
        if k.endswith('_dir') or k.endswith('_file'):
            assert type(params[k]) == type(Path('.')), "{0!r:} must be of type {1:}".format(v, type(Path('.')))

def read_json(path):
    """Simple 1-liner to read any .json file"""
    with open(path, 'r') as fin:
        data = json.load(fin)

    return data


def read_param_file(param_file=DEFAULT_PARAMS):
    """
    Direct read of a param.json file. Process paths, but no validation.
    """
    params = read_json(param_file)

    return process_param_paths(params)


def load_params(param_file=DEFAULT_PARAMS):
    """
    If param_file is a path, reads in param file. Validates and returns params
    whether read or passed.
    """

    # Read in the params file if a param dict is not passed
    if type(param_file) == str or type(param_file) == PosixPath:
        params = read_param_file(param_file)
    else:
        params = param_file

    # Validate the params
    validate_params(params)
    return params


def load_network(param_file=DEFAULT_PARAMS):
    """
    Load noad and edge files for network

    :param param_file: str, name of the file containing the data paramateres
        defaults to `data_param.json`
    :returns: (nodes, edges) data frames for network
    """
    params = load_params(param_file)

    tail = '_biolink.csv' if params['biolink'] else '.csv'
    nodes = pd.read_csv(params['nw_dir'].joinpath('nodes'+tail), dtype=str)
    edges = pd.read_csv(params['nw_dir'].joinpath('edges'+tail), dtype=str)

    return nodes, edges


def read_features(path):
    """
    Read a features file

    :param path: path to features file to read
    :return: features, list of feature names
    """
    with open(path, 'r') as fin:
        features = fin.read().rstrip('\n').split('\n')
    return features


def load_targets(param_file):
    """
    Read a data_param.json file and return the target node type and target
    edge type for the model generated.
    """

    params = load_params(param_file)

    return params['target_node'], params['target_edge']


def load_hyperparameters(param_file=DEFAULT_PARAMS):
    """
    Load previously tuned hyperparamerters

    :param param_file: str, name of the file containing the data paramateres
        defaults to `data_param.json`
    :returns: (hyperparam, features) hyperparam: dict with key value pairs of
        hyperparamers and their values
        features: list of features used in model
    """
    params = load_params(param_file)
    hyperp_dir = params['hyperp_dir']

    tune_features = read_features(hyperp_dir.joinpath('kept_features.txt'))

    with open(hyperp_dir.joinpath('best_param.pkl'), 'rb') as fin:
        hyperparam = pickle.load(fin)

    return hyperparam, tune_features


def load_model(param_file=DEFAULT_PARAMS):
    """
    Load previously trained model

    :param param_file: str, name of the file containing the data paramateres
        defaults to `data_param.json`

    :returns: (model, features, coef) model: sklearn pipline, pre-trained
        features: list of features used in model
        coef: DataFrame with feature names and coefficient values
    """
    params = load_params(param_file)
    model_dir = params['model_dir']

    with open(model_dir.joinpath('model.pkl'), 'rb') as fin:
        model = pickle.load(fin)

    features = read_features(model_dir.joinpath('feature_order.txt'))

    coef = pd.read_csv(model_dir.joinpath('coef.csv'))
    return model, features, coef


def validate_outdir(outdir):
    """Ensure outdir is a valid path, and create if does not exist"""

    if type(outdir) == str:
        outdir = Path(outdir).resolve()

    assert type(outdir) == type(Path('.'))

    if not outdir.exists():
        outdir.mkdir(parents=True)

    return outdir


def save_features(features, outdir, filename='kept_features.txt'):
    """
    Saves a list of selected features that can be reused in future runs.

    :param features: list of features to save
    :param outdir: Path to save location
    """
    outdir = validate_outdir(outdir)

    with open(outdir.joinpath(filename), 'w') as fout:
        for f in features:
            fout.write(f + '\n')


def save_hyperparam(hyperparam, tune_features, outdir):
    """
    Save the hyperparameters using naming convention for future loading via
    these utils.

    :param hyperparam: the dictionary of hyperparamters used for the model
    :param tune_features: list of features used in hyperparameter tuning
    :param outdir: Path to save location
    """

    outdir = validate_outdir(outdir)
    with open(outdir.joinpath('best_param.pkl'), 'wb') as fout:
        pickle.dump(hyperparam, fout)

    save_features(tune_features, outdir)


def save_model(model, feature_order, coefs, outdir):
    """
    Save the model to hard disk using naming convetion for future loading
    via these utils

    :param model: sklearn model to save as pickle
    :param feature_order: feature names in column order used for the model
    :param coefs: DataFrame of feature names and coefficient value
    :param outdir: Path to save location
    """

    outdir = validate_outdir(outdir)

    save_features(feature_order, outdir, 'feature_order.txt')

    with open(outdir.joinpath('model.pkl'), 'wb') as fout:
        pickle.dump(model, fout)

    coefs.to_csv(outdir.joinpath('coef.csv'), index=False)


def get_model_transformations(model, features):
    """
    Extracts the feature transofrmation information from the model

    :param model: the model
    :param features: the features in the model in proper order

    :returns: (ini_means, ma_scale) (dict, dict)
              {key: feature name, value: initial_feature_mean},
              {key: feature name, value: max_abls_scale_factor}
    """

    msat = model[0]
    max_abs = model[1]

    ini_means = {f: m for f, m in zip(features, msat.initial_mean_)}
    ma_scale = {f: m for f, m in zip(features, max_abs.scale_)}

    return ini_means, ma_scale


def load_semantic_info(param_file=DEFAULT_PARAMS):
    """
    Loads data about network semantics from file location in param_file
    """

    filename = load_params(param_file)['semantic_info_file']
    return pd.read_csv(filename)


def load_path_stats(param_file):
    """
    Loads stats on the path from a param file.
    """
    params = load_params(param_file)

    return pd.read_csv(params['path_stats_dir'].joinpath('pathcount_stats.csv'))

