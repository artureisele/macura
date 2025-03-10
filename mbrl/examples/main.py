import hydra
import numpy as np
import omegaconf
import torch

import mbrl.algorithms.mbpo as mbpo
import mbrl.algorithms.macura as macura
import mbrl.algorithms.m2ac as m2ac
import mbrl.algorithms.infoprop_dyna as infoprop_dyna

import mbrl.util.env

@hydra.main(config_path="conf", config_name="main", version_base="1.1")
def run(cfg: omegaconf.DictConfig):
    algo_name = cfg.algorithm.name
    task = cfg.overrides.env
    print("Running "+algo_name+" for the "+task+" task.")


    env, term_fn, reward_fn = mbrl.util.env.EnvHandler.make_env(cfg, test_env=False)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.algorithm.name == "mbpo":
        # test_env is used for evaluating the model after each training epoch but it is not clear why not env is used
        test_env, *_ = mbrl.util.env.EnvHandler.make_env(cfg, test_env=True)
        return mbpo.train(env, test_env, term_fn, cfg)
    if cfg.algorithm.name == "m2ac":
        test_env, *_ = mbrl.util.env.EnvHandler.make_env(cfg, test_env=True)
        return m2ac.train(env, test_env, term_fn, cfg)
    if cfg.algorithm.name == "macura":
        test_env, *_ = mbrl.util.env.EnvHandler.make_env(cfg, test_env=True)
        test_env2, *_ = mbrl.util.env.EnvHandler.make_env(cfg, test_env=True)
        return macura.train(env, test_env,test_env2 ,term_fn, cfg)
    if cfg.algorithm.name == "infoprop_dyna":
        test_env, *_ = mbrl.util.env.EnvHandler.make_env(cfg, test_env=True)
        return infoprop_dyna.train(env, test_env, term_fn, cfg)

if __name__ == "__main__":
        run()