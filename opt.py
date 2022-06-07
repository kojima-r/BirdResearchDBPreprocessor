import optuna
import argparse
import os
import copy
import subprocess, shlex
import json

def objective(trial,src_config,args):
    #x = trial.suggest_uniform('x', 0, 10)
    name="trial%04d"%(trial.number,)
    path=args.study_name+"/"+name
    config=copy.deepcopy(src_config)
    config["result_path"]=path
    ##
    config["learning_rate"]= trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    config["weight_decay"]= trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    optimizer = trial.suggest_categorical('optimizer', ['sgd', 'adam', 'adamW', 'rmsprop'])
    config["optimizer"]= optimizer
    config["batch_size"]= trial.suggest_int("batch_size", 8, 256)
    ##
    os.makedirs(path,exist_ok=True)
    conf_path=args.study_name+"/"+name+"/config.json"
    with open(conf_path, "w") as fp:
        json.dump(
            config,
            fp,
            ensure_ascii=False,
            indent=4,
            sort_keys=True,
        )
    cmd=["python","main.py","train","--loss_type",args.loss_type,"--max_epochs",args.max_epoch,"--config",conf_path]
    if args.gpu:
        cmd+=["--gpu",args.gpu]
    print("[EXEC]",cmd)
    #subprocess.Popen(shlex.split(cmd))
    subprocess.run(cmd)
    ##
    score=0
    try:
        result_path=path+"/result_valid.json"
        with open(result_path, "r") as fp:
            result=json.load(fp)
        score=result["val_loss"]
    except:
        score=1.0e10
    print("score:",score)
    return score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default=None, nargs="?", help="config json file"
    )
    parser.add_argument(
        "--study_name", type=str, default=None, help="config json file"
    )
    parser.add_argument(
        "--db", type=str, default="./study.db", help="config json file"
    )
    parser.add_argument(
        "--n_trials", type=int, default=100, help="config json file"
    )
    parser.add_argument(
        "--output", type=str, default="study.csv", help="output csv file"
    )
    parser.add_argument(
        "--gpu", type=str, default=None, help="gpu"
    )
    parser.add_argument(
        "--loss_type",   choices=['ce', 'focal', 'sigmoid', 'softmax'],default="ce")
    parser.add_argument(
        "--max_epoch",  type=str, default="100")
    args = parser.parse_args()

    if args.study_name is None:
        args.study_name="study_"+str(args.loss_type)+"_"+str(args.max_epoch)
    # start
    if os.path.exists(args.db):
        print("[REUSE]",args.db)
    else:
        print("[CREATE]",args.db)
    study = optuna.create_study(
        study_name=args.study_name,
        storage='sqlite:///'+args.db,
        load_if_exists=True)
    config={}
    if args.config:
        fp = open(args.config, "r")
        config.update(json.load(fp))

    study.optimize(lambda trial: objective(trial,config,args), n_trials=args.n_trials)
    #study.optimize(objective, timeout=120)
    
    outfile = args.output
    study.trials_dataframe().to_csv(outfile)

if __name__ == "__main__":
    main()

