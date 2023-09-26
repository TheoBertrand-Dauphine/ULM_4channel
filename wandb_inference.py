import wandb
run = wandb.init()
artifact = run.use_artifact('tbertrand/ULM_4channel/run-5ss48pkc-history:v0', type='wandb-history')
artifact_dir = artifact.download()