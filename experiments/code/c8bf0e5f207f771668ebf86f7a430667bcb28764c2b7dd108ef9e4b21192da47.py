"""Explicit deterministic native-training entry point; shares the original trainer."""
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def configure():
    # Set before CUDA context creation; never change global system or GPU settings.
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    from train import train as trainer
    from common.data_contracts import identity,sha256_file
    if getattr(trainer.training_contract,'deterministic_native_contract',False):return
    original=trainer.training_contract
    def training_contract(config,world,hold_files):
        contract=original(config,world,hold_files);contract.pop('id')
        contract['execution']=dict(deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                                   cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'],
                                   cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
                                   code={'train/deterministic_train.py':sha256_file(Path(__file__))})
        contract['id']=identity(contract)
        return contract
    training_contract.deterministic_native_contract=True
    trainer.training_contract=training_contract


if __name__=='__main__':
    configure()
    from train.train import main
    main()
