import torch

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0" #!!
import numpy as np
from datetime import datetime
import argparse
from utils import _logger, set_requires_grad
from dataloader.dataloader import data_generator
from trainer.trainer import Trainer, model_evaluate
# from models.TC import TC
from models.model import Model
from config_files.circRNA_Configs import Config as Configs


def main(args, epochs):
    start_time = datetime.now()
    device = torch.device(args.device)
    experiment_description = args.experiment_description
    data_type = args.selected_dataset
    method = 'CRBPSA'

    logs_save_dir = args.logs_save_dir
    os.makedirs(logs_save_dir, exist_ok=True)


    configs = Configs()

    # ##### fix random seeds for reproducibility ########
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    #####################################################

    experiment_log_dir = os.path.join(logs_save_dir, experiment_description + f"_seed_{SEED}")
    os.makedirs(experiment_log_dir, exist_ok=True)

    # Logging
    log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
    logger = _logger(log_file_name)
    logger.debug("=" * 45)
    logger.debug(f'Dataset: {data_type}')
    logger.debug(f'Method:  {method}')
    logger.debug("=" * 45)

    # Load datasets
    train_dl, test_dl = data_generator(data_type,configs )
    logger.debug("Data loaded ...")



    # Load Model
    model = Model(configs).to(device)

    model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

    # Trainer
    Trainer(data_type, epochs, model, model_optimizer, train_dl,  test_dl, device, logger, experiment_log_dir)

    logger.debug(f"runing time is : {datetime.now()-start_time}")



if __name__ == "__main__":

    for data_name in ['AGO1', 'AGO2', 'AGO3', 'ALKBH5', 'AUF1', 'C17ORF85', 'C22ORF28', 'CAPRIN1', 'DGCR8', 'EIF4A3', 'EWSR1', 'FMRP', 'FOX2', 'FUS', 'FXR1', 'FXR2', 'HNRNPC', 'HUR', 'IGF2BP1', 'IGF2BP2', 'IGF2BP3', 'LIN28A', 'LIN28B', 'METTL3', 'MOV10', 'PTB', 'PUM2', 'QKI', 'SFRS1', 'TAF15', 'TDP43', 'TIA1', 'TIAL1', 'TNRC6', 'U2AF65', 'WTAP', 'ZC3H7B']:

            parser = argparse.ArgumentParser()
            parser.add_argument('--selected_dataset', default=data_name, type=str,
                                help='Dataset of choice：circRNAs')
            parser.add_argument('--experiment_description', default=data_name + '_Exp1', type=str,
                                help='Experiment Description')
            parser.add_argument('--seed', default=123, type=int,
                                help='seed value')
            parser.add_argument('--logs_save_dir', default='experiments_logs(CRBPSA)', type=str,
                                help='saving directory')
            parser.add_argument('--device', default='cuda', type=str,
                                help='cpu or cuda')
            args = parser.parse_args()

            main(args, epochs=20)


