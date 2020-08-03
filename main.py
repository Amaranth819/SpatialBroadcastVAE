import argparse
import torch
import os
import torchvision
from chair import Chairs
from sbvae import SBVAE, weight_init
from tensorboardX import SummaryWriter

def total_param_num(net):
    num = 0
    for param in net.parameters():
        num += param.numel()
    return num

def train(config):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Network
    net = SBVAE().to(device)

    full_model_path = config.model_path + config.model_name
    if os.path.exists(full_model_path):
        ckpt = torch.load(full_model_path)
        net.load_state_dict(ckpt)
        print('Load the pretrained model successfully!')
    else:
        if not os.path.exists(config.model_path):
            os.makedirs(config.model_path)
        weight_init(net)
        print('First time training!')

    net.train()

    print('Number of parameters', total_param_num(net))

    # Optimizer
    optim = torch.optim.Adam(net.parameters(), config.lr, betas = [0.5, 0.999])

    # Summary
    summary = SummaryWriter(config.summary_path)

    # Dataset
    trainset = Chairs(config.dataset_root_path)
    trainset_loader = torch.utils.data.DataLoader(
        trainset, 
        batch_size = config.batch_size,
        shuffle = True)

    testset = Chairs(config.dataset_root_path, is_train = False)
    testset_loader = torch.utils.data.DataLoader(
        testset,
        batch_size = config.batch_size,
        shuffle = False)
    testset_size = len(testset_loader)

    print('%d training batches.' % len(trainset_loader))
    print('%d testing batches.' % testset_size)

    # Train
    counter, eval_counter = 1, 1
    for e in range(config.epoch):
        avg_loss, avg_mse, avg_kl = 0, 0, 0
        for idx, x in enumerate(trainset_loader):
            x = x.to(device)
            x.requires_grad_(True)

            # Update the network
            optim.zero_grad()
            mu, logvar, recon_x = net(x)
            mse, kl = net.loss_func(x, recon_x, mu, logvar)
            loss = mse + kl
            loss.backward()
            optim.step()

            # Output
            print('[Epoch %d|Train Batch %d] MSE = %.6f | KL = %.6f' % (e, idx, mse.item(), kl.item()))

            avg_loss += loss.item()
            avg_mse += mse.item()
            avg_kl += kl.item()

            if idx % 20 == 19:
                summary.add_scalar('Train/AVG_Loss', avg_loss / 50, counter)
                summary.add_scalar('Train/AVG_MSE', avg_mse / 50, counter)
                summary.add_scalar('Train/AVG_KL', avg_kl / 50, counter)

                counter += 1

        # Evaluation
        if e % 1 == 0:
            net.eval()
            with torch.no_grad():
                avg_eval_loss, avg_eval_mse, avg_eval_kl = 0, 0, 0
                show_x = None
                for idx, x in enumerate(testset_loader):
                    x = x.to(device)
                    mu, logvar, recon_x = net(x)
                    if idx == 0:
                        show_x = recon_x
                    mse, kl = net.loss_func(x, recon_x, mu, logvar)
                    loss = mse + kl

                    print('[Epoch %d|Test Batch %d] MSE = %.6f | KL = %.6f' % (e, idx, mse.item(), kl.item()))
                    avg_eval_loss += loss.item()
                    avg_eval_mse += mse.item()
                    avg_eval_kl += kl.item()

                avg_eval_loss / testset_size
                avg_eval_mse / testset_size
                avg_eval_kl / testset_size

                summary.add_scalar('Test/AVG_Loss', avg_eval_loss / testset_size, eval_counter)
                summary.add_scalar('Test/AVG_MSE', avg_eval_mse / testset_size, eval_counter)
                summary.add_scalar('Test/AVG_KL', avg_eval_kl / testset_size, eval_counter)

                eval_counter += 1

                show_x = torchvision.utils.make_grid(show_x, padding = 2, normalize = True).detach().cpu().numpy()
                summary.add_image('Eval/Recon_Epoch%d' % e, show_x, e)
                    

        # Save the model
        if e % 1 == 0:
            torch.save(net.state_dict(), full_model_path)

        if e % 10 == 9:
            torch.save(net.state_dict(), config.model_path + 'e%d_' % e + config.model_name)

    summary.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_train', type = bool, default = True)
    parser.add_argument('--dataset_root_path', type = str, default = './rendered_chairs/')
    parser.add_argument('--model_path', type = str, default = './model/')
    parser.add_argument('--model_name', type = str, default = 'model.pkl')
    parser.add_argument('--summary_path', type = str, default = './summary/')
    parser.add_argument('--batch_size', type = int, default = 32)
    parser.add_argument('--epoch', type = int, default = 100)
    parser.add_argument('--bs', type = int, default = 32)
    parser.add_argument('--lr', type = float, default = 1e-4)

    config = parser.parse_args()
    train(config)
