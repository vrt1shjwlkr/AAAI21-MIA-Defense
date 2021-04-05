from purchase_models import *
from purchase_distillation import *
from purchase_advtune import *

'''
check https://github.com/BayesWatch/pytorch-moonshine/blob/master/main.py for distillation loss function
'''

def main():
    parser = argparse.ArgumentParser(description='distillation experiments')

    parser.add_argument('--debug-', type=str, default='MEDIUM', help='debug level (default: MEDIUM)')
    parser.add_argument('--defence', type=str, default='distillation', help='MIA mitigation type (default: distillation)')
    
    parser.add_argument('--train-len', type=int, default=20000, help='Training data len (default: 20000)')
    parser.add_argument('--ref-len', type=int, default=20000, help='reference/distillation data len (default: 20000)')
    
    
    parser.add_argument('--batch-size', type=int, default=64, help='batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=0.0005, help='normal learning rate (default: 0.0005)')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor (default: 0.5)')
    parser.add_argument('--tr-epochs', type=int, default=100, help='# of training epochs for adv_regularization | # of training epochs for non-private training for distillation (default: 100)')
    parser.add_argument('--schedule', type=list, default=[30,80], help='training schedule (default: 30,80)')


    parser.add_argument('--distil-batch-size', type=int, default=64, help='batch size (default: 32)')
    parser.add_argument('--distil-lr', type=float, default=0.0005, help='normal learning rate (default: 0.1)')
    parser.add_argument('--distil-gamma', type=float, default=0.5, help='learning rate decay factor (default: 0.5)')
    parser.add_argument('--distil-epochs', type=int, default=200, help='# distillation training epochs (default: 100)')
    parser.add_argument('--distil-schedule', type=list, default=[50,90,150], help='distillation training schedule (default: 50,90,150)')
    parser.add_argument('--distil-temp', type=float, default=1.0, help='temperature of softmax (default: 1.0)')

    
    parser.add_argument('--alpha', type=int, default=0, help='adversarial tuning factor (default: 0)')
    parser.add_argument('--advtune-batch-size', type=int, default=64, help='batch size (default: 64)')
    parser.add_argument('--advtune-lr', type=float, default=0.001, help='normal learning rate (default: 0.001)')
    parser.add_argument('--advtune-gamma', type=float, default=0.1, help='learning rate decay factor (default: 0.1)')
    parser.add_argument('--advtune-tr-epochs', type=int, default=100, help='# of training epochs for adv_regularization | # of training epochs for non-private training for distillation (default: 100)')
    parser.add_argument('--advtune-schedule', type=list, default=[25,80], help='training schedule (default: 25,80)')
    

    parser.add_argument('--attack-epochs', type=int, default=200, help='current epoch number (default: 200)')
    parser.add_argument('--n-classes', type=int, default=100, help='Number of classes (default: 100)')
    
    args = parser.parse_args()
    try:
        args.distil_epochs=''.join(args.distil_epochs)
        args.distil_epochs=args.distil_epochs.split(',')
    except:
        pass
    
    args.distil_epochs=list(map(int,args.distil_epochs))
    
    assert torch.cuda.is_available(), 'No GPUs available... Exiting!'
    
    use_cuda=True

    if args.defence=='distillation':
        print('Processing distillation defence')
        distillation_defense(args.train,args.defense,args.evaluate,args.tr_len,args.ref_len,use_cuda,args.batch_size,args.lr,args.schedule,args.gamma,args.tr_epochs,args.distil_batch_size,args.distil_lr,args.distil_schedule,args.distil_gamma,args.distil_epochs,args.distil_temp,args.at_lr,args.at_schedule,args.at_gamma,args.at_epochs,args.n_classes)
    
    elif args.defence=='advtune':
        print('Processing adversarial regularization defence')
        advtune_defense(args.train,args.evaluate,args.tr_len,args.ref_len,args.use_cuda,args.batch_size,args.alpha,args.lr,args.schedule,args.gamma,args.tr_epochs,args.at_lr,args.at_schedule,args.at_gamma,args.at_epochs,args.n_classes):
    
    else:
        assert False, 'Unknown defence... Exiting!'

if __name__ == '__main__':
    main()