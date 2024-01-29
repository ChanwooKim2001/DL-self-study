import argparse #터미널에서 코드 수행하도록 도와주는 파이썬 라이브러리리
# 아나콘다, 파이참 같은 창을 실제로 켜지 않고도 터미널에서도 이 변수들을 다 컨트롤 가능
import utils
import datasets
from tsne import tsne

if __name__ == "__main__":
    
    # argparse 사용법: 이 양식을 그대로 따서 두고 수정하면 됨됨
    # 1. 선언
    parser = argparse.ArgumentParser(description='CIFAR10 image classification') 
    # 2. control하고자 하는 변수들을 쭉 입력해주면 됨
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--num_epochs', default=51, type=int, help='training epoch')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--l2', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--model_name', default='resnet18', type=str, help='model name')
    parser.add_argument('--pretrained', default=None, type=str, help='model path')
    parser.add_argument('--train', default='train', type=str, help='train and eval')
    args = parser.parse_args()
    print(args)

    if args.train == 'train':
        # 데이터 불러오기
        trainloader, testloader = datasets.dataloader(args.batch_size, 'train')
        # 모델 불러오기 및 학습하기
        learning = utils.SupervisedLearning(trainloader, testloader, args.model_name, args.pretrained)
        print('Completed loading your datasets.')
        learning.train(args.num_epochs, args.lr, args.l2)
    else:
        trainloader, testloader = datasets.dataloader(args.batch_size, 'eval')
        learning = utils.SupervisedLearning(trainloader, testloader, args.model_name, args.pretrained)
        print('Completed loading your datasets.')
        train_acc = learning.eval(trainloader)
        test_acc = learning.eval(testloader)
        print(f' Train Accuracy: {train_acc}, Test Accuraccy: {test_acc}')

        # t-SNE graph
        tsne(testloader, args.model_name, args.pretrained)

