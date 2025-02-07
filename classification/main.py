# coding:utf8
from config import opt
import os
import torch as t
import torchvision
import models
from data.dataset import DogCat
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchnet import meter
from tensorboardX import SummaryWriter
# from utils.visualize import Visualizer
# import ipdb


def test(**kwargs):
    opt.parse(kwargs)
    # ipdb.set_trace()
    # configure model
    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()

    # data
    train_data = DogCat(opt.test_data_root, test=True)
    test_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    results = []
    for ii, (data, path) in enumerate(test_dataloader):
        input_data = t.autograd.Variable(data, volatile=True)
        if opt.use_gpu:
            input_data = input_data.cuda()
        score = model(input_data)
        probability = t.nn.functional.softmax(score)[:, 0].data.tolist()
        # label = score.max(dim = 1)[1].data.tolist()

        batch_results = [(path_, probability_) for path_, probability_ in zip(path, probability)]

        results += batch_results
    write_csv(results, opt.result_file)

    return results


def write_csv(results, file_name):
    import csv
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows(results)


def train(**kwargs):
    opt.parse(kwargs)
    # vis = Visualizer(opt.env)
    writer = SummaryWriter()

    # step1: configure model
    model = getattr(models, opt.model)()

    res = model(t.autograd.Variable(t.Tensor(1, 3, opt.input_size, opt.input_size), requires_grad=True))
    writer.add_graph(model, res)

    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()

    # step2: data
    train_data = DogCat(opt.train_data_root, train=True)
    val_data = DogCat(opt.train_data_root, train=False)
    train_dataloader = DataLoader(train_data, opt.batch_size,
                                  shuffle=True, num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, opt.batch_size,
                                shuffle=False, num_workers=opt.num_workers)

    # step3: criterion and optimizer
    criterion = t.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = t.optim.Adam(model.parameters(), lr=lr, weight_decay=opt.weight_decay)

    # step4: meters
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e100

    # train
    for epoch in range(opt.max_epoch):

        loss_meter.reset()
        confusion_matrix.reset()

        for ii, (data, label) in enumerate(train_dataloader):

            # train model 
            input_data = Variable(data)
            target = Variable(label)
            if opt.use_gpu:
                input_data = input_data.cuda()
                target = target.cuda()

            optimizer.zero_grad()
            score = model(input_data)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()

            # meters update and visualize
            loss_meter.add(loss.data[0])
            confusion_matrix.add(score.data, target.data)

            """
            if ii % opt.print_freq == opt.print_freq - 1:
                vis.plot('loss', loss_meter.value()[0])
                # 进入debug模式
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()
            """
        model.save()

        # validate and visualize
        val_cm, val_accuracy = val(model, val_dataloader)

        writer.add_scalar('data/train_loss', loss_meter.value()[0], epoch)
        # writer.add_scalar('data/train_cm', confusion_matrix.value()[0], epoch)
        writer.add_scalar('data/val_acc', val_accuracy, epoch)
        # writer.add_scalar('data/val_cm', val_cm.value()[0], epoch)
        print("epoch:{epoch},lr:{lr},loss:{loss},val_acc:{val_accuracy}".format(
            epoch=epoch, loss=loss_meter.value()[0], val_accuracy=val_accuracy, lr=lr))

        # for name, param in model.named_parameters():
        #     writer.add_histogram(name, param.data, epoch)

        # out = torchvision.utils.make_grid(data)
        # writer.add_image('Image', out, epoch)

        # update learning rate
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        writer.add_scalar('data/lr', lr, epoch)
        previous_loss = loss_meter.value()[0]
    writer.export_scalars_to_json("./checkpoints/all_scalars.json")
    writer.close()


def val(model, dataloader):
    """
    计算模型在验证集上的准确率等信息
    """
    model.eval()
    confusion_matrix = meter.ConfusionMeter(2)
    for ii, data in enumerate(dataloader):
        input, label = data
        val_input = Variable(input, volatile=True)
        val_label = Variable(label.type(t.LongTensor), volatile=True)
        if opt.use_gpu:
            val_input = val_input.cuda()
            val_label = val_label.cuda()
        score = model(val_input)
        confusion_matrix.add(score.data.squeeze(), label.type(t.LongTensor))

    model.train()
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    return confusion_matrix, accuracy


def helper():
    """
    打印帮助的信息： python file.py help
    """

    print('''
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:'''.format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)


if __name__ == '__main__':
    import fire

    fire.Fire()
