import dataloaders as dl
import rotnetfuncs as rnf
import model_classes as models
import importlib as impL
# impL.reload(rnf)
# impL.reload(models)
# impL.reload(dl)
def try_things_st01(rot_deg_inc = 1, network_id = 1):
    rootFold="/home/doga/PycharmProjects/objectLocalization"
    dataset_name = "mnist"
    data = dl.load_datasets(dataset_name, rootFold=rootFold, rot_deg_inc=rot_deg_inc)
    classCount=int(360/rot_deg_inc)
    print("rot_deg_inc {}, class count = {}".format(rot_deg_inc, classCount))
    if network_id==0:
        cm = models.MLP(nb_filters=[64, 64], kernel_size=[3, 3], hidCounts=[128], rot_deg_inc=rot_deg_inc, classCount=classCount, network_id=network_id)
    elif network_id==1:
        cm = models.MLP(nb_filters=[64, 64], kernel_size=[3, 3], hidCounts=[128], rot_deg_inc=rot_deg_inc, classCount=classCount, network_id=network_id)
    return cm, data

def try_things_st02(cm, data, epochCnt=20, batch_size=128):
    cm.train_model(data["tr"], epochCnt=epochCnt, batch_size=batch_size)

def try_things_st03(cm, data, batch_size=128):
    cm.evaluate_model(data["te"], batch_size=batch_size)