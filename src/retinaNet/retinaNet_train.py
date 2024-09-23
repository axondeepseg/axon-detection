import sys
import os
import distutils.core
import subprocess
from detectron2.engine import DefaultTrainer
import certifi
import ssl
import git
import pip

# Function to install Python packages programmatically using pip
def install_package(package_name):
    pip.main(['install', package_name])

# Install pyyaml programmatically
install_package('pyyaml==5.1')

# Clone the Detectron2 repository programmatically using GitPython
repo_url = "https://github.com/facebookresearch/detectron2"
local_dir = "./detectron2"

if not os.path.exists(local_dir):
    git.Repo.clone_from(repo_url, local_dir)
else:
    print(f"Repository already exists in {local_dir}")

# Load setup.py and install dependencies
dist = distutils.core.run_setup(os.path.join(local_dir, 'setup.py'))

# Install the dependencies from the setup.py file
for dep in dist.install_requires:
    install_package(dep)

# Add Detectron2 to the system path
sys.path.insert(0, os.path.abspath(local_dir))


# Tells urllib to use the certificates of certifi
def create_context():
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.load_verify_locations(certifi.where())
    return context

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("../src/data-coco/annotations/json_annotation_train.json")
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)

# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg.MODEL.DEVICE = "cpu"  # Force CPU usage
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

print('\n METADATA CATALOG')
print(list(detectron2.data.MetadataCatalog))

# For SSL Certification
ssl._create_default_https_context = create_context()

trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()