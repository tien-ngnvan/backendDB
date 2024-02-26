import os
import time
from omegaconf import DictConfig
from hydra import compose, initialize_config_dir

from src.pipeline.data_ingestion import InitializeDatabase
from src.configs.configuration import ConfigurationManager

from src.reader.dir_reader import DirectoryReader
import glob

startTime_load = int(round(time.time() * 1000))

docs_files = glob.glob(os.path.abspath(os.curdir) + "/data/*.docx")
print("pdf_files: ", docs_files)
reader = DirectoryReader(
    input_files=docs_files
)
print("reader: ", reader.__dict__)
docs_files = reader.load_data()
if docs_files:
    print("Load pdf success")

# ------------------

# using hydra to load config
initialize_config_dir(version_base=None, config_dir=os.path.abspath(os.curdir) +  "/configs/")
cfg: DictConfig = compose("config.yaml")
print(cfg)

# load manager
manager = ConfigurationManager(config=cfg)

splitter_config = manager.get_sentence_splitter_config()
milvus_config = manager.get_milvus_config()
encoder_config = manager.get_cross_embed_config()
other_config = manager.get_orther_config()


# cần 2 params and config: của milvus và index -> done
pipeline = InitializeDatabase(
    splitter_config=splitter_config,
    milvus_config=milvus_config,
    encoder_config=encoder_config,
    other_config=other_config,
)
endTime_load = int(round(time.time() * 1000))
print(f"Time for load pipeline: {endTime_load - startTime_load} ms")


startTime_run= int(round(time.time() * 1000))
pipeline.main(documents=docs_files)
endTime_run= int(round(time.time() * 1000))
print(f"Time for load pipeline: {endTime_run - startTime_run} ms")