import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from datasets import load_from_disk, load_dataset
import torch
import yaml
import torch.distributed as dist
# from lmdeploy import pipeline, TurbomindEngineConfig